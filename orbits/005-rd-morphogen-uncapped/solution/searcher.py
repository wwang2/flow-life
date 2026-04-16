#!/usr/bin/env python3
"""
Two-phase fission-detecting searcher for RD Morphogen V2.

Phase 1: Survival filter -- run 500 steps, check mass > 50% initial
Phase 2: Fission proxy at 3000-5000 steps -- detect spatially distinct
         daughters matching the evaluator's Tier 4 detection logic.

Search strategy:
  - Start from orbit 003's proven params
  - Grid search over key fission-related parameters:
    * w_inner (repulsion strength): stronger drives pinching
    * morph_coupling: how much morphogen guides growth
    * soft_drift_threshold: how much mass drift to allow
  - Fitness = fission_proxy (primary) + 0.2 * stability (secondary)
"""

from __future__ import annotations

import sys
import time

import numpy as np
import scipy.ndimage

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from substrate import RDMorphogenSubstrateV2


def create_fission_blob(
    grid_size: int = 256,
    seed: int = 42,
    sigma_y: float = 18.0,
    sigma_x: float = 12.0,
    center_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Create elongated Gaussian blob biased for fission."""
    rng = np.random.RandomState(seed)
    H, W = grid_size, grid_size
    cy = H // 2 + center_offset[0]
    cx = W // 2 + center_offset[1]
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    blob = np.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    noise = rng.randn(H, W) * 0.03
    state = np.clip(blob + noise, 0.0, 1.0).astype(np.float32)
    return state[np.newaxis]  # (1, H, W)


def detect_components(state_np, parent_init_mass, grid_size=256):
    """Detect spatially distinct components -- mirrors evaluator logic exactly."""
    mass_field = state_np.sum(axis=0) if state_np.ndim == 3 else state_np.copy()
    pk = mass_field.max()
    if pk < 1e-12:
        return [], mass_field

    BINARIZE_THRESHOLD = 0.05
    MIN_DAUGHTER_MASS_FRAC = 0.10
    TOROIDAL_PAD = 15

    binary = (mass_field > BINARIZE_THRESHOLD * pk).astype(np.int32)
    H, W = binary.shape
    padded = np.pad(binary, TOROIDAL_PAD, mode="wrap")
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    labeled, _ = scipy.ndimage.label(padded, structure=struct)
    labeled = labeled[TOROIDAL_PAD:TOROIDAL_PAD + H, TOROIDAL_PAD:TOROIDAL_PAD + W]

    comps = []
    for lbl in np.unique(labeled):
        if lbl == 0:
            continue
        pmask = labeled == lbl
        cmass = float(mass_field[pmask].sum())
        if cmass < MIN_DAUGHTER_MASS_FRAC * parent_init_mass:
            continue
        ys, xs = np.where(pmask)
        cy = float(ys.mean())
        cx = float(xs.mean())
        comps.append({
            "label": int(lbl),
            "mass": cmass,
            "centroid": np.array([cy, cx]),
            "mask": pmask,
        })
    return comps, mass_field


def toroidal_distance(c1, c2, H=256, W=256):
    """Minimum-image distance on a torus."""
    dy = abs(float(c1[0]) - float(c2[0]))
    dx = abs(float(c1[1]) - float(c2[1]))
    dy = min(dy, H - dy)
    dx = min(dx, W - dx)
    return (dy**2 + dx**2) ** 0.5


def fission_score(
    substrate: RDMorphogenSubstrateV2,
    pattern: np.ndarray,
    params: dict,
    grid_size: int = 256,
    phase1_steps: int = 500,
    phase2_start: int = 2000,
    phase2_end: int = 6000,
    check_interval: int = 100,
) -> tuple[float, float, np.ndarray]:
    """Evaluate a candidate for fission potential.

    Returns (fission_score, stability_score, best_state_for_saving).

    Phase 1: Run phase1_steps, check survival (mass > 50% initial).
    Phase 2: Continue to phase2_end, checking for daughter components every
             check_interval steps. Score = max daughters detected at any check.
    """
    substrate.reset()
    parent_init_mass = float(pattern.sum())
    if parent_init_mass < 1e-8:
        return 0.0, 0.0, pattern

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor(pattern, dtype=torch.float32, device=device)
        substrate.to(device)
    else:
        return 0.0, 0.0, pattern

    masses = []
    best_daughter_count = 0
    best_state = None
    # Track persistent daughters via bucket tracking (like evaluator)
    candidates = {}  # bucket -> {"streak": int}
    MIN_CENTROID_SEP = 20
    PERSISTENCE_FRAMES = 3
    confirmed_daughters = 0

    with torch.no_grad():
        total_steps = phase2_end
        for step in range(1, total_steps + 1):
            t = substrate.update_step(t, params)
            t = t.clamp(0.0)  # soft floor only
            current_mass = float(t.sum().item())
            masses.append(current_mass)

            # Phase 1 check: survival at phase1_steps
            if step == phase1_steps:
                if current_mass < 0.5 * parent_init_mass:
                    return 0.0, 0.0, pattern
                # Mass creation check (evaluator kills at 3x)
                if current_mass > 2.8 * parent_init_mass:
                    return 0.0, 0.0, pattern

            # Phase 2: check for fission
            if step >= phase2_start and step % check_interval == 0:
                state_np = t.detach().cpu().numpy()

                comps, mf = detect_components(state_np, parent_init_mass, grid_size)
                if len(comps) < 2:
                    # Reset streaks for all candidates
                    for b in candidates:
                        candidates[b]["streak"] = 0
                    continue

                # Find parent (largest component)
                parent_idx = max(range(len(comps)), key=lambda i: comps[i]["mass"])
                parent_centroid = comps[parent_idx]["centroid"]

                frame_buckets = set()
                for i, comp in enumerate(comps):
                    if i == parent_idx:
                        continue
                    sep = toroidal_distance(comp["centroid"], parent_centroid, grid_size, grid_size)
                    if sep < MIN_CENTROID_SEP:
                        continue

                    bucket = (round(comp["centroid"][0] / 5.0), round(comp["centroid"][1] / 5.0))
                    matched = None
                    for pb in list(candidates.keys()):
                        if abs(bucket[0] - pb[0]) <= 2 and abs(bucket[1] - pb[1]) <= 2:
                            matched = pb
                            break
                    if matched is not None:
                        candidates[matched]["streak"] += 1
                        frame_buckets.add(matched)
                        if candidates[matched]["streak"] >= PERSISTENCE_FRAMES:
                            if not candidates[matched].get("confirmed"):
                                confirmed_daughters += 1
                                candidates[matched]["confirmed"] = True
                    else:
                        candidates[bucket] = {"streak": 1}
                        frame_buckets.add(bucket)

                for b in list(candidates.keys()):
                    if b not in frame_buckets and not candidates[b].get("confirmed"):
                        candidates[b]["streak"] = 0

                n_daughters = len(comps) - 1  # rough count
                if n_daughters > best_daughter_count:
                    best_daughter_count = n_daughters
                    best_state = state_np.copy()

    # Stability score from mass trajectory
    m = np.array(masses)
    mu = m.mean()
    if mu < 1e-8:
        stability = 0.0
    else:
        cv = m.std() / mu
        stability = max(0.0, 1.0 - cv)

    # Fission score: combination of confirmed persistent daughters + rough count
    fission = confirmed_daughters + 0.3 * best_daughter_count
    final_state = best_state if best_state is not None else t.detach().cpu().numpy()

    return float(fission), float(stability), final_state


def search_patterns(
    grid_size: int = 256,
    n_candidates: int = 20,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Search for fission-capable patterns with parameter variations.

    Strategy:
      1. Start from orbit 003's default params
      2. Try variations on key parameters:
         - w_inner: [-0.6, -0.8, -1.0, -1.2] (repulsion strength)
         - morph_coupling: [0.05, 0.08, 0.12] (morphogen influence)
         - soft_drift_threshold: [0.10, 0.15, 0.20] (mass drift tolerance)
      3. Try different initial blob shapes
      4. Fitness = fission_proxy + 0.2 * stability
    """
    rng = np.random.default_rng(seed)
    substrate = RDMorphogenSubstrateV2(grid_size=grid_size)
    base_params = substrate.get_default_params()

    best_pattern = None
    best_score = -1.0
    best_params = None
    best_fission = 0.0
    best_stability = 0.0

    # Parameter grid
    w_inner_vals = [-0.6, -0.8, -1.0, -1.2]
    morph_coupling_vals = [0.05, 0.08, 0.12]
    drift_thresh_vals = [0.10, 0.15, 0.25]

    # Blob shape variations
    blob_shapes = [
        (18.0, 12.0),  # elongated (fission-biased)
        (22.0, 10.0),  # very elongated
        (15.0, 15.0),  # circular
        (20.0, 8.0),   # extremely elongated
        (16.0, 14.0),  # slightly elongated
    ]

    print(f"Searching {n_candidates} candidates (seed={seed})...", flush=True)
    t_start = time.time()

    candidate_idx = 0
    for w_i_idx, w_inner in enumerate(w_inner_vals):
        if candidate_idx >= n_candidates:
            break
        for mc_idx, morph_c in enumerate(morph_coupling_vals):
            if candidate_idx >= n_candidates:
                break

            # Pick drift threshold and blob shape based on candidate index
            dt_idx = candidate_idx % len(drift_thresh_vals)
            blob_idx = candidate_idx % len(blob_shapes)

            params = dict(base_params)
            params["w_inner"] = w_inner
            params["morph_coupling"] = morph_c
            params["soft_drift_threshold"] = drift_thresh_vals[dt_idx]

            sy, sx = blob_shapes[blob_idx]
            # Add some random offset
            oy = int(rng.integers(-5, 5))
            ox = int(rng.integers(-5, 5))

            pattern = create_fission_blob(
                grid_size, seed + candidate_idx * 7,
                sigma_y=sy, sigma_x=sx,
                center_offset=(oy, ox),
            )

            label = f"w_i={w_inner:.1f} mc={morph_c:.2f} dt={drift_thresh_vals[dt_idx]:.2f} blob={sy:.0f}x{sx:.0f}"

            fission, stability, final_state = fission_score(
                substrate, pattern, params, grid_size,
            )

            score = fission + 0.2 * stability
            candidate_idx += 1

            if score > best_score:
                best_score = score
                best_pattern = pattern
                best_params = dict(params)
                best_fission = fission
                best_stability = stability
                print(f"  [{candidate_idx}] {label} fission={fission:.2f} stab={stability:.3f} score={score:.3f} ** NEW BEST **",
                      flush=True)
            else:
                print(f"  [{candidate_idx}] {label} fission={fission:.2f} stab={stability:.3f} score={score:.3f}",
                      flush=True)

            elapsed = time.time() - t_start
            if elapsed > 600:  # 10 min budget for search
                print(f"  Time budget reached ({elapsed:.0f}s), stopping search", flush=True)
                break

    if best_pattern is None:
        best_pattern = create_fission_blob(grid_size, seed)
        best_params = base_params

    print(f"\nBest: fission={best_fission:.2f} stability={best_stability:.3f} total={best_score:.3f}", flush=True)
    print(f"Best params: w_inner={best_params.get('w_inner')}, morph_coupling={best_params.get('morph_coupling')}, "
          f"soft_drift={best_params.get('soft_drift_threshold')}", flush=True)
    print(f"Search took {time.time() - t_start:.1f}s", flush=True)

    return best_pattern, best_score, best_params
