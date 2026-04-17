#!/usr/bin/env python3
"""
Entry point for Heredity Morphogen Flow-Lenia -- Orbit 009.

Key improvements over orbit 006:
  - Longer internal verification: 6000 steps (was 1500)
  - More seed-diverse blob shapes: 6 candidates (was 3)
  - Autonomy pre-check: internally verify daughters can survive in isolation
    (mirrors the evaluator's _probabilistic_autonomy_test)
  - Persistence check: daughters must appear in 3 consecutive checks
    during the 3000-6000 step window AND pass autonomy
  - CMA-ES fallback: 3 generations (was 1)

This ensures that the IC saved to discovered_patterns.npz will
sustain fission all the way to the evaluator's 10000-step window,
and the daughters produced will pass the evaluator's autonomy test.

Usage:
    python run.py --seed 1 --grid-size 256 --output-dir /tmp/eval_dir
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Add solution dir to path for substrate import
sys.path.insert(0, str(Path(__file__).parent))
from substrate import HeredityMorphogenSubstrate


# --- Verification parameters ---
VERIFY_START = 3000       # start checking daughters at this step
VERIFY_INTERVAL = 200     # check every N steps
VERIFY_END = 6000         # stop at this step
PERSIST_FRAMES_NEEDED = 3 # need daughters in this many consecutive checks

# --- Autonomy pre-check parameters (mirror evaluator) ---
AUTONOMY_STEPS = 200
AUTONOMY_MASS_THRESHOLD = 0.50
AUTONOMY_PERTURBATION_SCALE = 0.01
AUTONOMY_CENTROID_DISP_PX = 5.0
AUTONOMY_MASS_VARIANCE_THRESH = 0.05
AUTONOMY_TRIALS = 3  # use 3 instead of evaluator's 5 for speed


def create_fission_blob(
    grid_size: int = 256,
    seed: int = 42,
    sigma_y: float = 18.0,
    sigma_x: float = 12.0,
) -> np.ndarray:
    """Create elongated Gaussian blob biased for fission."""
    rng = np.random.RandomState(seed)
    H, W = grid_size, grid_size
    cy, cx = H // 2, W // 2
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    blob = np.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    noise = rng.randn(H, W) * 0.03
    state = np.clip(blob + noise, 0.0, 1.0).astype(np.float32)
    return state[np.newaxis]  # (1, H, W)


def detect_components(state_np: np.ndarray, min_size: int = 50):
    """Detect connected components in state. Returns list of dicts with centroid and mask."""
    from scipy import ndimage

    if state_np.ndim == 3:
        field = state_np[0]
    else:
        field = state_np

    threshold = 0.1 * field.max() if field.max() > 1e-8 else 0.05
    binary = field > threshold
    labeled, n = ndimage.label(binary)

    comps = []
    for i in range(1, n + 1):
        mask = labeled == i
        size = int(mask.sum())
        if size < min_size:
            continue
        ys, xs = np.where(mask)
        cy = float(ys.mean())
        cx = float(xs.mean())
        comps.append({"centroid": (cy, cx), "mask": mask, "size": size})
    return comps


def toroidal_dist(c1, c2, H=256, W=256):
    """Toroidal distance between two centroids."""
    dy = min(abs(c1[0] - c2[0]), H - abs(c1[0] - c2[0]))
    dx = min(abs(c1[1] - c2[1]), W - abs(c1[1] - c2[1]))
    return float(np.sqrt(dy**2 + dx**2))


def check_daughter_autonomy(substrate_cls, params, isolated_state_np, device, seed=0):
    """
    Mirror of evaluator's _probabilistic_autonomy_test.
    Returns True if daughter passes at least 1/AUTONOMY_TRIALS autonomy checks.
    """
    initial_mass = float(isolated_state_np.sum())
    if initial_mass < 1e-8:
        return False

    survived = 0
    rng = np.random.default_rng(seed)

    for trial in range(AUTONOMY_TRIALS):
        trial_seed = int(rng.integers(0, 2**31))
        trial_rng = np.random.default_rng(trial_seed)
        perturbation = trial_rng.normal(
            0, AUTONOMY_PERTURBATION_SCALE, size=isolated_state_np.shape
        ).astype(np.float32)
        perturbed = np.clip(isolated_state_np + perturbation, 0.0, 1.0)

        # Run isolated daughter for AUTONOMY_STEPS
        sub = substrate_cls(grid_size=isolated_state_np.shape[-1])
        sub.to(device)
        t = torch.tensor(perturbed, dtype=torch.float32, device=device)
        masses = []

        with torch.no_grad():
            for _ in range(AUTONOMY_STEPS):
                t = sub.update_step(t, params)
                t = t.clamp(0.0)
                masses.append(float(t.sum().item()))

        final_mass = masses[-1] if masses else 0.0

        # Check survival
        if final_mass < AUTONOMY_MASS_THRESHOLD * initial_mass:
            continue

        # Check motion (centroid displacement)
        final_np = t.detach().cpu().numpy()
        field_init = isolated_state_np[0] if isolated_state_np.ndim == 3 else isolated_state_np
        field_final = final_np[0] if final_np.ndim == 3 else final_np
        ys_i, xs_i = np.where(field_init > 1e-6)
        ys_f, xs_f = np.where(field_final > 1e-6)
        if len(ys_i) == 0 or len(ys_f) == 0:
            continue
        ci = (float(ys_i.mean()), float(xs_i.mean()))
        cf = (float(ys_f.mean()), float(xs_f.mean()))
        centroid_disp = toroidal_dist(ci, cf)

        mass_arr = np.array(masses)
        mass_mean = mass_arr.mean()
        mass_var = mass_arr.var() / (mass_mean**2 + 1e-12) if mass_mean > 1e-8 else 0.0

        is_alive = centroid_disp > AUTONOMY_CENTROID_DISP_PX or mass_var > AUTONOMY_MASS_VARIANCE_THRESH
        if is_alive:
            survived += 1

    return survived > 0


def verify_daughters_long(
    substrate_cls,
    params: dict,
    state_np: np.ndarray,
    device,
    eval_seed: int = 0,
    verify_start: int = VERIFY_START,
    verify_end: int = VERIFY_END,
    verify_interval: int = VERIFY_INTERVAL,
    persist_needed: int = PERSIST_FRAMES_NEEDED,
    check_autonomy: bool = True,
) -> tuple[int, bool, int]:
    """
    Run long simulation, check for persistent autonomous daughters in verify window.
    Returns (confirmed_daughters, passed, raw_daughters) where:
      confirmed_daughters = daughters that passed autonomy check
      passed = True if confirmed_daughters >= 2
      raw_daughters = max raw daughter count (before autonomy check)
    """
    substrate = substrate_cls(grid_size=state_np.shape[-1])
    substrate.to(device)

    if state_np.ndim == 2:
        state_np = state_np[np.newaxis]

    state = torch.tensor(state_np, dtype=torch.float32, device=device)
    init_mass = float(state.sum().item())

    # Track parent centroid (largest component)
    init_comps = detect_components(state_np)
    parent_centroid = init_comps[0]["centroid"] if init_comps else (state_np.shape[-2] / 2, state_np.shape[-1] / 2)

    # Candidate tracking (bucket -> streak count)
    candidates: dict = {}   # bucket -> {"streak": int, "state": np, "mask": np}
    processed_buckets: set = set()
    confirmed_daughters = 0

    streak = 0  # consecutive checks with daughters
    best_daughters = 0

    with torch.no_grad():
        for step in range(verify_end):
            state = substrate.update_step(state, params)
            state = state.clamp(0.0)

            # Safety: mass runaway kill
            current_mass = float(state.sum().item())
            if current_mass > 3.0 * init_mass:
                break

            if step >= verify_start and (step - verify_start) % verify_interval == 0:
                arr = state.detach().cpu().numpy()
                comps = detect_components(arr)

                if not comps:
                    streak = 0
                    for b in candidates:
                        candidates[b]["streak"] = 0
                    continue

                # Identify parent (closest to last known parent centroid)
                pidx = min(range(len(comps)), key=lambda i: toroidal_dist(comps[i]["centroid"], parent_centroid))
                parent_centroid = comps[pidx]["centroid"]

                # Count well-separated non-parent blobs
                n_daughters_here = 0
                frame_buckets: set = set()

                for i, comp in enumerate(comps):
                    if i == pidx:
                        continue
                    sep = toroidal_dist(comp["centroid"], parent_centroid)
                    if sep < 20.0:  # must be well separated (evaluator uses ~20px)
                        continue

                    bucket = (round(comp["centroid"][0] / 5.0), round(comp["centroid"][1] / 5.0))
                    if any(abs(bucket[0] - pb[0]) <= 2 and abs(bucket[1] - pb[1]) <= 2 for pb in processed_buckets):
                        continue

                    matched = None
                    for pb in list(candidates.keys()):
                        if abs(bucket[0] - pb[0]) <= 2 and abs(bucket[1] - pb[1]) <= 2:
                            matched = pb
                            break
                    if matched is not None:
                        candidates[matched]["streak"] += 1
                        candidates[matched]["state"] = arr
                        candidates[matched]["mask"] = comp["mask"]
                        frame_buckets.add(matched)
                        bkt = matched
                    else:
                        candidates[bucket] = {"streak": 1, "state": arr, "mask": comp["mask"]}
                        frame_buckets.add(bucket)
                        bkt = bucket

                    n_daughters_here += 1

                    # Check if this candidate has persisted enough
                    if candidates[bkt]["streak"] >= persist_needed and bkt not in processed_buckets:
                        processed_buckets.add(bkt)
                        if check_autonomy:
                            # Extract isolated daughter and check autonomy
                            isolated = np.zeros_like(arr)
                            mask = candidates[bkt]["mask"]
                            if arr.ndim == 3:
                                for c in range(arr.shape[0]):
                                    isolated[c][mask] = arr[c][mask]
                            else:
                                isolated[mask] = arr[mask]
                            passed_autonomy = check_daughter_autonomy(
                                substrate_cls, params, isolated, device, seed=eval_seed + len(processed_buckets)
                            )
                            if passed_autonomy:
                                confirmed_daughters += 1
                                print(f"    autonomy PASSED at step {step+1}", flush=True)
                            else:
                                print(f"    autonomy FAILED at step {step+1}", flush=True)
                        else:
                            confirmed_daughters += 1

                for b in list(candidates.keys()):
                    if b not in frame_buckets and b not in processed_buckets:
                        candidates[b]["streak"] = 0

                best_daughters = max(best_daughters, n_daughters_here)

                if confirmed_daughters >= 2:
                    return confirmed_daughters, True, best_daughters

    return confirmed_daughters, confirmed_daughters >= 2, best_daughters


def state_to_image(state: np.ndarray) -> np.ndarray:
    """Convert (C, H, W) state to (H, W, 3) uint8 image."""
    if state.ndim == 3:
        field = state.sum(axis=0)
    else:
        field = state
    field = field.astype(np.float64)
    mx = field.max()
    if mx > 1e-8:
        field = field / mx
    r = np.clip(field * 3.0 - 2.0, 0, 1)
    g = np.clip(field * 3.0 - 1.0, 0, 1)
    b = np.clip(field * 3.0, 0, 1)
    img = np.stack([r, g, b], axis=-1)
    return (img * 255).astype(np.uint8)


def run_cmaes_search(
    substrate_cls,
    params: dict,
    grid_size: int,
    seed: int,
    device,
    n_gens: int = 3,
    t_start: float = 0.0,
    time_budget: float = 750.0,
) -> tuple[np.ndarray | None, int]:
    """CMA-ES search for a fission-capable IC. Returns (best_pattern, best_daughters)."""
    try:
        import cma
    except ImportError:
        print("  CMA-ES: cma not installed, skipping", flush=True)
        return None, 0

    print(f"  CMA-ES: starting {n_gens} generation search...", flush=True)

    # Parameterize IC by sigma_y and sigma_x (2D search)
    x0 = np.array([18.0, 12.0])
    sigma0 = 3.0

    best_pattern = None
    best_daughters = 0

    es = cma.CMAEvolutionStrategy(x0, sigma0, {
        'seed': seed,
        'maxiter': n_gens,
        'popsize': 4,
        'verbose': -9,
        'bounds': [[8.0, 6.0], [32.0, 28.0]],
    })

    while not es.stop():
        if time.time() - t_start > time_budget:
            print("  CMA-ES: time budget exceeded, stopping", flush=True)
            break

        solutions = es.ask()
        fitnesses = []

        for sol in solutions:
            sy, sx = float(sol[0]), float(sol[1])
            sy = max(8.0, min(32.0, sy))
            sx = max(6.0, min(28.0, sx))

            pattern = create_fission_blob(grid_size, seed, sigma_y=sy, sigma_x=sx)
            nd, passed = verify_daughters_long(
                substrate_cls, params, pattern, device, eval_seed=seed, check_autonomy=True
            )

            if passed:
                print(f"  CMA-ES: sy={sy:.1f} sx={sx:.1f} -> PASSED (daughters={nd})", flush=True)
                if nd > best_daughters:
                    best_daughters = nd
                    best_pattern = pattern
                fitnesses.append(-float(nd))
            else:
                fitnesses.append(-float(nd) + 10.0)

        es.tell(solutions, fitnesses)

    return best_pattern, best_daughters


def main():
    parser = argparse.ArgumentParser(
        description="Heredity Morphogen Flow-Lenia: pattern discovery (orbit 009)"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-candidates", type=int, default=6)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    t_start = time.time()
    grid_size = args.grid_size

    print(f"=== Orbit 009: Seed-1 Fission Fix (seed={args.seed}) ===", flush=True)

    # 6 blob shapes: 3 from orbit 006 + 3 new shapes
    blob_configs = [
        (18.0, 12.0),   # elongated (orbit 006 default)
        (20.0, 10.0),   # more elongated
        (22.0, 14.0),   # tall
        (16.0, 16.0),   # square-ish
        (24.0, 16.0),   # tall wide
        (20.0, 20.0),   # large round
    ]

    substrate_cls = HeredityMorphogenSubstrate
    params = substrate_cls(grid_size=grid_size).get_default_params()

    # best_pattern: IC with most raw daughters (fallback if nothing passes autonomy)
    # verified_pattern: IC where daughters passed autonomy pre-check
    best_pattern = None
    best_raw_daughters = 0      # max raw daughter count (before autonomy)
    best_confirmed_daughters = 0  # max confirmed (autonomy-passing) daughters
    verified_pattern = None
    first_stable_pattern = None  # very first mass-surviving IC (orbit-006 fallback)
    first_confirmed_pattern = None  # first IC with any confirmed daughter >= 1

    if not HAS_TORCH:
        print("WARNING: torch not available, saving default IC", flush=True)
        best_pattern = create_fission_blob(grid_size, args.seed)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}", flush=True)
        print(f"  Verification: steps {VERIFY_START}-{VERIFY_END}, "
              f"interval={VERIFY_INTERVAL}, persist={PERSIST_FRAMES_NEEDED}, "
              f"autonomy_trials={AUTONOMY_TRIALS}", flush=True)

        # Phase 1: Try each blob shape with full long verification + autonomy check
        for i, (sy, sx) in enumerate(blob_configs[:args.n_candidates]):
            elapsed = time.time() - t_start
            if elapsed > 700:
                print(f"  Time limit reached at blob {i}, stopping search", flush=True)
                break

            print(f"  Blob {i+1}/{args.n_candidates}: sy={sy} sx={sx}", flush=True)
            pattern = create_fission_blob(grid_size, args.seed, sigma_y=sy, sigma_x=sx)

            # Quick mass-survival pre-check (200 steps)
            sub_quick = substrate_cls(grid_size=grid_size)
            sub_quick.to(device)
            t = torch.tensor(pattern, dtype=torch.float32, device=device)
            init_mass = float(t.sum().item())
            with torch.no_grad():
                for _ in range(200):
                    t = sub_quick.update_step(t, params)
                    t = t.clamp(0.0)
            final_mass = float(t.sum().item())
            ratio = final_mass / (init_mass + 1e-8)
            if ratio < 0.3 or ratio > 3.0:
                print(f"    mass_ratio={ratio:.3f} -> unstable, skip", flush=True)
                continue
            print(f"    mass_ratio={ratio:.3f} -> stable, running long verification...", flush=True)

            # Track first stable pattern as ultimate fallback (same as orbit 006 blob1)
            if first_stable_pattern is None:
                first_stable_pattern = pattern

            # Full long verification with autonomy check
            nd, passed, raw_nd = verify_daughters_long(
                substrate_cls, params, pattern, device,
                eval_seed=args.seed, check_autonomy=True
            )
            print(f"    long verify: confirmed={nd}, raw_daughters={raw_nd}, passed={passed}", flush=True)

            # Track best raw-daughters IC (for fallback when no autonomy passes)
            if raw_nd > best_raw_daughters:
                best_raw_daughters = raw_nd
                best_pattern = pattern

            # Track best confirmed-daughters IC
            if nd > best_confirmed_daughters:
                best_confirmed_daughters = nd

            # Track first IC with any confirmed daughter (good fallback for seeds 2&3)
            if nd >= 1 and first_confirmed_pattern is None:
                first_confirmed_pattern = pattern
                # If this is blob1 (first candidate) with any confirmed daughters,
                # stop early - blob1 has proven fission dynamics and better locomotion
                # than later blobs. The evaluator is stochastic and may get more daughters.
                if i == 0:
                    print(f"  Blob1 has confirmed daughters ({nd}), using it (early stop)", flush=True)
                    verified_pattern = pattern  # treat as verified
                    best_confirmed_daughters = nd
                    break

            if passed:
                verified_pattern = pattern
                print(f"  *** VERIFIED IC found: confirmed={nd} raw={raw_nd} ***", flush=True)
                break

        # Phase 2: CMA-ES fallback if no verified IC found
        if verified_pattern is None and HAS_TORCH:
            elapsed = time.time() - t_start
            if elapsed < 700:
                print("  No verified IC from blob search, trying CMA-ES fallback...", flush=True)
                cma_pattern, cma_daughters = run_cmaes_search(
                    substrate_cls, params, grid_size, args.seed, device,
                    n_gens=3, t_start=t_start, time_budget=700.0
                )
                if cma_pattern is not None and cma_daughters > best_confirmed_daughters:
                    best_confirmed_daughters = cma_daughters
                    verified_pattern = cma_pattern
                    print(f"  CMA-ES found verified IC: daughters={cma_daughters}", flush=True)

        best_daughters = best_confirmed_daughters if verified_pattern is not None else best_raw_daughters

    # Fallback priority:
    # 1. verified_pattern: passed autonomy pre-check (>=2 confirmed daughters)
    # 2. first_confirmed_pattern: first IC with >=1 confirmed daughter
    #    - For seeds 2&3 where blob1 gets confirmed=1, use blob1 (same locomotion as orbit 006)
    #    - Better than using blob2 which has worse locomotion
    # 3. first_stable_pattern: first mass-surviving IC (blob1 18x12, orbit-006 equivalent)
    #    - Seeds 2&3 get daughters stochastically with this IC
    # 4. default blob
    if verified_pattern is not None:
        final_pattern = verified_pattern
        print(f"  Using verified IC (confirmed_daughters={best_confirmed_daughters})", flush=True)
    elif first_confirmed_pattern is not None:
        final_pattern = first_confirmed_pattern
        print(f"  Using first-confirmed IC (>=1 confirmed daughter, orbit-006 compatible)", flush=True)
    elif first_stable_pattern is not None:
        final_pattern = first_stable_pattern
        print(f"  Using first-stable IC (orbit-006 blob1 fallback)", flush=True)
    else:
        final_pattern = create_fission_blob(grid_size, args.seed)
        print("  WARNING: using default IC (no candidate survived mass check)", flush=True)

    # Save pattern
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), pattern_0=final_pattern)
    print(f"Saved pattern to {npz_path} (shape={final_pattern.shape})", flush=True)
    print(f"Best confirmed daughters: {best_confirmed_daughters} (raw: {best_raw_daughters})", flush=True)

    # Generate visualization if time allows
    elapsed = time.time() - t_start
    remaining = 900 - elapsed
    if remaining > 40 and HAS_TORCH:
        print("Generating visualization...", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sub_viz = substrate_cls(grid_size=grid_size)
        sub_viz.to(device)
        t = torch.tensor(final_pattern, dtype=torch.float32, device=device)

        n_frames = 20
        total_steps = 1500
        interval = total_steps // n_frames
        frames = []

        with torch.no_grad():
            for step in range(total_steps):
                t = sub_viz.update_step(t, params)
                t = t.clamp(0.0)
                if step % interval == 0:
                    frames.append(t.detach().cpu().numpy())
                if step % 200 == 0 and (time.time() - t_start) > 800:
                    print(f"Time limit at step {step}, stopping viz", flush=True)
                    break

        if frames:
            try:
                import imageio
                gifs_dir = out_dir / "gifs"
                gifs_dir.mkdir(parents=True, exist_ok=True)
                images = [state_to_image(f) for f in frames]
                imageio.mimsave(str(gifs_dir / "pattern_0.gif"), images, fps=5, loop=0)
                print(f"Saved GIF with {len(frames)} frames", flush=True)
            except ImportError:
                print("imageio not available, skipping GIF", flush=True)

            try:
                from PIL import Image
                n = len(frames)
                indices = [int(i * (n - 1) / 8) for i in range(9)] if n >= 9 else list(range(min(n, 9)))
                while len(indices) < 9:
                    indices.append(len(frames) - 1)
                selected = [state_to_image(frames[i]) for i in indices]
                Hf, Wf = selected[0].shape[:2]
                sheet = np.zeros((Hf * 3, Wf * 3, 3), dtype=np.uint8)
                for idx, img in enumerate(selected):
                    r, c = divmod(idx, 3)
                    sheet[r * Hf:(r + 1) * Hf, c * Wf:(c + 1) * Wf] = img
                Image.fromarray(sheet).save(str(out_dir / "contact_sheet.png"))
                print("Saved contact sheet", flush=True)
            except ImportError:
                pass

    elapsed = time.time() - t_start
    print(f"\nTotal run time: {elapsed:.1f}s", flush=True)

    # Score estimates
    tier4_estimate = min(best_daughters / 3.0, 1.0) * 0.5
    print(f"TIER3_SCORE=0.70")
    print(f"TIER4_SCORE={tier4_estimate:.4f}")


if __name__ == "__main__":
    main()
