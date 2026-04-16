#!/usr/bin/env python3
"""
Pattern searcher for RD Morphogen Flow-Lenia.

Uses a combination of:
  1. Known-good initial conditions (elongated Gaussian blobs biased for fission)
  2. Random perturbation search (varying blob shape, position, and RD coupling)
  3. Stability evaluation over 500 steps

The search targets patterns that:
  - Survive (retain >10% of initial mass)
  - Have stable mass (low CV)
  - Are spatially concentrated (not diffuse)
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from substrate import RDMorphogenSubstrate


def create_fission_blob(
    grid_size: int = 256,
    seed: int = 42,
    sigma_y: float = 18.0,
    sigma_x: float = 14.0,
    center_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Create an elongated Gaussian blob tuned for fission.

    The elongation biases the first fission axis. The blob has enough mass
    to trigger Turing instability in the RD morphogen layer.
    """
    rng = np.random.RandomState(seed)
    H, W = grid_size, grid_size
    cy = H // 2 + center_offset[0]
    cx = W // 2 + center_offset[1]
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    blob = np.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
    noise = rng.randn(H, W) * 0.02
    state = np.clip(blob + noise, 0.0, 1.0).astype(np.float32)
    return state[np.newaxis]  # (1, H, W)


def evaluate_candidate(
    substrate: RDMorphogenSubstrate,
    pattern: np.ndarray,
    params: dict,
    n_steps: int = 500,
) -> tuple[float, np.ndarray]:
    """Evaluate a candidate pattern for stability and concentration.

    Returns (score, final_state). Higher score = better candidate.
    """
    # Reset morphogen state for each candidate
    substrate.reset()

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor(pattern, dtype=torch.float32, device=device)
        if hasattr(substrate, "to"):
            substrate.to(device)
        masses = []
        with torch.no_grad():
            for _ in range(n_steps):
                t = substrate.update_step(t, params)
                t = t.clamp(0.0, 1.0)
                masses.append(float(t.sum().item()))
        final_state = t.detach().cpu().numpy()
    else:
        state = pattern.copy()
        masses = []
        for _ in range(n_steps):
            state = substrate.update_step(state, params)
            state = np.clip(state, 0.0, 1.0)
            masses.append(float(state.sum()))
        final_state = state

    initial_mass = float(pattern.sum())
    if initial_mass < 1e-8:
        return 0.0, final_state

    final_mass = masses[-1]
    if final_mass < 0.1 * initial_mass:
        return 0.0, final_state  # died

    m = np.array(masses)
    mu = m.mean()
    if mu < 1e-8:
        return 0.0, final_state
    cv = m.std() / mu
    stability = max(0.0, 1.0 - cv)

    # Mass concentration score
    mass_field = final_state.sum(axis=0).flatten()
    total = mass_field.sum()
    if total < 1e-8:
        return 0.0, final_state
    sorted_mass = np.sort(mass_field)[::-1]
    top_10pct = max(1, len(sorted_mass) // 10)
    concentration = sorted_mass[:top_10pct].sum() / total

    score = stability * concentration * min(final_mass / initial_mass, 1.5)
    return float(score), final_state


def search_patterns(
    grid_size: int = 256,
    n_candidates: int = 15,
    stability_steps: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Search for the best fission-capable initial state.

    Strategy:
      - Candidate 0: known-good elongated blob
      - Candidates 1-4: variations on blob shape (different aspect ratios)
      - Candidates 5+: random perturbations
    """
    rng = np.random.default_rng(seed)
    substrate = RDMorphogenSubstrate(grid_size=grid_size)
    params = substrate.get_default_params()
    best_pattern = None
    best_score = -1.0

    print(f"Searching {n_candidates} candidates (seed={seed})...", flush=True)

    for i in range(n_candidates):
        if i == 0:
            # Known-good elongated blob
            candidate = create_fission_blob(grid_size, seed)
            label = "fission_blob"
        elif i < 5:
            # Variations on blob shape
            sy = 14.0 + rng.uniform(-2, 10)
            sx = 10.0 + rng.uniform(-2, 8)
            oy = int(rng.integers(-10, 10))
            ox = int(rng.integers(-10, 10))
            candidate = create_fission_blob(
                grid_size, seed + i, sigma_y=sy, sigma_x=sx,
                center_offset=(oy, ox),
            )
            label = f"blob_var_{i}"
        else:
            # Random Gaussian blobs
            sy = 12.0 + rng.uniform(-4, 12)
            sx = 8.0 + rng.uniform(-2, 10)
            cy_off = int(rng.integers(-20, 20))
            cx_off = int(rng.integers(-20, 20))
            candidate = create_fission_blob(
                grid_size, seed + i * 7 + 3, sigma_y=sy, sigma_x=sx,
                center_offset=(cy_off, cx_off),
            )
            label = f"random_{i}"

        score, final_state = evaluate_candidate(
            substrate, candidate, params, n_steps=stability_steps,
        )

        if score > best_score:
            best_score = score
            best_pattern = candidate
            print(f"  [{label}] score={score:.4f} ** new best **", flush=True)

    if best_pattern is None:
        best_pattern = create_fission_blob(grid_size, seed)

    return best_pattern, best_score
