#!/usr/bin/env python3
"""
Fast searcher for RD Morphogen V2.

Strategy: quick survival filter (300 steps) over a small set of blob shapes
and parameter variations. The real fission testing happens in the evaluator --
we just need to find a viable initial condition.

Designed to complete in <3 minutes on CPU (256x256 grid).
"""

from __future__ import annotations

import sys
import time

import numpy as np

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


def quick_survival_score(
    substrate: RDMorphogenSubstrateV2,
    pattern: np.ndarray,
    params: dict,
    n_steps: int = 300,
) -> float:
    """Quick survival + stability check (300 steps).

    Returns a score combining survival, mass stability, and concentration.
    """
    substrate.reset()
    parent_init_mass = float(pattern.sum())
    if parent_init_mass < 1e-8:
        return 0.0

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor(pattern, dtype=torch.float32, device=device)
        substrate.to(device)
    else:
        return 0.0

    masses = []
    with torch.no_grad():
        for step in range(n_steps):
            t = substrate.update_step(t, params)
            # Match evaluator: clamp to [0, 1] after each step
            t = t.clamp(0.0, 1.0)
            current_mass = float(t.sum().item())
            masses.append(current_mass)
            # Early abort: mass creation kill (evaluator kills at 3x)
            # Allow some overshoot during initial transient (first 50 steps)
            if step > 50 and current_mass > 2.9 * parent_init_mass:
                return 0.0
            # Early abort: pattern died
            if current_mass < 0.1 * parent_init_mass:
                return 0.0

    final_mass = masses[-1]
    m = np.array(masses)
    mu = m.mean()
    if mu < 1e-8:
        return 0.0

    # Stability
    cv = m.std() / mu
    stability = max(0.0, 1.0 - cv)

    # Survival ratio
    survival = min(final_mass / parent_init_mass, 2.0) / 2.0

    # Mass concentration
    final_state = t.detach().cpu().numpy()
    mass_field = final_state.sum(axis=0).flatten()
    total = mass_field.sum()
    if total < 1e-8:
        return 0.0
    sorted_mass = np.sort(mass_field)[::-1]
    top_10pct = max(1, len(sorted_mass) // 10)
    concentration = sorted_mass[:top_10pct].sum() / total

    # Combined score: favor stable, surviving, concentrated patterns
    score = stability * survival * concentration
    return float(score)


def search_patterns(
    grid_size: int = 256,
    n_candidates: int = 8,
    seed: int = 42,
) -> tuple:
    """Fast search for viable initial conditions.

    Tests a small grid of parameter variations x blob shapes.
    Each candidate runs only 300 steps for survival testing.
    Total budget: ~3 minutes on CPU.
    """
    rng = np.random.default_rng(seed)
    substrate = RDMorphogenSubstrateV2(grid_size=grid_size)
    base_params = substrate.get_default_params()

    best_pattern = None
    best_score = -1.0
    best_params = None

    # Small parameter grid: focus on key fission drivers
    configs = [
        # (w_inner, morph_coupling, drift_thresh, sigma_y, sigma_x)
        (-0.8,  0.08, 0.15, 18.0, 12.0),  # baseline (orbit 003 params)
        (-1.0,  0.08, 0.15, 20.0, 10.0),  # stronger repulsion, more elongated
        (-0.8,  0.12, 0.20, 18.0, 12.0),  # stronger coupling, more drift
        (-1.0,  0.10, 0.20, 22.0, 10.0),  # strong repulsion + coupling
        (-0.6,  0.08, 0.15, 16.0, 14.0),  # weaker repulsion, rounder
        (-1.2,  0.06, 0.25, 20.0, 8.0),   # very strong repulsion, very elongated
    ]

    print(f"Searching {min(n_candidates, len(configs))} candidates (seed={seed})...", flush=True)
    t_start = time.time()

    for i, (w_i, mc, dt_thresh, sy, sx) in enumerate(configs[:n_candidates]):
        params = dict(base_params)
        params["w_inner"] = w_i
        params["morph_coupling"] = mc
        params["soft_drift_threshold"] = dt_thresh

        pattern = create_fission_blob(
            grid_size, seed + i * 7,
            sigma_y=sy, sigma_x=sx,
            center_offset=(0, 0),
        )

        score = quick_survival_score(substrate, pattern, params, n_steps=300)

        label = f"w_i={w_i:.1f} mc={mc:.2f} dt={dt_thresh:.2f} blob={sy:.0f}x{sx:.0f}"
        if score > best_score:
            best_score = score
            best_pattern = pattern
            best_params = dict(params)
            print(f"  [{i+1}] {label} score={score:.4f} ** NEW BEST **", flush=True)
        else:
            print(f"  [{i+1}] {label} score={score:.4f}", flush=True)

        elapsed = time.time() - t_start
        if elapsed > 180:  # 3 min budget
            print(f"  Time budget reached ({elapsed:.0f}s), stopping", flush=True)
            break

    if best_pattern is None:
        best_pattern = create_fission_blob(grid_size, seed)
        best_params = base_params

    print(f"\nBest score: {best_score:.4f}", flush=True)
    print(f"Search took {time.time() - t_start:.1f}s", flush=True)

    return best_pattern, best_score, best_params
