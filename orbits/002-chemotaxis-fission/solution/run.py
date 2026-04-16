#!/usr/bin/env python3
"""
Chemotaxis-Fission Flow-Lenia -- Discovery Pipeline

Entry point for the evaluator. Searches for initial conditions that produce
stable, fission-capable patterns under the ChemotaxisSubstrate dynamics.

Usage:
    python run.py --seed 42 --grid-size 256 --output-dir ./output
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import substrate from same directory
sys.path.insert(0, str(Path(__file__).parent))
from substrate import ChemotaxisSubstrate


def create_fission_initial_state(
    grid_size: int = 256, seed: int = 42
) -> np.ndarray:
    """Create an elongated Gaussian blob biased toward fission.

    The elongation breaks symmetry so the first fission axis is predictable.
    Slightly off-center placement creates asymmetric chemotaxis gradient
    for locomotion.
    """
    rng = np.random.RandomState(seed)
    H, W = grid_size, grid_size
    # Slightly off-center for asymmetric gradient (locomotion)
    cy = H // 2 + rng.randint(-10, 10)
    cx = W // 2 + rng.randint(-10, 10)
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")

    # Elongated blob: sigma_y > sigma_x biases fission along y-axis
    sigma_y = 18.0 + rng.uniform(-2, 4)
    sigma_x = 13.0 + rng.uniform(-2, 3)
    blob = np.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))

    # Light texture noise to break perfect symmetry
    noise = rng.randn(H, W) * 0.02
    state = np.clip(blob + noise, 0.0, 1.0).astype(np.float32)
    return state[np.newaxis]  # (1, H, W)


def search_patterns(
    grid_size: int = 256,
    n_candidates: int = 12,
    stability_steps: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Search for best fission-capable initial state.

    Strategy:
      - Candidate 0: known-good elongated blob
      - Candidates 1-N: random variations of blob shape and position
      - Score = mass stability * concentration (survival filter)
    """
    rng = np.random.default_rng(seed)
    substrate = ChemotaxisSubstrate(grid_size=grid_size)
    params = substrate.get_default_params()
    best_pattern = None
    best_score = -1.0

    print(f"Searching {n_candidates} candidates (seed={seed})...", flush=True)

    for i in range(n_candidates):
        if i == 0:
            candidate = create_fission_initial_state(grid_size, seed)
            label = "fission_blob"
        else:
            # Random variations
            sigma_y = 14.0 + rng.uniform(-4, 8)
            sigma_x = 10.0 + rng.uniform(-2, 6)
            cy = grid_size // 2 + rng.integers(-20, 20)
            cx = grid_size // 2 + rng.integers(-20, 20)
            H, W = grid_size, grid_size
            y = np.arange(H) - cy
            x = np.arange(W) - cx
            yy, xx = np.meshgrid(y, x, indexing="ij")
            blob = np.exp(-0.5 * ((yy / sigma_y) ** 2 + (xx / sigma_x) ** 2))
            noise = rng.standard_normal((H, W)) * 0.03
            candidate = np.clip(blob + noise, 0.0, 1.0).astype(np.float32)[np.newaxis]
            label = f"variant_{i}"

        # Evaluate stability
        if HAS_TORCH:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            t = torch.tensor(candidate, dtype=torch.float32, device=device)
            substrate_eval = ChemotaxisSubstrate(grid_size=grid_size)
            substrate_eval.to(device)
            masses = []
            with torch.no_grad():
                for _ in range(stability_steps):
                    t = substrate_eval.update_step(t, params)
                    t = t.clamp(0.0, 1.0)
                    masses.append(float(t.sum().item()))
            final_mass = masses[-1]
            initial_mass = float(candidate.sum())
        else:
            continue

        if initial_mass < 1e-8:
            continue
        survival = final_mass / initial_mass
        if survival < 0.1:
            continue
        m = np.array(masses)
        cv = m.std() / max(m.mean(), 1e-8)
        score = max(0, 1.0 - cv) * min(survival, 1.0)

        if score > best_score:
            best_score = score
            best_pattern = candidate
            print(f"  [{label}] score={score:.4f} ** new best **", flush=True)

    if best_pattern is None:
        best_pattern = create_fission_initial_state(grid_size, seed)

    return best_pattern, best_score


def make_contact_sheet(frames: list[np.ndarray], grid_size: int) -> np.ndarray:
    """Create a 3x3 contact sheet from 9 frames."""
    assert len(frames) >= 9, f"Need 9 frames, got {len(frames)}"
    selected = frames[:9]
    rows = []
    for r in range(3):
        row_imgs = []
        for c in range(3):
            f = selected[r * 3 + c]
            if f.ndim == 3:
                f = f.sum(axis=0)
            # Normalize to [0, 255]
            fmax = f.max()
            if fmax > 1e-8:
                f = f / fmax
            img = (f * 255).clip(0, 255).astype(np.uint8)
            row_imgs.append(img)
        rows.append(np.concatenate(row_imgs, axis=1))
    sheet = np.concatenate(rows, axis=0)
    return sheet


def main():
    parser = argparse.ArgumentParser(
        description="Chemotaxis-Fission Flow-Lenia discovery pipeline"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-candidates", type=int, default=12)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Search for best initial condition
    best_pattern, best_score = search_patterns(
        grid_size=args.grid_size,
        n_candidates=args.n_candidates,
        seed=args.seed,
    )

    # Run the best pattern for lifecycle GIF + contact sheet
    substrate = ChemotaxisSubstrate(grid_size=args.grid_size)
    params = substrate.get_default_params()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save discovered patterns
    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), pattern_0=best_pattern)
    print(f"Saved best pattern to {npz_path} (shape={best_pattern.shape})")

    # Run lifecycle simulation for contact sheet and GIF
    capture_steps = [0, 500, 1000, 2000, 3000, 4000, 5000, 7000, 10000]
    frames = []

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor(best_pattern.copy(), dtype=torch.float32, device=device)
        substrate.to(device)

        frames.append(best_pattern.copy())
        step = 0
        capture_idx = 1  # already captured step 0

        gif_frames = [best_pattern.copy()]
        gif_interval = max(1, 10000 // 60)

        with torch.no_grad():
            for s in range(1, 10001):
                t = substrate.update_step(t, params)
                t = t.clamp(0.0, 1.0)
                step = s

                if capture_idx < len(capture_steps) and step == capture_steps[capture_idx]:
                    frames.append(t.detach().cpu().numpy().copy())
                    capture_idx += 1

                if step % gif_interval == 0:
                    gif_frames.append(t.detach().cpu().numpy().copy())

    # Create contact sheet
    if len(frames) >= 9:
        try:
            from PIL import Image
            sheet = make_contact_sheet(frames, args.grid_size)
            sheet_path = out_dir / "contact_sheet.png"
            Image.fromarray(sheet, mode="L").save(str(sheet_path))
            print(f"Saved contact sheet to {sheet_path}")
        except ImportError:
            print("PIL not available, skipping contact sheet", file=sys.stderr)

    # Create GIFs
    gifs_dir = out_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        gif_images = []
        for f in gif_frames:
            if f.ndim == 3:
                f = f.sum(axis=0)
            fmax = f.max()
            if fmax > 1e-8:
                f = f / fmax
            img = Image.fromarray((f * 255).clip(0, 255).astype(np.uint8), mode="L")
            gif_images.append(img)
        if gif_images:
            gif_path = gifs_dir / "pattern_0.gif"
            gif_images[0].save(
                str(gif_path),
                save_all=True,
                append_images=gif_images[1:],
                duration=100,
                loop=0,
            )
            print(f"Saved GIF to {gif_path}")
    except ImportError:
        print("PIL not available, skipping GIF", file=sys.stderr)

    # Print scores (informational -- evaluator computes its own)
    print(f"TIER3_SCORE=0.0")
    print(f"TIER4_SCORE=0.0")


if __name__ == "__main__":
    main()
