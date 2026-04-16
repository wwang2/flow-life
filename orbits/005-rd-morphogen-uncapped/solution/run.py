#!/usr/bin/env python3
"""
Entry point for the RD Morphogen V2 (Uncapped) Flow-Lenia solution.

Usage:
    python run.py --seed 42 --grid-size 256 --output-dir ./output

Produces:
  - discovered_patterns.npz: top patterns as float32, keys pattern_0..pattern_N
  - gifs/pattern_{i}.gif: lifecycle GIF for each pattern
  - contact_sheet.png: 3x3 grid of 9 frames from best pattern
  - Prints TIER3_SCORE=<float> and TIER4_SCORE=<float> to stdout
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

from substrate import RDMorphogenSubstrateV2
from searcher import search_patterns, create_fission_blob


def generate_lifecycle_frames(
    substrate: RDMorphogenSubstrateV2,
    pattern: np.ndarray,
    params: dict,
    n_frames: int = 60,
    total_steps: int = 5000,
) -> list[np.ndarray]:
    """Generate lifecycle frames for GIF creation."""
    substrate.reset()
    frame_interval = max(1, total_steps // n_frames)
    frames = []

    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t = torch.tensor(pattern, dtype=torch.float32, device=device)
        substrate.to(device)
        with torch.no_grad():
            for step in range(total_steps):
                t = substrate.update_step(t, params)
                t = t.clamp(0.0)
                if step % frame_interval == 0:
                    frame = t.detach().cpu().numpy()
                    frames.append(frame)
    else:
        state = pattern.copy()
        for step in range(total_steps):
            state = substrate.update_step(state, params)
            state = np.clip(state, 0.0, None)
            if step % frame_interval == 0:
                frames.append(state.copy())

    return frames


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
    # Colormap: black -> blue -> cyan -> white
    r = np.clip(field * 3.0 - 2.0, 0, 1)
    g = np.clip(field * 3.0 - 1.0, 0, 1)
    b = np.clip(field * 3.0, 0, 1)
    img = np.stack([r, g, b], axis=-1)
    return (img * 255).astype(np.uint8)


def save_gif(frames: list[np.ndarray], path: str, fps: int = 5):
    """Save frames as GIF."""
    try:
        import imageio
        images = [state_to_image(f) for f in frames]
        imageio.mimsave(path, images, fps=fps, loop=0)
    except ImportError:
        print("Warning: imageio not available, skipping GIF", file=sys.stderr)


def save_contact_sheet(frames: list[np.ndarray], path: str):
    """Save 3x3 grid of 9 evenly-spaced frames."""
    n = len(frames)
    if n < 9:
        indices = list(range(n))
        while len(indices) < 9:
            indices.append(n - 1)
    else:
        indices = [int(i * (n - 1) / 8) for i in range(9)]
    selected = [frames[i] for i in indices]

    images = [state_to_image(f) for f in selected]
    H, W = images[0].shape[:2]

    sheet = np.zeros((H * 3, W * 3, 3), dtype=np.uint8)
    for idx, img in enumerate(images):
        row = idx // 3
        col = idx % 3
        sheet[row * H:(row + 1) * H, col * W:(col + 1) * W] = img

    try:
        from PIL import Image
        Image.fromarray(sheet).save(path)
    except ImportError:
        try:
            import imageio
            imageio.imwrite(path, sheet)
        except ImportError:
            print("Warning: PIL/imageio not available, skipping contact sheet",
                  file=sys.stderr)


def estimate_scores(
    substrate: RDMorphogenSubstrateV2,
    pattern: np.ndarray,
    params: dict,
) -> tuple[float, float]:
    """Quick self-estimate of tier3 and tier4 scores.

    These are informational only -- the evaluator does its own scoring.
    """
    import scipy.ndimage

    substrate.reset()
    if not HAS_TORCH:
        return 0.0, 0.0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = torch.tensor(pattern, dtype=torch.float32, device=device)
    substrate.to(device)

    # Quick stability check (500 steps)
    masses = []
    with torch.no_grad():
        for _ in range(500):
            t = substrate.update_step(t, params)
            t = t.clamp(0.0)
            masses.append(float(t.sum().item()))

    m = np.array(masses)
    mu = m.mean()
    cv = m.std() / mu if mu > 1e-8 else 1.0
    homeostasis = max(0.0, 1.0 - cv)

    # Quick fission check (run more steps)
    substrate.reset()
    t = torch.tensor(pattern, dtype=torch.float32, device=device)
    substrate.to(device)
    n_daughters = 0

    with torch.no_grad():
        for step in range(5000):
            t = substrate.update_step(t, params)
            t = t.clamp(0.0)

            if step >= 2000 and step % 500 == 0:
                state_np = t.detach().cpu().numpy()
                mass_field = state_np.sum(axis=0) if state_np.ndim == 3 else state_np
                pk = mass_field.max()
                if pk > 1e-12:
                    binary = (mass_field > 0.05 * pk).astype(np.int32)
                    padded = np.pad(binary, 15, mode="wrap")
                    struct = scipy.ndimage.generate_binary_structure(2, 2)
                    labeled, n_comp = scipy.ndimage.label(padded, structure=struct)
                    labeled = labeled[15:15+256, 15:15+256]
                    init_mass = float(pattern.sum())
                    count = 0
                    for lbl in np.unique(labeled):
                        if lbl == 0:
                            continue
                        cmass = float(mass_field[labeled == lbl].sum())
                        if cmass > 0.10 * init_mass:
                            count += 1
                    if count > 1:
                        n_daughters = max(n_daughters, count - 1)

    tier3_est = 0.6 * 0.5 + 0.4 * homeostasis  # rough SSIM estimate ~0.5
    tier4_est = min(n_daughters / 3.0, 1.0) * 0.5  # rough: only replication, no heredity

    return tier3_est, tier4_est


def main():
    parser = argparse.ArgumentParser(
        description="RD Morphogen V2 (Uncapped) Flow-Lenia: pattern discovery"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-candidates", type=int, default=20)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    t_start = time.time()

    # Search for best fission-capable pattern
    best_pattern, best_score, best_params = search_patterns(
        grid_size=args.grid_size,
        n_candidates=args.n_candidates,
        seed=args.seed,
    )
    print(f"Best search score: {best_score:.4f}", flush=True)

    # Save patterns
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), pattern_0=best_pattern)
    print(f"Saved pattern to {npz_path} (shape={best_pattern.shape})", flush=True)

    # Generate lifecycle GIF using best params
    substrate = RDMorphogenSubstrateV2(grid_size=args.grid_size)

    frames = generate_lifecycle_frames(
        substrate, best_pattern, best_params,
        n_frames=60, total_steps=5000,
    )

    gifs_dir = out_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    save_gif(frames, str(gifs_dir / "pattern_0.gif"))

    # Contact sheet
    save_contact_sheet(frames, str(out_dir / "contact_sheet.png"))

    # Estimate scores
    substrate2 = RDMorphogenSubstrateV2(grid_size=args.grid_size)
    tier3_est, tier4_est = estimate_scores(substrate2, best_pattern, best_params)

    elapsed = time.time() - t_start
    print(f"\nTotal run time: {elapsed:.1f}s", flush=True)
    print(f"TIER3_SCORE={tier3_est:.4f}")
    print(f"TIER4_SCORE={tier4_est:.4f}")


if __name__ == "__main__":
    main()
