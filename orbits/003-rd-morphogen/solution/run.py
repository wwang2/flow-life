#!/usr/bin/env python3
"""
Entry point for the RD Morphogen Flow-Lenia solution.

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
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Import from local modules (run.py cwd is solution/)
from substrate import RDMorphogenSubstrate
from searcher import search_patterns, create_fission_blob


def generate_lifecycle_frames(
    substrate: RDMorphogenSubstrate,
    pattern: np.ndarray,
    params: dict,
    n_frames: int = 60,
    total_steps: int = 3000,
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
                t = t.clamp(0.0, 1.0)
                if step % frame_interval == 0:
                    frame = t.detach().cpu().numpy()
                    frames.append(frame)
    else:
        state = pattern.copy()
        for step in range(total_steps):
            state = substrate.update_step(state, params)
            state = np.clip(state, 0.0, 1.0)
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
    # Use a simple colormap: black -> blue -> cyan -> white
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


def main():
    parser = argparse.ArgumentParser(
        description="RD Morphogen Flow-Lenia: pattern discovery"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-candidates", type=int, default=15)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Search for best pattern
    best_pattern, best_score = search_patterns(
        grid_size=args.grid_size,
        n_candidates=args.n_candidates,
        seed=args.seed,
    )
    print(f"Best stability score: {best_score:.4f}", flush=True)

    # Save patterns
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), pattern_0=best_pattern)
    print(f"Saved pattern to {npz_path} (shape={best_pattern.shape})", flush=True)

    # Generate lifecycle GIF
    substrate = RDMorphogenSubstrate(grid_size=args.grid_size)
    params = substrate.get_default_params()

    frames = generate_lifecycle_frames(
        substrate, best_pattern, params,
        n_frames=60, total_steps=3000,
    )

    gifs_dir = out_dir / "gifs"
    gifs_dir.mkdir(parents=True, exist_ok=True)
    save_gif(frames, str(gifs_dir / "pattern_0.gif"))

    # Contact sheet
    save_contact_sheet(frames, str(out_dir / "contact_sheet.png"))

    # Print scores (informational -- evaluator does its own scoring)
    print(f"TIER3_SCORE={best_score:.4f}")
    print(f"TIER4_SCORE=0.0000")


if __name__ == "__main__":
    main()
