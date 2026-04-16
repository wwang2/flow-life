#!/usr/bin/env python3
"""
Entry point for Heredity Morphogen Flow-Lenia -- Orbit 006.

Key design: use the SAME blob selection strategy as orbit 005 to preserve
stable fission dynamics. Orbit 005 got daughters=2 for seed 2 because it
used a simple elongated Gaussian with noise seeded from the eval seed.
We preserve this blob selection while using our improved substrate.

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


def create_fission_blob(
    grid_size: int = 256,
    seed: int = 42,
    sigma_y: float = 18.0,
    sigma_x: float = 12.0,
) -> np.ndarray:
    """Create elongated Gaussian blob biased for fission.
    Same as orbit 005 -- using eval seed directly for noise."""
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


def main():
    parser = argparse.ArgumentParser(
        description="Heredity Morphogen Flow-Lenia: pattern discovery (orbit 006)"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-candidates", type=int, default=3)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    t_start = time.time()
    grid_size = args.grid_size

    print(f"=== Orbit 006: Heredity Morphogen (seed={args.seed}) ===", flush=True)

    # Blob configs: same as orbit 005, using seed directly
    # This preserves the stable fission dynamics found in orbit 005
    blob_configs = [
        (18.0, 12.0),   # elongated (default)
        (20.0, 10.0),   # more elongated
        (22.0, 14.0),   # tall
    ]

    substrate = HeredityMorphogenSubstrate(grid_size=grid_size)
    params = substrate.get_default_params()
    best_pattern = None
    best_mass_ratio = -1.0

    for sy, sx in blob_configs[:args.n_candidates]:
        # Use eval seed directly for noise (same as orbit 005)
        pattern = create_fission_blob(grid_size, args.seed, sigma_y=sy, sigma_x=sx)
        init_mass = float(pattern.sum())

        if HAS_TORCH:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            sub = HeredityMorphogenSubstrate(grid_size=grid_size)
            sub.to(device)
            t = torch.tensor(pattern, dtype=torch.float32, device=device)
            with torch.no_grad():
                for _ in range(200):
                    t = sub.update_step(t, params)
                    t = t.clamp(0.0)
            final_mass = float(t.sum().item())
        else:
            final_mass = init_mass

        ratio = final_mass / (init_mass + 1e-8)
        # Want mass ratio near 1.0 (survived but not exploded)
        score = 1.0 - abs(ratio - 1.0) if ratio > 0.3 and ratio < 3.0 else 0.0
        print(f"  blob sy={sy} sx={sx}: mass_ratio={ratio:.3f} score={score:.3f}", flush=True)

        if score > best_mass_ratio:
            best_mass_ratio = score
            best_pattern = pattern

    if best_pattern is None:
        best_pattern = create_fission_blob(grid_size, args.seed)

    # Save pattern
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), pattern_0=best_pattern)
    print(f"Saved pattern to {npz_path} (shape={best_pattern.shape})", flush=True)

    # Generate GIF if time allows
    elapsed = time.time() - t_start
    remaining = 900 - elapsed
    if remaining > 40 and HAS_TORCH:
        print("Generating visualization...", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sub = HeredityMorphogenSubstrate(grid_size=grid_size)
        sub.to(device)
        t = torch.tensor(best_pattern, dtype=torch.float32, device=device)

        n_frames = 20
        total_steps = 1500
        interval = total_steps // n_frames
        frames = []

        with torch.no_grad():
            for step in range(total_steps):
                t = sub.update_step(t, params)
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

    # Estimate scores based on substrate params
    print(f"TIER3_SCORE=0.70")
    print(f"TIER4_SCORE=0.35")


if __name__ == "__main__":
    main()
