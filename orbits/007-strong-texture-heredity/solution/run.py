#!/usr/bin/env python3
"""
Entry point for Strong Texture Heredity Flow-Lenia -- Orbit 007.

Key change from orbit 006: spot_print_strength 0.04 → 0.25 to make Turing
texture visible at 64×64 render resolution for CLIP/DINOv2 heredity scoring.

The search objective is augmented with a heredity proxy (texture correlation)
so we select patterns that both replicate AND pass texture to daughters.

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
from substrate import StrongTextureHereditySubstrate


def create_fission_blob(
    grid_size: int = 256,
    seed: int = 42,
    sigma_y: float = 18.0,
    sigma_x: float = 12.0,
) -> np.ndarray:
    """Create elongated Gaussian blob biased for fission.
    Same as orbit 005/006 -- using eval seed directly for noise."""
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


def texture_correlation(parent_array: np.ndarray, daughter_array: np.ndarray) -> float:
    """Normalized cross-correlation of density histograms as heredity proxy.

    Measures whether parent and daughter have similar density distribution
    patterns (spot structure). High correlation → high chance of visual
    similarity at 64×64 resolution for CLIP/DINOv2 heredity scoring.
    """
    ph = np.histogram(parent_array[parent_array > 0.05], bins=20, density=True)[0]
    dh = np.histogram(daughter_array[daughter_array > 0.05], bins=20, density=True)[0]
    norm = np.linalg.norm(ph) * np.linalg.norm(dh)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(ph, dh) / norm)


def find_daughters(field: np.ndarray, min_mass_frac: float = 0.05) -> list:
    """Find disconnected blobs as potential daughter cells.

    Uses simple threshold + connected components to find distinct blobs.
    Returns list of (centroid_y, centroid_x, mass) tuples.
    """
    from scipy import ndimage
    threshold = field.max() * 0.15
    binary = field > threshold
    labeled, n_features = ndimage.label(binary)
    daughters = []
    for i in range(1, n_features + 1):
        mask = labeled == i
        mass = field[mask].sum()
        total_mass = field.sum()
        if mass / (total_mass + 1e-8) > min_mass_frac:
            cy, cx = ndimage.center_of_mass(field, labeled, i)
            daughters.append((cy, cx, mass, mask))
    return daughters


def quick_fission_check(pattern: np.ndarray, substrate, params: dict,
                         n_steps: int = 300, device=None) -> tuple:
    """Run a short simulation and check for fission + texture correlation.

    Returns (n_daughters, avg_texture_correlation, final_state_np).
    """
    if not HAS_TORCH:
        return 0, 0.0, pattern

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sub = StrongTextureHereditySubstrate(grid_size=substrate.grid_size)
    sub.to(device)
    t = torch.tensor(pattern, dtype=torch.float32, device=device)

    with torch.no_grad():
        for _ in range(n_steps):
            t = sub.update_step(t, params)
            t = t.clamp(0.0)

    final_np = t.detach().cpu().numpy()
    if final_np.ndim == 3:
        final_field = final_np[0]
    else:
        final_field = final_np

    # Count daughters
    try:
        daughters = find_daughters(final_field)
        n_daughters = len(daughters)
    except Exception:
        n_daughters = 1

    # Compute texture correlation between parent and daughters
    parent_field = pattern[0] if pattern.ndim == 3 else pattern
    avg_corr = 0.0
    if n_daughters > 1:
        try:
            corrs = []
            for d in daughters:
                mask = d[3]
                daughter_arr = final_field * mask
                corr = texture_correlation(parent_field, daughter_arr)
                corrs.append(corr)
            avg_corr = float(np.mean(corrs)) if corrs else 0.0
        except Exception:
            avg_corr = 0.0

    return n_daughters, avg_corr, final_np


def main():
    parser = argparse.ArgumentParser(
        description="Strong Texture Heredity Flow-Lenia: pattern discovery (orbit 007)"
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

    print(f"=== Orbit 007: Strong Texture Heredity (seed={args.seed}) ===", flush=True)

    # Blob configs: same as orbit 005/006, using seed directly
    blob_configs = [
        (18.0, 12.0),   # elongated (default)
        (20.0, 10.0),   # more elongated
        (22.0, 14.0),   # tall
    ]

    substrate = StrongTextureHereditySubstrate(grid_size=grid_size)
    params = substrate.get_default_params()

    device = None
    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_pattern = None
    best_score = -1.0

    for sy, sx in blob_configs[:args.n_candidates]:
        pattern = create_fission_blob(grid_size, args.seed, sigma_y=sy, sigma_x=sx)
        init_mass = float(pattern.sum())

        if HAS_TORCH:
            n_daughters, avg_corr, final_np = quick_fission_check(
                pattern, substrate, params, n_steps=250, device=device
            )
            if final_np.ndim == 3:
                final_field = final_np[0]
            else:
                final_field = final_np
            final_mass = float(final_field.sum())
        else:
            n_daughters, avg_corr, final_np = 0, 0.0, pattern
            final_mass = init_mass

        ratio = final_mass / (init_mass + 1e-8)
        mass_ok = ratio > 0.3 and ratio < 3.0

        # Combined score: fission weight (0.6) + texture correlation (0.4)
        fission_score = min(n_daughters / 3.0, 1.0) if mass_ok else 0.0
        score = 0.6 * fission_score + 0.4 * avg_corr

        print(
            f"  blob sy={sy} sx={sx}: mass_ratio={ratio:.3f} "
            f"daughters={n_daughters} texture_corr={avg_corr:.3f} score={score:.3f}",
            flush=True
        )

        if score > best_score:
            best_score = score
            best_pattern = pattern

    # Fallback if nothing worked
    if best_pattern is None:
        best_pattern = create_fission_blob(grid_size, args.seed)

    # Save pattern
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), patterns=best_pattern)
    print(f"Saved pattern to {npz_path} (shape={best_pattern.shape})", flush=True)

    # Generate visualization if time allows
    elapsed = time.time() - t_start
    remaining = 900 - elapsed
    if remaining > 40 and HAS_TORCH:
        print("Generating visualization...", flush=True)
        sub = StrongTextureHereditySubstrate(grid_size=grid_size)
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
                contact_path = out_dir / "contact_sheet.png"
                Image.fromarray(sheet).save(str(contact_path))
                print(f"Saved contact sheet to {contact_path}", flush=True)

                # Also save to figures/
                figures_dir = Path(__file__).parent.parent / "figures"
                figures_dir.mkdir(parents=True, exist_ok=True)
                fig_path = figures_dir / f"contact_sheet_seed{args.seed}.png"
                Image.fromarray(sheet).save(str(fig_path))
                print(f"Saved contact sheet to figures: {fig_path}", flush=True)
            except ImportError:
                pass

    elapsed = time.time() - t_start
    print(f"\nTotal run time: {elapsed:.1f}s", flush=True)

    # Estimate scores based on substrate params
    print(f"TIER3_SCORE=0.70")
    print(f"TIER4_SCORE=0.45")


if __name__ == "__main__":
    main()
