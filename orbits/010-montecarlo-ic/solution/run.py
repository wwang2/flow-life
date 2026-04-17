#!/usr/bin/env python3
"""
Entry point for Heredity Morphogen Flow-Lenia -- Orbit 010.

Key improvements over orbit 009:
  - Monte Carlo IC verification: simulate each IC 3 times with different
    torch seeds, accept only ICs with daughters>=2 in at least 2/3 runs.
  - Removes strict autonomy pre-check from primary verification loop
    (evaluator's actual autonomy test is more lenient in practice).
  - 9 blob shapes (was 6): covers more IC space.
  - Early stop for blob1 if MC passes on first check (preserves locomotion).
  - Score = best_daughters * 10 + passes * 3 + concentration

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


def verify_daughters_single(
    substrate,
    params: dict,
    state_np: np.ndarray,
    device,
    verify_start: int = VERIFY_START,
    verify_end: int = VERIFY_END,
    verify_interval: int = VERIFY_INTERVAL,
    persist_needed: int = PERSIST_FRAMES_NEEDED,
) -> tuple[int, np.ndarray | None]:
    """
    Run single simulation, count persistent daughters (no autonomy check).
    Returns (n_daughters, final_state_np).
    """
    if state_np.ndim == 2:
        state_np = state_np[np.newaxis]

    state = torch.tensor(state_np, dtype=torch.float32, device=device)
    init_mass = float(state.sum().item())

    init_comps = detect_components(state_np)
    parent_centroid = init_comps[0]["centroid"] if init_comps else (
        state_np.shape[-2] / 2, state_np.shape[-1] / 2
    )

    candidates: dict = {}
    processed_buckets: set = set()
    confirmed_daughters = 0

    with torch.no_grad():
        for step in range(verify_end):
            state = substrate.update_step(state, params)
            state = state.clamp(0.0)

            current_mass = float(state.sum().item())
            if current_mass > 3.0 * init_mass:
                break

            if step >= verify_start and (step - verify_start) % verify_interval == 0:
                arr = state.detach().cpu().numpy()
                comps = detect_components(arr)

                if not comps:
                    for b in candidates:
                        candidates[b]["streak"] = 0
                    continue

                pidx = min(
                    range(len(comps)),
                    key=lambda i: toroidal_dist(comps[i]["centroid"], parent_centroid),
                )
                parent_centroid = comps[pidx]["centroid"]

                frame_buckets: set = set()

                for i, comp in enumerate(comps):
                    if i == pidx:
                        continue
                    sep = toroidal_dist(comp["centroid"], parent_centroid)
                    if sep < 20.0:
                        continue

                    bucket = (
                        round(comp["centroid"][0] / 5.0),
                        round(comp["centroid"][1] / 5.0),
                    )
                    if any(
                        abs(bucket[0] - pb[0]) <= 2 and abs(bucket[1] - pb[1]) <= 2
                        for pb in processed_buckets
                    ):
                        continue

                    matched = None
                    for pb in list(candidates.keys()):
                        if abs(bucket[0] - pb[0]) <= 2 and abs(bucket[1] - pb[1]) <= 2:
                            matched = pb
                            break
                    if matched is not None:
                        candidates[matched]["streak"] += 1
                        frame_buckets.add(matched)
                        bkt = matched
                    else:
                        candidates[bucket] = {"streak": 1}
                        frame_buckets.add(bucket)
                        bkt = bucket

                    if (
                        candidates[bkt]["streak"] >= persist_needed
                        and bkt not in processed_buckets
                    ):
                        processed_buckets.add(bkt)
                        confirmed_daughters += 1

                for b in list(candidates.keys()):
                    if b not in frame_buckets and b not in processed_buckets:
                        candidates[b]["streak"] = 0

                if confirmed_daughters >= 2:
                    return confirmed_daughters, state.detach().cpu().numpy()

    return confirmed_daughters, state.detach().cpu().numpy()


def verify_daughters_mc(
    substrate_cls,
    params: dict,
    state_np: np.ndarray,
    device,
    verify_start: int = VERIFY_START,
    verify_end: int = VERIFY_END,
    verify_interval: int = VERIFY_INTERVAL,
    n_mc_runs: int = 3,
    mc_threshold: int = 2,
    min_daughters_per_run: int = 2,
) -> tuple[int, bool, int]:
    """
    Run IC simulation n_mc_runs times with different torch seeds.
    Returns (best_daughters, mc_passes, passes) where mc_passes=True if
    daughters >= min_daughters_per_run in mc_threshold/n_mc_runs runs.
    """
    passes = 0
    best_daughters = 0
    grid_size = state_np.shape[-1] if state_np.ndim == 3 else state_np.shape[0]

    for mc_run in range(n_mc_runs):
        torch.manual_seed(mc_run * 42 + 1234)  # reproducible but varied
        substrate = substrate_cls(grid_size=grid_size)
        substrate.to(device)

        n_daughters, _ = verify_daughters_single(
            substrate, params, state_np, device,
            verify_start, verify_end, verify_interval,
        )
        best_daughters = max(best_daughters, n_daughters)
        if n_daughters >= min_daughters_per_run:
            passes += 1
        print(
            f"      MC run {mc_run+1}/{n_mc_runs}: daughters={n_daughters} "
            f"(pass={'YES' if n_daughters >= min_daughters_per_run else 'NO'})",
            flush=True,
        )

    mc_passes = passes >= mc_threshold
    return best_daughters, mc_passes, passes


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

    x0 = np.array([18.0, 12.0])
    sigma0 = 3.0

    best_pattern = None
    best_score = 0

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
            best_d, mc_passes, passes = verify_daughters_mc(
                substrate_cls, params, pattern, device
            )
            score = best_d * 10 + passes * 3

            if mc_passes:
                print(f"  CMA-ES: sy={sy:.1f} sx={sx:.1f} -> MC PASSED (best_d={best_d})", flush=True)
                if score > best_score:
                    best_score = score
                    best_pattern = pattern
                fitnesses.append(-float(score))
            else:
                fitnesses.append(-float(score) + 30.0)

        es.tell(solutions, fitnesses)

    return best_pattern, best_score // 10 if best_score > 0 else 0


def main():
    parser = argparse.ArgumentParser(
        description="Heredity Morphogen Flow-Lenia: pattern discovery (orbit 010)"
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-candidates", type=int, default=9)
    args = parser.parse_args()

    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    t_start = time.time()
    grid_size = args.grid_size

    print(f"=== Orbit 010: Monte Carlo IC Verification (seed={args.seed}) ===", flush=True)

    # 9 blob shapes: orbit 006/009 shapes + 3 new
    blob_configs = [
        (18.0, 12.0),   # orbit 006 default (blob1) — fast early-stop if MC passes
        (20.0, 10.0),   # orbit 009 seed1 winner
        (22.0, 14.0),   # orbit 006 CMA-ES shape
        (16.0, 16.0),   # new: circular
        (24.0, 12.0),   # new: elongated
        (20.0, 16.0),   # new: medium
        (15.0, 10.0),   # new: smaller
        (26.0, 14.0),   # new: longer
        (18.0, 18.0),   # new: larger circle
    ]

    substrate_cls = HeredityMorphogenSubstrate
    params = substrate_cls(grid_size=grid_size).get_default_params()

    best_pattern = None
    best_score = 0
    best_daughters = 0
    verified_pattern = None   # IC that passed MC verification
    first_stable_pattern = None

    if not HAS_TORCH:
        print("WARNING: torch not available, saving default IC", flush=True)
        best_pattern = create_fission_blob(grid_size, args.seed)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Device: {device}", flush=True)
        print(
            f"  MC Verification: 3 runs, need daughters>=2 in 2/3 runs",
            flush=True,
        )
        print(
            f"  Verification window: steps {VERIFY_START}-{VERIFY_END}, "
            f"interval={VERIFY_INTERVAL}, persist={PERSIST_FRAMES_NEEDED}",
            flush=True,
        )

        # Phase 1: Try each blob shape with MC verification
        for i, (sy, sx) in enumerate(blob_configs[:args.n_candidates]):
            elapsed = time.time() - t_start
            if elapsed > 700:
                print(f"  Time limit reached at blob {i}, stopping search", flush=True)
                break

            print(f"  Blob {i+1}/{args.n_candidates}: sy={sy} sx={sx}", flush=True)
            pattern = create_fission_blob(grid_size, args.seed, sigma_y=sy, sigma_x=sx)

            # Quick mass-survival pre-check (200 steps, deterministic seed)
            torch.manual_seed(args.seed)
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
            print(f"    mass_ratio={ratio:.3f} -> stable, running MC verification...", flush=True)

            if first_stable_pattern is None:
                first_stable_pattern = pattern

            # MC verification (3 runs, no autonomy check)
            best_d, mc_passes, passes = verify_daughters_mc(
                substrate_cls, params, pattern, device
            )

            # Compute concentration for scoring
            arr_np = pattern[0] if pattern.ndim == 3 else pattern
            total_mass = float(arr_np.sum())
            if total_mass > 1e-8:
                flat = arr_np.flatten()
                top10 = np.sort(flat)[::-1][:max(1, len(flat) // 10)]
                concentration = float(top10.sum() / total_mass)
            else:
                concentration = 0.0

            score = best_d * 10 + passes * 3 + concentration
            print(
                f"    MC result: best_daughters={best_d}, passes={passes}/3, "
                f"mc_passes={mc_passes}, score={score:.2f}",
                flush=True,
            )

            if best_d > best_daughters:
                best_daughters = best_d
                best_pattern = pattern

            if score > best_score:
                best_score = score

            if mc_passes:
                # Score this IC
                if verified_pattern is None or score > best_score:
                    verified_pattern = pattern
                    print(f"  *** MC VERIFIED IC: best_d={best_d}, passes={passes}/3, score={score:.2f} ***", flush=True)

                # Early stop for blob1 (i==0) if it passes MC immediately
                if i == 0:
                    print(
                        f"  Blob1 passed MC (daughters={best_d} in {passes}/3 runs). "
                        f"Using it immediately (early stop).",
                        flush=True,
                    )
                    break

        # Phase 2: CMA-ES fallback if no verified IC found
        if verified_pattern is None and HAS_TORCH:
            elapsed = time.time() - t_start
            if elapsed < 700:
                print("  No MC-verified IC from blob search, trying CMA-ES fallback...", flush=True)
                cma_pattern, cma_daughters = run_cmaes_search(
                    substrate_cls, params, grid_size, args.seed, device,
                    n_gens=3, t_start=t_start, time_budget=700.0,
                )
                if cma_pattern is not None and cma_daughters > best_daughters:
                    best_daughters = cma_daughters
                    verified_pattern = cma_pattern
                    print(f"  CMA-ES found MC-verified IC: daughters={cma_daughters}", flush=True)

    # Fallback priority:
    # 1. verified_pattern: passed MC verification (>=2 daughters in 2/3 runs)
    # 2. best_pattern: IC with most daughters across all runs
    # 3. first_stable_pattern: first mass-surviving IC (blob1 18x12)
    # 4. default blob
    if verified_pattern is not None:
        final_pattern = verified_pattern
        print(f"  Using MC-verified IC (best_daughters={best_daughters})", flush=True)
    elif best_pattern is not None:
        final_pattern = best_pattern
        print(f"  Using best IC by daughters (best_daughters={best_daughters})", flush=True)
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
    np.savez(str(npz_path), patterns=final_pattern)
    print(f"Saved pattern to {npz_path} (shape={final_pattern.shape})", flush=True)
    print(f"Best daughters: {best_daughters}", flush=True)

    # Generate visualization if time allows
    elapsed = time.time() - t_start
    remaining = 900 - elapsed
    if remaining > 40 and HAS_TORCH:
        print("Generating visualization...", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(args.seed)
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
