#!/usr/bin/env python3
"""
Fission-biased two-phase searcher for uncapped two-kernel Flow-Lenia.

Phase 1: survival filter (500 steps, mass > 50% initial)
Phase 2: fission proxy scoring matching the evaluator's Tier 4 exact logic:
  - scipy.ndimage.label on binarized state (threshold = 0.05 * peak mass)
  - daughter mass > 0.10 * initial_mass
  - centroid separation > 20px
  - persistent across 3 consecutive 100-step checks

CMA-ES search over kernel params starting from good.py defaults.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from scipy.ndimage import label as nd_label, center_of_mass
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Add solution dir to path for substrate import
sys.path.insert(0, str(Path(__file__).parent))
from substrate import MultiKernelFlowLenia


# ── Evaluator-matching daughter detection ─────────────────────────────────────

def count_daughters(state_np: np.ndarray, initial_mass: float) -> int:
    """Count persistent daughters matching eval-v1 Tier 4 logic."""
    if not HAS_SCIPY:
        return 0
    field = state_np[0] if state_np.ndim == 3 else state_np
    peak = float(field.max())
    if peak < 1e-8:
        return 0
    threshold = 0.05 * peak
    binary = (field > threshold).astype(np.int32)
    labeled, n_components = nd_label(binary)
    if n_components < 2:
        return 0

    min_mass = 0.10 * initial_mass
    min_sep = 20.0
    daughters = []
    for i in range(1, n_components + 1):
        mask = (labeled == i)
        comp_mass = float(field[mask].sum())
        if comp_mass < min_mass:
            continue
        cy, cx = center_of_mass(mask)
        daughters.append((cy, cx, comp_mass))

    if len(daughters) < 2:
        return 0

    # Check centroid separation between any two daughter pair
    for i in range(len(daughters)):
        for j in range(i + 1, len(daughters)):
            dy = daughters[i][0] - daughters[j][0]
            dx = daughters[i][1] - daughters[j][1]
            sep = math.sqrt(dy * dy + dx * dx)
            if sep >= min_sep:
                return len(daughters)
    return 0


def run_and_score_fission(
    substrate: MultiKernelFlowLenia,
    params: dict,
    initial_state: np.ndarray,
    fission_steps: int = 5000,
    check_interval: int = 100,
    persistence_frames: int = 3,
    device=None,
) -> tuple[float, int, np.ndarray]:
    """Run simulation and score by max daughter count with persistence check.

    Returns (fission_score, max_daughters, final_state).
    """
    if device is None and HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_mass = float(initial_state.sum())

    if HAS_TORCH:
        t = torch.tensor(initial_state, dtype=torch.float32, device=device)
        substrate.to(device)

        consecutive_count = 0
        max_daughters = 0
        best_state = initial_state.copy()

        for step in range(fission_steps):
            with torch.no_grad():
                t = substrate.update_step(t, params)

            if (step + 1) % check_interval == 0:
                state_np = t.cpu().numpy()
                d = count_daughters(state_np, initial_mass)
                if d >= 2:
                    consecutive_count += 1
                    if d > max_daughters:
                        max_daughters = d
                        best_state = state_np.copy()
                    if consecutive_count >= persistence_frames:
                        break
                else:
                    consecutive_count = 0

        final_state = t.cpu().numpy()
    else:
        state = initial_state.copy()
        consecutive_count = 0
        max_daughters = 0
        best_state = initial_state.copy()
        for step in range(fission_steps):
            flow = substrate.compute_flow(state, params)
            state = substrate.apply_growth(state, flow, params)
            state = np.clip(state, 0.0, 1.0)
            if (step + 1) % check_interval == 0:
                d = count_daughters(state, initial_mass)
                if d >= 2:
                    consecutive_count += 1
                    if d > max_daughters:
                        max_daughters = d
                        best_state = state.copy()
                    if consecutive_count >= persistence_frames:
                        break
                else:
                    consecutive_count = 0
        final_state = state

    # Score: replication fraction (0→1 for 1 daughter, increases with more)
    score = min(max_daughters / 3.0, 1.0) if max_daughters >= 2 else 0.0
    return score, max_daughters, best_state


def survival_filter(
    substrate: MultiKernelFlowLenia,
    params: dict,
    state: np.ndarray,
    steps: int = 500,
    min_survival: float = 0.5,
    device=None,
) -> tuple[bool, float]:
    """Phase 1: check the pattern survives N steps with >50% mass retention."""
    if device is None and HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    initial_mass = float(state.sum())
    if initial_mass < 1e-8:
        return False, 0.0

    if HAS_TORCH:
        t = torch.tensor(state, dtype=torch.float32, device=device)
        substrate.to(device)
        with torch.no_grad():
            for _ in range(steps):
                t = substrate.update_step(t, params)
        final_mass = float(t.sum().item())
    else:
        s = state.copy()
        for _ in range(steps):
            flow = substrate.compute_flow(s, params)
            s = substrate.apply_growth(s, flow, params)
            s = np.clip(s, 0.0, 1.0)
        final_mass = float(s.sum())

    ratio = final_mass / initial_mass
    return ratio >= min_survival, ratio


def make_elongated_blob(grid_size: int, rng, offset_y: int = 0, offset_x: int = 0,
                        sy: float = 18.0, sx: float = 14.0) -> np.ndarray:
    H, W = grid_size, grid_size
    cy, cx = H // 2 + offset_y, W // 2 + offset_x
    y = np.arange(H) - cy
    x = np.arange(W) - cx
    yy, xx = np.meshgrid(y, x, indexing="ij")
    blob = np.exp(-0.5 * ((yy / sy) ** 2 + (xx / sx) ** 2))
    noise = rng.standard_normal((H, W)) * 0.02
    return np.clip(blob + noise, 0.0, 1.0).astype(np.float32)[np.newaxis]


# ── CMA-ES (minimal implementation) ──────────────────────────────────────────

class CMAESSimple:
    """Minimal CMA-ES for low-dimensional continuous optimization."""

    def __init__(self, x0: np.ndarray, sigma0: float = 0.3):
        self.n = len(x0)
        self.mean = x0.copy()
        self.sigma = sigma0
        self.C = np.eye(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)
        self.lam = 4 + int(3 * math.log(self.n))
        self.mu = self.lam // 2
        w_raw = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights = w_raw / w_raw.sum()
        self.mueff = 1.0 / (self.weights ** 2).sum()
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1,
                       2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, math.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs
        self.chiN = self.n ** 0.5 * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))
        self.gen = 0

    def ask(self) -> list[np.ndarray]:
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)
        D = np.diag(eigvals ** 0.5)
        BD = eigvecs @ D
        self._zs = [np.random.randn(self.n) for _ in range(self.lam)]
        self._ys = [BD @ z for z in self._zs]
        return [self.mean + self.sigma * y for y in self._ys]

    def tell(self, solutions: list[np.ndarray], fitnesses: list[float]):
        # Minimize — sort ascending
        order = np.argsort(fitnesses)
        top = order[:self.mu]
        old_mean = self.mean.copy()
        self.mean = sum(self.weights[i] * solutions[top[i]] for i in range(self.mu))
        step = (self.mean - old_mean) / self.sigma

        # Update evolution paths
        eigvals, eigvecs = np.linalg.eigh(self.C)
        eigvals = np.maximum(eigvals, 1e-20)
        invsqrtC = eigvecs @ np.diag(eigvals ** -0.5) @ eigvecs.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * invsqrtC @ step
        hsig = (np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * (self.gen + 1))) / self.chiN
                < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hsig * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * step

        # Update covariance
        ys_top = [self._ys[top[i]] for i in range(self.mu)]
        artmp = np.array(ys_top)
        self.C = ((1 - self.c1 - self.cmu) * self.C
                  + self.c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * sum(self.weights[i] * np.outer(ys_top[i], ys_top[i]) for i in range(self.mu)))
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))
        self.gen += 1


# ── Main search ───────────────────────────────────────────────────────────────

def params_from_vector(v: np.ndarray, base: dict) -> dict:
    """Decode CMA-ES vector to substrate params."""
    p = dict(base)
    p["mu_inner"]    = float(np.clip(v[0], 0.05, 0.4))
    p["sigma_inner"] = float(np.clip(v[1], 0.005, 0.08))
    p["w_inner"]     = float(np.clip(v[2], -2.0, -0.1))
    p["mu_outer"]    = float(np.clip(v[3], 0.1, 0.5))
    p["sigma_outer"] = float(np.clip(v[4], 0.02, 0.15))
    p["w_outer"]     = float(np.clip(v[5], 0.2, 2.5))
    p["flow_strength"] = float(np.clip(v[6], 0.01, 0.2))
    p["dt"] = 0.2  # fixed
    return p


def vector_from_params(p: dict) -> np.ndarray:
    return np.array([
        p["mu_inner"], p["sigma_inner"], p["w_inner"],
        p["mu_outer"], p["sigma_outer"], p["w_outer"],
        p["flow_strength"],
    ], dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--output-dir", type=str, default="./output")
    parser.add_argument("--n-candidates", type=int, default=20)
    parser.add_argument("--cmaes-gens", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = None

    G = args.grid_size
    substrate = MultiKernelFlowLenia(grid_size=G)
    base_params = substrate.get_default_params()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Searching {args.n_candidates} candidates (seed={args.seed})...", flush=True)

    best_pattern = None
    best_fission_score = -1.0
    best_params = base_params
    best_daughters = 0

    # Phase A: evaluate good.py defaults + IC variants first
    ic_variants = []
    for i in range(min(6, args.n_candidates)):
        sy = 14.0 + rng.uniform(-2, 10)
        sx = 10.0 + rng.uniform(-2, 8)
        oy = int(rng.integers(-15, 15))
        ox = int(rng.integers(-15, 15))
        ic = make_elongated_blob(G, rng, offset_y=oy, offset_x=ox, sy=sy, sx=sx)
        ic_variants.append(ic)

    for idx, ic in enumerate(ic_variants):
        label = "elongated_blob" if idx == 0 else f"blob_var_{idx}"
        survived, ratio = survival_filter(substrate, base_params, ic, steps=500, device=device)
        if not survived:
            continue
        fscore, ndaughters, _ = run_and_score_fission(substrate, base_params, ic, device=device)
        composite = fscore
        if composite > best_fission_score:
            best_fission_score = composite
            best_pattern = ic.copy()
            best_params = base_params
            best_daughters = ndaughters
            print(f"  [{label}] score={composite:.4f} daughters={ndaughters} ** new best **", flush=True)
        else:
            print(f"  [{label}] score={composite:.4f} daughters={ndaughters}", flush=True)

    # Phase B: CMA-ES search over kernel params (if budget allows)
    remaining = args.n_candidates - len(ic_variants)
    if remaining >= substrate.get_default_params().__len__() and args.cmaes_gens > 0:
        x0 = vector_from_params(base_params)
        cmaes = CMAESSimple(x0, sigma0=0.2)
        for gen in range(args.cmaes_gens):
            solutions = cmaes.ask()
            fitnesses = []
            for sol in solutions:
                p = params_from_vector(sol, base_params)
                ic = make_elongated_blob(G, rng)
                survived, _ = survival_filter(substrate, p, ic, steps=300, device=device)
                if not survived:
                    fitnesses.append(1.0)  # worst (minimizing)
                    continue
                fscore, ndaughters, _ = run_and_score_fission(
                    substrate, p, ic, fission_steps=3000, device=device)
                neg_score = 1.0 - fscore
                fitnesses.append(neg_score)
                if fscore > best_fission_score:
                    best_fission_score = fscore
                    best_pattern = ic.copy()
                    best_params = p
                    best_daughters = ndaughters
                    print(f"  [cmaes-gen{gen}] score={fscore:.4f} daughters={ndaughters} ** new best **", flush=True)
            cmaes.tell(solutions, fitnesses)

    if best_pattern is None:
        best_pattern = make_elongated_blob(G, rng)
        print("  No fission found — saving fallback blob", flush=True)

    # Save outputs
    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), pattern_0=best_pattern)
    print(f"Saved best pattern to {npz_path} (shape={best_pattern.shape})", flush=True)

    # Save GIF
    gif_dir = out_dir / "gifs"
    gif_dir.mkdir(exist_ok=True)
    if HAS_TORCH and HAS_IMAGEIO:
        try:
            t = torch.tensor(best_pattern, dtype=torch.float32, device=device)
            substrate2 = MultiKernelFlowLenia(grid_size=G)
            substrate2.to(device)
            frames = []
            with torch.no_grad():
                for step in range(3000):
                    t = substrate2.update_step(t, best_params)
                    if step % 150 == 0:
                        arr = (t[0].cpu().numpy() * 255).astype(np.uint8)
                        frames.append(np.stack([arr, arr, arr], axis=-1))
            imageio.mimsave(str(gif_dir / "pattern_0.gif"), frames, fps=8)
            print(f"Saved GIF ({len(frames)} frames)", flush=True)
        except Exception as e:
            print(f"GIF save failed: {e}", flush=True)

    # Save contact sheet (simple 3x3 grid of frames)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t2 = torch.tensor(best_pattern, dtype=torch.float32, device=device)
        sub3 = MultiKernelFlowLenia(grid_size=G)
        sub3.to(device)
        capture_steps = [0, 300, 600, 900, 1200, 1500, 2000, 2500, 3000]
        captures = []
        step = 0
        with torch.no_grad():
            for target in capture_steps:
                while step < target:
                    t2 = sub3.update_step(t2, best_params)
                    step += 1
                captures.append(t2[0].cpu().numpy())
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        for i, ax in enumerate(axes.flat):
            ax.imshow(captures[i], cmap="viridis", vmin=0, vmax=1)
            ax.set_title(f"step {capture_steps[i]}", fontsize=8)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_dir / "contact_sheet.png"), dpi=72)
        plt.close()
        print("Saved contact sheet", flush=True)
    except Exception as e:
        print(f"Contact sheet save failed: {e}", flush=True)

    print(f"\nBest fission score: {best_fission_score:.4f}")
    print(f"Best daughters: {best_daughters}")
    t4_est = min(best_daughters / 3.0, 1.0) if best_daughters >= 2 else 0.0
    print(f"TIER3_SCORE=0.000000")
    print(f"TIER4_SCORE={t4_est:.6f}")


if __name__ == "__main__":
    main()
