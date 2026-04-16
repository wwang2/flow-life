#!/usr/bin/env python3
"""
Baseline Flow-Lenia solution: Orbium soliton from Plantec et al. 2023.

A minimal, self-contained baseline that:
  - Implements standard Flow-Lenia (Gaussian kernel, standard growth function)
  - Runs a simple random search for stable patterns (50 random seeds, keep
    patterns that survive 500 steps with stable mass)
  - Saves the best pattern to discovered_patterns.npz

Usage:
    python baseline.py --seed 42 --grid-size 256 --output-dir ./output

This inherits BaseFlowLenia from the evaluator's expected interface and
provides a concrete substrate + search procedure.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np

try:
    import torch
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ===================================================================
# BaseFlowLenia interface (matching evaluator expectations)
# ===================================================================


class BaseFlowLenia:
    """Abstract base class for Flow-Lenia substrates.

    Subclasses must implement:
      - compute_flow(state, params) -> flow_field
      - apply_growth(state, flow, params) -> new_state

    The base class provides:
      - update_step(state, params) -> new_state (calls compute_flow + apply_growth)
      - get_default_params() -> dict
    """

    def __init__(self, grid_size: int = 256):
        self.grid_size = grid_size

    def compute_flow(self, state, params):
        raise NotImplementedError

    def apply_growth(self, state, flow, params):
        raise NotImplementedError

    def update_step(self, state, params):
        flow = self.compute_flow(state, params)
        return self.apply_growth(state, flow, params)

    def get_default_params(self) -> dict:
        raise NotImplementedError


# ===================================================================
# Standard Flow-Lenia substrate (Gaussian kernel + bell growth)
# ===================================================================


class OrbiumSubstrate(BaseFlowLenia):
    """Standard Flow-Lenia with Gaussian kernel and bell-curve growth function.

    This implements the classic Lenia kernel from Chan 2019, extended with
    the flow-field formulation from Plantec et al. 2023. The Orbium soliton
    is the canonical stable glider pattern in this substrate.

    Parameters:
        R:     kernel radius (default 13)
        mu:    growth function center (default 0.15)
        sigma: growth function width (default 0.017)
        dt:    time step (default 0.1)
        beta:  kernel ring peaks (default [1.0] for single-ring Gaussian)
    """

    def __init__(self, grid_size: int = 256):
        super().__init__(grid_size)
        self._kernel_cache = {}

    def get_default_params(self) -> dict:
        """Orbium-like parameters from Lenia literature."""
        return {
            "R": 13,
            "mu": 0.15,
            "sigma": 0.017,
            "dt": 0.1,
            "beta": [1.0],
        }

    def _build_kernel(self, R: int, beta: list[float], device=None):
        """Build a Gaussian ring kernel in Fourier space.

        The kernel is a sum of Gaussian rings at radii determined by beta peaks.
        For standard Orbium, beta=[1.0] gives a single Gaussian ring."""
        cache_key = (R, tuple(beta), self.grid_size, str(device))
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        size = self.grid_size
        # Create coordinate grid centered at (0,0) with wrap-around
        mid = size // 2
        y = np.arange(size) - mid
        x = np.arange(size) - mid
        Y, X = np.meshgrid(y, x, indexing="ij")
        D = np.sqrt(X**2 + Y**2) / R  # normalized distance

        # Gaussian ring kernel: K(r) = sum_i beta_i * exp(-((r - r_i)^2) / (2 * sigma_k^2))
        n_rings = len(beta)
        kernel = np.zeros_like(D)
        sigma_k = 0.3  # kernel ring width
        for i, b in enumerate(beta):
            r_i = (i + 0.5) / n_rings  # ring center in [0, 1]
            kernel += b * np.exp(-((D - r_i) ** 2) / (2 * sigma_k**2))

        # Zero out beyond radius 1.0 (normalized)
        kernel[D > 1.0] = 0.0

        # Normalize
        kernel_sum = kernel.sum()
        if kernel_sum > 1e-12:
            kernel /= kernel_sum

        # Shift to put center at (0,0) for FFT
        kernel = np.fft.fftshift(kernel)

        if HAS_TORCH and device is not None:
            kernel_fft = torch.fft.rfft2(
                torch.tensor(kernel, dtype=torch.float32, device=device)
            )
        else:
            kernel_fft = np.fft.rfft2(kernel)

        self._kernel_cache[cache_key] = kernel_fft
        return kernel_fft

    def _growth_function(self, u, mu: float, sigma: float):
        """Bell-curve growth function: G(u) = 2 * exp(-((u - mu)^2) / (2*sigma^2)) - 1

        Maps potential field values to growth rates in [-1, 1].
        Peak growth at u = mu, with width controlled by sigma."""
        if HAS_TORCH and isinstance(u, torch.Tensor):
            return 2.0 * torch.exp(-((u - mu) ** 2) / (2 * sigma**2)) - 1.0
        else:
            return 2.0 * np.exp(-((u - mu) ** 2) / (2 * sigma**2)) - 1.0

    def compute_flow(self, state, params):
        """Compute the flow field via FFT convolution with the kernel.

        Flow = G(K * state), where K is the Gaussian ring kernel and G is
        the bell-curve growth function."""
        R = params["R"]
        mu = params["mu"]
        sigma = params["sigma"]
        beta = params.get("beta", [1.0])

        if HAS_TORCH and isinstance(state, torch.Tensor):
            device = state.device
            kernel_fft = self._build_kernel(R, beta, device=device)

            # Handle multi-channel: process each channel independently
            if state.ndim == 3:
                flows = []
                for c in range(state.shape[0]):
                    state_fft = torch.fft.rfft2(state[c])
                    potential = torch.fft.irfft2(
                        state_fft * kernel_fft, s=(self.grid_size, self.grid_size)
                    )
                    flows.append(self._growth_function(potential, mu, sigma))
                return torch.stack(flows, dim=0)
            else:
                state_fft = torch.fft.rfft2(state)
                potential = torch.fft.irfft2(
                    state_fft * kernel_fft, s=(self.grid_size, self.grid_size)
                )
                return self._growth_function(potential, mu, sigma)
        else:
            kernel_fft = self._build_kernel(R, beta)
            if state.ndim == 3:
                flows = []
                for c in range(state.shape[0]):
                    state_fft = np.fft.rfft2(state[c])
                    potential = np.fft.irfft2(
                        state_fft * kernel_fft, s=(self.grid_size, self.grid_size)
                    )
                    flows.append(self._growth_function(potential, mu, sigma))
                return np.stack(flows, axis=0)
            else:
                state_fft = np.fft.rfft2(state)
                potential = np.fft.irfft2(
                    state_fft * kernel_fft, s=(self.grid_size, self.grid_size)
                )
                return self._growth_function(potential, mu, sigma)

    def apply_growth(self, state, flow, params):
        """Apply growth with time step dt: state_{t+1} = clamp(state_t + dt * flow)."""
        dt = params["dt"]
        if HAS_TORCH and isinstance(state, torch.Tensor):
            return (state + dt * flow).clamp(0.0, 1.0)
        else:
            return np.clip(state + dt * flow, 0.0, 1.0)

    def to(self, device):
        """Move to device (for GPU compatibility with evaluator)."""
        # Kernel cache will be rebuilt on demand for the new device
        self._kernel_cache = {}
        return self


# Alias for evaluator discovery
SubstrateVariant = OrbiumSubstrate


# ===================================================================
# Orbium soliton initial pattern
# ===================================================================


def orbium_seed(grid_size: int = 256, center: tuple[int, int] | None = None) -> np.ndarray:
    """Generate the classic Orbium soliton pattern.

    This is the canonical stable glider from Lenia (Chan 2019). It is a
    crescent-shaped pattern that translates smoothly across the grid.

    The pattern is placed at the specified center (default: grid center)."""
    if center is None:
        center = (grid_size // 2, grid_size // 2)

    # Orbium pattern from Lenia literature (15x15 template)
    orbium = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0.1, 0.14, 0.1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0.08, 0.24, 0.30, 0.24, 0.08, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0.10, 0.34, 0.54, 0.62, 0.54, 0.34, 0.10, 0, 0, 0, 0],
            [0, 0, 0, 0.06, 0.28, 0.54, 0.78, 0.88, 0.78, 0.54, 0.28, 0.06, 0, 0, 0],
            [0, 0, 0.04, 0.22, 0.52, 0.78, 0.94, 1.00, 0.94, 0.78, 0.52, 0.22, 0.04, 0, 0],
            [0, 0.02, 0.16, 0.42, 0.72, 0.92, 1.00, 1.00, 1.00, 0.92, 0.72, 0.42, 0.16, 0.02, 0],
            [0, 0.06, 0.26, 0.56, 0.84, 0.98, 1.00, 1.00, 1.00, 0.98, 0.84, 0.56, 0.26, 0.06, 0],
            [0, 0.04, 0.20, 0.48, 0.76, 0.94, 1.00, 1.00, 1.00, 0.94, 0.76, 0.48, 0.20, 0.04, 0],
            [0, 0, 0.10, 0.32, 0.60, 0.82, 0.94, 0.98, 0.94, 0.82, 0.60, 0.32, 0.10, 0, 0],
            [0, 0, 0.02, 0.14, 0.36, 0.58, 0.76, 0.84, 0.76, 0.58, 0.36, 0.14, 0.02, 0, 0],
            [0, 0, 0, 0.02, 0.12, 0.28, 0.44, 0.52, 0.44, 0.28, 0.12, 0.02, 0, 0, 0],
            [0, 0, 0, 0, 0.02, 0.08, 0.16, 0.20, 0.16, 0.08, 0.02, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0.02, 0.04, 0.02, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.float32,
    )

    # Place on grid (single channel)
    state = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    ph, pw = orbium.shape
    cy, cx = center
    y0 = cy - ph // 2
    x0 = cx - pw // 2

    # Handle wrapping
    for dy in range(ph):
        for dx in range(pw):
            state[0, (y0 + dy) % grid_size, (x0 + dx) % grid_size] = orbium[dy, dx]

    return state


# ===================================================================
# Random search for stable patterns
# ===================================================================


def random_gaussian_blob(
    grid_size: int, rng: np.random.Generator
) -> np.ndarray:
    """Generate a random Gaussian blob as a candidate initial pattern.

    Each blob has randomized:
      - center position
      - radius (5-30 pixels)
      - peak amplitude (0.3-1.0)
      - optional asymmetry (elliptical)
    """
    state = np.zeros((1, grid_size, grid_size), dtype=np.float32)

    cy = rng.integers(0, grid_size)
    cx = rng.integers(0, grid_size)
    radius = rng.uniform(5, 30)
    peak = rng.uniform(0.3, 1.0)

    # Asymmetry: different sigma for y and x
    sigma_y = radius * rng.uniform(0.5, 1.5)
    sigma_x = radius * rng.uniform(0.5, 1.5)

    y = np.arange(grid_size) - cy
    x = np.arange(grid_size) - cx
    # Toroidal wrap
    y = np.where(y > grid_size // 2, y - grid_size, y)
    y = np.where(y < -grid_size // 2, y + grid_size, y)
    x = np.where(x > grid_size // 2, x - grid_size, x)
    x = np.where(x < -grid_size // 2, x + grid_size, x)

    Y, X = np.meshgrid(y, x, indexing="ij")
    blob = peak * np.exp(-(Y**2) / (2 * sigma_y**2) - (X**2) / (2 * sigma_x**2))

    # Add some noise for texture
    noise = rng.uniform(0, 0.1, size=(grid_size, grid_size)).astype(np.float32)
    blob_noisy = np.clip(blob + noise * (blob > 0.01), 0.0, 1.0)

    state[0] = blob_noisy.astype(np.float32)
    return state


def evaluate_stability(
    substrate: BaseFlowLenia,
    pattern: np.ndarray,
    params: dict,
    n_steps: int = 500,
) -> tuple[float, np.ndarray]:
    """Run a pattern for n_steps and evaluate stability.

    Returns (stability_score, final_state).

    Stability score is based on:
      1. Survival: pattern must retain > 10% of initial mass
      2. Mass stability: low coefficient of variation
      3. Localization: mass should be concentrated, not diffuse

    A higher score means a more stable, living-like pattern."""
    if HAS_TORCH:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        t_state = torch.tensor(pattern, dtype=torch.float32, device=device)
        if hasattr(substrate, "to"):
            substrate.to(device)

        masses = []
        with torch.no_grad():
            for _ in range(n_steps):
                t_state = substrate.update_step(t_state, params)
                t_state = t_state.clamp(0.0, 1.0)
                masses.append(float(t_state.sum().item()))

        final_state = t_state.detach().cpu().numpy()
    else:
        state = pattern.copy()
        masses = []
        for _ in range(n_steps):
            state = substrate.update_step(state, params)
            state = np.clip(state, 0.0, 1.0)
            masses.append(float(state.sum()))
        final_state = state

    masses = np.array(masses, dtype=np.float64)
    initial_mass = float(pattern.sum())

    if initial_mass < 1e-8:
        return 0.0, final_state

    # Check 1: survival
    final_mass = masses[-1]
    if final_mass < 0.1 * initial_mass:
        return 0.0, final_state  # died

    # Check 2: mass stability (low CV is good)
    mean_mass = masses.mean()
    if mean_mass < 1e-8:
        return 0.0, final_state
    cv = masses.std() / mean_mass
    stability = max(0.0, 1.0 - cv)

    # Check 3: localization (mass concentration)
    mass_field = final_state.sum(axis=0).flatten()
    total = mass_field.sum()
    if total < 1e-8:
        return 0.0, final_state
    sorted_mass = np.sort(mass_field)[::-1]
    top_10pct = max(1, len(sorted_mass) // 10)
    concentration = sorted_mass[:top_10pct].sum() / total

    # Combined score
    score = stability * concentration
    return float(score), final_state


def search_patterns(
    grid_size: int = 256,
    n_candidates: int = 50,
    stability_steps: int = 500,
    seed: int = 42,
) -> tuple[np.ndarray, float]:
    """Simple random search for stable patterns.

    Strategy:
    1. Always include the known Orbium soliton as candidate 0
    2. Generate 49 random Gaussian blobs
    3. Evaluate each for 500 steps
    4. Return the pattern with the highest stability score

    This is intentionally simple. More sophisticated search (CMA-ES,
    gradient-based optimization, MAP-Elites) can find much better patterns."""
    rng = np.random.default_rng(seed)
    substrate = OrbiumSubstrate(grid_size=grid_size)
    params = substrate.get_default_params()

    best_pattern = None
    best_score = -1.0
    best_final = None

    print(f"Searching {n_candidates} candidates (seed={seed}, grid={grid_size})...",
          flush=True)

    for i in range(n_candidates):
        if i == 0:
            # Candidate 0: known Orbium soliton
            candidate = orbium_seed(grid_size)
            label = "Orbium"
        else:
            # Random Gaussian blob
            candidate = random_gaussian_blob(grid_size, rng)
            label = f"random_{i}"

        score, final_state = evaluate_stability(
            substrate, candidate, params, n_steps=stability_steps
        )

        if score > best_score:
            best_score = score
            best_pattern = candidate
            best_final = final_state
            print(f"  [{label}] score={score:.4f} ** new best **", flush=True)
        elif i % 10 == 0:
            print(f"  [{label}] score={score:.4f}", flush=True)

    print(f"Best score: {best_score:.4f}")

    # Return the initial pattern (not final state) -- the evaluator will
    # run its own simulation from the initial pattern
    if best_pattern is None:
        best_pattern = orbium_seed(grid_size)

    return best_pattern, best_score


# ===================================================================
# CLI entry point (matches evaluator's expected interface)
# ===================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Baseline Flow-Lenia solution: Orbium soliton + random search"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--grid-size", type=int, default=256, help="Grid size (default 256)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./output",
        help="Output directory for discovered_patterns.npz",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=50,
        help="Number of random candidates to search (default 50)",
    )
    args = parser.parse_args()

    # Set seeds
    np.random.seed(args.seed)
    if HAS_TORCH:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Search for stable patterns
    best_pattern, best_score = search_patterns(
        grid_size=args.grid_size,
        n_candidates=args.n_candidates,
        seed=args.seed,
    )

    # Save to output directory
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_path = out_dir / "discovered_patterns.npz"
    np.savez(str(npz_path), pattern_0=best_pattern)
    print(f"Saved best pattern to {npz_path} (shape={best_pattern.shape})")

    # Also save the substrate.py reference (for evaluator to import)
    # In a real solution, substrate.py would be in the solution directory.
    # This baseline IS the substrate.
    print(f"Baseline stability score: {best_score:.4f}")


if __name__ == "__main__":
    main()
