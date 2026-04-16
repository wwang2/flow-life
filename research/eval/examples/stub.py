"""
Stub solution for eval harness sanity checks.
Returns known-trivial scores: a static Gaussian blob with no repair or replication.
"""
import numpy as np


def get_substrate():
    """Returns a trivial substrate: plain diffusion, no growth dynamics."""
    class TrivialSubstrate:
        def compute_flow(self, state, params):
            # Zero flow — no movement
            return np.zeros((2, state.shape[-2], state.shape[-1]), dtype=np.float32)

        def apply_growth(self, state, flow, params):
            # Pure diffusion kernel (3x3)
            from scipy.ndimage import uniform_filter
            return np.clip(uniform_filter(state, size=3), 0, 1).astype(np.float32)

        def get_default_params(self):
            return {"mu_k": 0.5, "sigma_k": 0.15, "mu_g": 0.15, "sigma_g": 0.015}

    return TrivialSubstrate()


def get_initial_pattern(seed: int = 1, grid_size: int = 256):
    """Returns a static Gaussian blob — no locomotion, no repair, no replication."""
    rng = np.random.default_rng(seed)
    state = np.zeros((1, grid_size, grid_size), dtype=np.float32)
    cx = rng.integers(64, grid_size - 64)
    cy = rng.integers(64, grid_size - 64)
    y, x = np.ogrid[:grid_size, :grid_size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    state[0] = np.clip(np.exp(-dist**2 / (2 * 20**2)), 0, 1).astype(np.float32)
    return state
