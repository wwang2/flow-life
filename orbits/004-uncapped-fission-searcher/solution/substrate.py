#!/usr/bin/env python3
"""
Uncapped two-kernel Flow-Lenia substrate for fission.

Directly based on research/eval/examples/good.py (the proven fission reference).
Key design: NO mass caps, NO growth gating, dt=0.2.
The Turing instability (inner repulsion R=10, outer attraction R=25) drives fission
when the organism grows beyond a critical mass.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BaseFlowLenia:
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


class MultiKernelFlowLenia(BaseFlowLenia):
    """Uncapped two-kernel Flow-Lenia — identical to good.py reference substrate.

    Parameters tuned by the fission-biased searcher (run.py).
    Default params are the good.py proven fission parameters.
    """

    def __init__(self, grid_size: int = 256):
        super().__init__(grid_size)
        self.n_channels = 1
        self._kernels_built = False
        self._device = None

    def get_default_params(self) -> dict:
        return {
            "R_inner": 10,
            "mu_inner": 0.15,
            "sigma_inner": 0.017,
            "w_inner": -0.8,
            "R_outer": 25,
            "mu_outer": 0.28,
            "sigma_outer": 0.055,
            "w_outer": 1.2,
            "dt": 0.2,
            "flow_strength": 0.05,
        }

    def _build_kernel_fft(self, R: int, device):
        H, W = self.grid_size, self.grid_size
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device, dtype=torch.float32) - cy
        x = torch.arange(W, device=device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(yy ** 2 + xx ** 2)
        shell_width = max(R / 3.0, 2.0)
        kernel = torch.exp(-0.5 * ((dist - R) / shell_width) ** 2)
        kernel[dist > R * 1.5] = 0.0
        s = kernel.sum()
        if s > 1e-8:
            kernel /= s
        kernel = torch.fft.fftshift(kernel)
        return torch.fft.rfft2(kernel)

    def _build_kernels(self, params, device):
        R_i = int(params["R_inner"])
        R_o = int(params["R_outer"])
        self._kernel_inner_fft = self._build_kernel_fft(R_i, device)
        self._kernel_outer_fft = self._build_kernel_fft(R_o, device)
        self._flow_kernel_fft = self._build_kernel_fft((R_i + R_o) // 2, device)
        self._kernels_built = True
        self._device = device

    def _growth_fn(self, u, mu, sigma):
        return 2.0 * torch.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1.0

    def _f(self, v):
        if isinstance(v, torch.Tensor):
            return float(v)
        return float(v)

    def compute_flow(self, state, params):
        device = state.device
        if not self._kernels_built or self._device != device:
            self._build_kernels(params, device)
        mass = state.sum(dim=0) if state.dim() == 3 else state
        mass_fft = torch.fft.rfft2(mass)
        potential = torch.fft.irfft2(
            mass_fft * self._flow_kernel_fft,
            s=(self.grid_size, self.grid_size),
        )
        flow_y = torch.roll(potential, -1, 0) - torch.roll(potential, 1, 0)
        flow_x = torch.roll(potential, -1, 1) - torch.roll(potential, 1, 1)
        strength = self._f(params.get("flow_strength", 0.05))
        return torch.stack([-flow_x, flow_y], dim=0) * strength

    def apply_growth(self, state, flow, params):
        device = state.device
        if not self._kernels_built or self._device != device:
            self._build_kernels(params, device)

        field = state[0] if (state.dim() == 3 and state.shape[0] == 1) else state
        if state.dim() == 3 and state.shape[0] > 1:
            field = state.sum(dim=0)

        field_fft = torch.fft.rfft2(field)
        u_inner = torch.fft.irfft2(field_fft * self._kernel_inner_fft,
                                   s=(self.grid_size, self.grid_size))
        u_outer = torch.fft.irfft2(field_fft * self._kernel_outer_fft,
                                   s=(self.grid_size, self.grid_size))

        g_i = self._growth_fn(u_inner, self._f(params["mu_inner"]), self._f(params["sigma_inner"]))
        g_o = self._growth_fn(u_outer, self._f(params["mu_outer"]), self._f(params["sigma_outer"]))
        growth = self._f(params["w_inner"]) * g_i + self._f(params["w_outer"]) * g_o

        H, W = self.grid_size, self.grid_size
        device = state.device
        gy = torch.arange(H, device=device, dtype=torch.float32)
        gx = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
        src_y = (grid_y - flow[0]) % H
        src_x = (grid_x - flow[1]) % W
        y0 = src_y.long() % H
        x0 = src_x.long() % W
        y1 = (y0 + 1) % H
        x1 = (x0 + 1) % W
        fy = src_y - src_y.floor()
        fx = src_x - src_x.floor()
        advected = (
            field[y0, x0] * (1 - fy) * (1 - fx)
            + field[y1, x0] * fy * (1 - fx)
            + field[y0, x1] * (1 - fy) * fx
            + field[y1, x1] * fy * fx
        )

        dt = self._f(params["dt"])
        new_field = (advected + dt * growth).clamp(0.0, 1.0)
        if state.dim() == 3:
            return new_field.unsqueeze(0)
        return new_field

    def to(self, device):
        self._kernels_built = False
        self._device = None
        return self
