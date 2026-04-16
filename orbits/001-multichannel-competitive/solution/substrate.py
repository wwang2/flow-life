"""
Multi-channel competitive Flow-Lenia with gated growth and soft mass cap.

Two-kernel architecture with Turing instability for fission:
  - Inner kernel (short-range): repulsive at high density
  - Outer kernel (long-range): attractive at medium density
  - Growth gating: only applied where mass exists (prevents boundary explosion)
  - Soft mass cap: rescales field when total mass exceeds 2.5x reference
  - Semi-Lagrangian advection: toroidal bilinear interpolation for flow

The interplay of short-range repulsion and long-range attraction creates
organisms that grow, maintain homeostasis, self-repair after damage, and
undergo genuine fission into daughter organisms when they exceed a critical mass.
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
    """Abstract base class matching evaluator interface."""
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


class MultiChannelCompetitiveLenia(BaseFlowLenia):
    """Two-kernel Flow-Lenia with gated growth and soft mass cap.

    The substrate maintains compact organisms through three mechanisms:
    1. Growth gating: growth only where field > threshold (prevents boundary mass creation)
    2. Soft mass cap: if total mass exceeds 2.5x initial, rescale to prevent explosion
    3. Flow advection: semi-Lagrangian with toroidal wrapping for mass-conserving transport

    The two-kernel Turing instability drives fission when organisms exceed critical mass.
    """

    def __init__(self, grid_size: int = 256):
        super().__init__(grid_size)
        self._kernels_built = False
        self._device = None
        self._cached_params_hash = None
        self._reference_mass = None  # Set on first update_step call
        self._step_count = 0

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
            "growth_threshold": 0.05,
            "mass_cap_factor": 2.5,
        }

    def _params_hash(self, params):
        return (
            int(params.get("R_inner", 10)),
            int(params.get("R_outer", 25)),
            id(self._device),
        )

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
        kernel_sum = kernel.sum()
        if kernel_sum > 1e-8:
            kernel /= kernel_sum
        kernel = torch.fft.fftshift(kernel)
        return torch.fft.rfft2(kernel)

    def _build_kernels(self, params, device):
        R_inner = int(params.get("R_inner", 10))
        R_outer = int(params.get("R_outer", 25))
        self._kernel_inner_fft = self._build_kernel_fft(R_inner, device)
        self._kernel_outer_fft = self._build_kernel_fft(R_outer, device)
        self._flow_kernel_fft = self._build_kernel_fft(
            (R_inner + R_outer) // 2, device
        )
        self._kernels_built = True
        self._device = device
        self._cached_params_hash = self._params_hash(params)

    def _ensure_kernels(self, params, device):
        ph = self._params_hash(params)
        if not self._kernels_built or self._device != device or self._cached_params_hash != ph:
            self._build_kernels(params, device)

    def _growth_function(self, u, mu, sigma):
        return 2.0 * torch.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1.0

    def _to_float(self, val):
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    def compute_flow(self, state, params):
        device = state.device
        self._ensure_kernels(params, device)
        mass = state.sum(dim=0) if state.dim() == 3 else state
        mass_fft = torch.fft.rfft2(mass)
        potential = torch.fft.irfft2(
            mass_fft * self._flow_kernel_fft,
            s=(self.grid_size, self.grid_size),
        )
        flow_y = torch.roll(potential, -1, 0) - torch.roll(potential, 1, 0)
        flow_x = torch.roll(potential, -1, 1) - torch.roll(potential, 1, 1)
        strength = self._to_float(params.get("flow_strength", 0.05))
        return torch.stack([-flow_x, flow_y], dim=0) * strength

    def apply_growth(self, state, flow, params):
        device = state.device
        self._ensure_kernels(params, device)

        # Get the mass field
        if state.dim() == 3 and state.shape[0] == 1:
            field = state[0]
        elif state.dim() == 3 and state.shape[0] > 1:
            field = state.sum(dim=0)
        else:
            field = state

        # Set reference mass on first call
        if self._reference_mass is None:
            self._reference_mass = float(field.sum().item())

        # FFT convolution with inner and outer kernels
        field_fft = torch.fft.rfft2(field)
        u_inner = torch.fft.irfft2(
            field_fft * self._kernel_inner_fft,
            s=(self.grid_size, self.grid_size),
        )
        u_outer = torch.fft.irfft2(
            field_fft * self._kernel_outer_fft,
            s=(self.grid_size, self.grid_size),
        )

        mu_i = self._to_float(params["mu_inner"])
        si_i = self._to_float(params["sigma_inner"])
        mu_o = self._to_float(params["mu_outer"])
        si_o = self._to_float(params["sigma_outer"])
        w_i = self._to_float(params["w_inner"])
        w_o = self._to_float(params["w_outer"])
        dt = self._to_float(params["dt"])
        threshold = self._to_float(params.get("growth_threshold", 0.05))
        mass_cap = self._to_float(params.get("mass_cap_factor", 2.5))

        g_inner = self._growth_function(u_inner, mu_i, si_i)
        g_outer = self._growth_function(u_outer, mu_o, si_o)
        growth = w_i * g_inner + w_o * g_outer

        # Growth gating: only apply growth where field exceeds threshold
        # Below threshold: only allow negative growth (decay), preventing mass creation
        gate = (field > threshold).float()
        gated_growth = growth * gate + torch.clamp(growth, max=0.0) * (1.0 - gate)

        # Semi-Lagrangian advection with toroidal wrapping
        H, W = self.grid_size, self.grid_size
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

        new_field = (advected + dt * gated_growth).clamp(0.0, 1.0)

        # Soft mass cap: if total mass exceeds cap * reference, rescale
        # This allows growth for fission but prevents mass explosion
        if self._reference_mass > 1e-8:
            current_mass = new_field.sum().item()
            max_allowed = mass_cap * self._reference_mass
            if current_mass > max_allowed:
                scale = max_allowed / current_mass
                new_field = new_field * scale

        self._step_count += 1

        if state.dim() == 3:
            return new_field.unsqueeze(0)
        return new_field

    def to(self, device):
        self._kernels_built = False
        self._device = None
        self._cached_params_hash = None
        self._reference_mass = None
        self._step_count = 0
        return self


# Alias for evaluator discovery
SubstrateVariant = MultiChannelCompetitiveLenia
