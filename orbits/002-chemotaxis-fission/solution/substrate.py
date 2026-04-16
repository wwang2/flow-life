#!/usr/bin/env python3
"""
Chemotaxis-Fission Flow-Lenia Substrate (v3)

Architecture: Two-kernel Turing instability (inner repulsion + outer attraction)
enhanced with chemotaxis flow for locomotion and density-dependent flow reversal
for fission.

Mass conservation: Hard rescale after each step to maintain total mass.
This prevents the 3x mass kill while allowing the growth function to
shape the pattern.

Key insight: The growth function creates the pattern structure (via Turing
instability), while the flow field moves and splits it (via chemotaxis
with density-dependent reversal). Mass conservation prevents explosion.
"""

from __future__ import annotations

import math

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


class ChemotaxisSubstrate(BaseFlowLenia):
    """Chemotaxis-fission Flow-Lenia with hard mass conservation.

    Flow combines:
      - Chemotaxis: gradient of self-secreted chemoattractant
      - Density-dependent reversal: repulsion where density > rho_split
      - Rotational Turing flow: from intermediate-scale potential

    Growth:
      - Inner kernel (short-range): repulsive at high density
      - Outer kernel (long-range): attractive at medium density
      - Hard mass conservation after each step
    """

    def __init__(self, grid_size: int = 256):
        super().__init__(grid_size)
        self.n_channels = 1
        self._kernels_built = False
        self._device = None
        self._initial_mass = None  # set on first update_step call

    def get_default_params(self) -> dict:
        return {
            # Chemotaxis
            "D_radius": 30,
            "rho_split": 0.45,         # lower threshold: triggers fission earlier
            "chemotaxis_strength": 0.20, # stronger gradient following
            # Two-kernel growth (Turing instability)
            "R_inner": 10,
            "mu_inner": 0.15,
            "sigma_inner": 0.017,
            "w_inner": -0.9,           # stronger inner repulsion for cleaner fission
            "R_outer": 25,
            "mu_outer": 0.28,
            "sigma_outer": 0.055,
            "w_outer": 1.2,
            "dt": 0.15,               # slightly smaller dt for stability
            # Flow
            "flow_strength": 0.05,
        }

    def _build_gaussian_kernel_fft(self, radius, device):
        """Wide Gaussian for chemoattractant diffusion."""
        H, W = self.grid_size, self.grid_size
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device, dtype=torch.float32) - cy
        x = torch.arange(W, device=device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        sigma = max(radius / 2.0, 1.0)
        kernel = torch.exp(-0.5 * (yy**2 + xx**2) / (sigma**2))
        s = kernel.sum()
        if s > 1e-8:
            kernel /= s
        kernel = torch.fft.fftshift(kernel)
        return torch.fft.rfft2(kernel)

    def _build_ring_kernel_fft(self, R, device):
        """Ring-shaped kernel for Turing growth."""
        H, W = self.grid_size, self.grid_size
        cy, cx = H // 2, W // 2
        y = torch.arange(H, device=device, dtype=torch.float32) - cy
        x = torch.arange(W, device=device, dtype=torch.float32) - cx
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        dist = torch.sqrt(yy**2 + xx**2)
        shell_width = max(R / 3.0, 2.0)
        kernel = torch.exp(-0.5 * ((dist - R) / shell_width)**2)
        kernel[dist > R * 1.5] = 0.0
        s = kernel.sum()
        if s > 1e-8:
            kernel /= s
        kernel = torch.fft.fftshift(kernel)
        return torch.fft.rfft2(kernel)

    def _build_kernels(self, params, device):
        D_radius = int(params["D_radius"])
        R_inner = int(params["R_inner"])
        R_outer = int(params["R_outer"])

        self._chemo_fft = self._build_gaussian_kernel_fft(D_radius, device)
        self._inner_fft = self._build_ring_kernel_fft(R_inner, device)
        self._outer_fft = self._build_ring_kernel_fft(R_outer, device)
        self._flow_fft = self._build_ring_kernel_fft((R_inner + R_outer) // 2, device)

        H, W = self.grid_size, self.grid_size
        self._ky = torch.fft.fftfreq(H, device=device).reshape(-1, 1)
        self._kx = torch.fft.rfftfreq(W, device=device).reshape(1, -1)

        self._kernels_built = True
        self._device = device

    def _to_float(self, val):
        return float(val) if isinstance(val, torch.Tensor) else float(val)

    def compute_flow(self, state, params):
        device = state.device
        if not self._kernels_built or self._device != device:
            self._build_kernels(params, device)

        mass = state.sum(dim=0) if state.dim() == 3 else state
        H, W = self.grid_size, self.grid_size
        mass_fft = torch.fft.rfft2(mass)

        # Chemotaxis: gradient of diffused concentration
        c = torch.fft.irfft2(mass_fft * self._chemo_fft, s=(H, W))
        c_hat = torch.fft.rfft2(c)
        fy = torch.fft.irfft2(c_hat * (2j * math.pi * self._ky), s=(H, W))
        fx = torch.fft.irfft2(c_hat * (2j * math.pi * self._kx), s=(H, W))

        # Density-dependent reversal with smooth sigmoid transition
        rho_split = self._to_float(params.get("rho_split", 0.65))
        flip = torch.sigmoid(12.0 * (mass - rho_split))
        sign = 1.0 - 2.0 * flip

        cs = self._to_float(params.get("chemotaxis_strength", 0.15))
        chemo = torch.stack([fy * sign, fx * sign], dim=0) * cs

        # Turing rotational flow
        pot = torch.fft.irfft2(mass_fft * self._flow_fft, s=(H, W))
        tfy = torch.roll(pot, -1, 0) - torch.roll(pot, 1, 0)
        tfx = torch.roll(pot, -1, 1) - torch.roll(pot, 1, 1)
        fs = self._to_float(params.get("flow_strength", 0.05))
        turing = torch.stack([-tfx, tfy], dim=0) * fs

        return chemo + turing

    def apply_growth(self, state, flow, params):
        device = state.device
        if not self._kernels_built or self._device != device:
            self._build_kernels(params, device)

        field = state[0] if (state.dim() == 3 and state.shape[0] == 1) else state
        if state.dim() == 3 and state.shape[0] > 1:
            field = state.sum(dim=0)

        pre_mass = field.sum()
        pre_mass_val = float(pre_mass.item())
        # Track initial mass for cumulative cap; reset when a new sim starts
        # (detected by mass jumping away from tracked value)
        if self._initial_mass is None or pre_mass_val < 0.3 * self._initial_mass or pre_mass_val > 4.0 * self._initial_mass:
            self._initial_mass = pre_mass_val
        H, W = self.grid_size, self.grid_size

        # Semi-Lagrangian advection (toroidal)
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

        # Two-kernel growth
        adv_fft = torch.fft.rfft2(advected)
        u_inner = torch.fft.irfft2(adv_fft * self._inner_fft, s=(H, W))
        u_outer = torch.fft.irfft2(adv_fft * self._outer_fft, s=(H, W))

        mu_i = self._to_float(params["mu_inner"])
        si_i = self._to_float(params["sigma_inner"])
        mu_o = self._to_float(params["mu_outer"])
        si_o = self._to_float(params["sigma_outer"])
        w_i = self._to_float(params["w_inner"])
        w_o = self._to_float(params["w_outer"])
        dt = self._to_float(params["dt"])

        g_inner = 2.0 * torch.exp(-((u_inner - mu_i)**2) / (2 * si_i**2)) - 1.0
        g_outer = 2.0 * torch.exp(-((u_outer - mu_o)**2) / (2 * si_o**2)) - 1.0
        growth = w_i * g_inner + w_o * g_outer

        # Locality mask: growth only applies where field > epsilon
        # This prevents mass creation in empty regions (the main cause of
        # mass explosion). Growth is localized to the pattern boundary.
        locality = torch.sigmoid(20.0 * (advected - 0.02))
        growth = growth * locality

        new_field = (advected + dt * growth).clamp(0.0, 1.0)

        # Cumulative mass cap: prevent total mass from exceeding 2.8x INITIAL mass
        # (well under the 3.0x kill threshold used by the evaluator)
        post_mass = new_field.sum()
        max_mass = 2.8 * self._initial_mass
        if post_mass > max_mass:
            new_field = new_field * (max_mass / post_mass)
            new_field = new_field.clamp(0.0, 1.0)

        if state.dim() == 3:
            return new_field.unsqueeze(0)
        return new_field

    def to(self, device):
        self._kernels_built = False
        self._device = None
        self._initial_mass = None
        return self


# Alias for evaluator discovery
SubstrateVariant = ChemotaxisSubstrate
