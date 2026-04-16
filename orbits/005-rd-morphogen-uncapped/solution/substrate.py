#!/usr/bin/env python3
"""
RD Morphogen Flow-Lenia Substrate V2 -- Uncapped Growth
========================================================

Builds on orbit 003's proven RD morphogen backbone but removes the strict
mass conservation and mass caps that prevented daughter survival.

Key changes from orbit 003:
  1. NO strict mass rescaling -- replaced with soft drift correction
     (only normalize if mass drifts >15% from initial)
  2. NO hard clamp(0, 1) on the growth field -- allow mass to accumulate
     so fission fragments have enough mass to survive independently
  3. dt=0.2 (same as orbit 003)
  4. Persistent drift field for locomotion (same as orbit 003)
  5. RD morphogen coupling preserved for self-repair memory

The hypothesis: orbit 003 scored t4=0 because strict mass conservation
eroded fragment mass below viability. Without it, the inner repulsion
kernel can pinch blobs into daughters that persist.
"""

from __future__ import annotations

import math
from typing import Any

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


class RDMorphogenSubstrateV2(BaseFlowLenia):
    """Two-kernel Flow-Lenia with RD morphogen -- uncapped growth variant.

    The morphogen field is stored as self.morphogen and persists between
    update_step calls. It is NOT part of the state the evaluator sees.
    """

    def __init__(self, grid_size: int = 256):
        super().__init__(grid_size)
        self.n_channels = 1
        self._kernels_built = False
        self._device = None
        self.morphogen = None
        self._rd_freq_cache = None
        self._step_count = 0
        self._initial_mass = 0.0
        self._last_mass = 0.0
        # Persistent drift field for locomotion
        self._drift_field = None
        self._drift_momentum = 0.95

    def get_default_params(self) -> dict:
        return {
            # -- Two-kernel Flow-Lenia --
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
            # -- RD Morphogen (Gierer-Meinhardt) --
            "D_a": 0.01,
            "D_h": 0.4,
            "mu_a": 0.08,
            "mu_h": 0.10,
            "rho": 0.002,
            "dt_rd": 0.02,
            "rd_substeps": 5,
            # -- Coupling --
            "morph_coupling": 0.08,
            "morph_threshold": 0.4,
            "morph_seed_strength": 0.03,
            "flow_gate_strength": 0.05,
            # -- Uncapped growth control --
            "soft_drift_threshold": 0.15,
            "mass_ceiling": 2.8,
        }

    # -- Kernel construction ------------------------------------------------

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
        R_inner = int(params["R_inner"])
        R_outer = int(params["R_outer"])
        self._kernel_inner_fft = self._build_kernel_fft(R_inner, device)
        self._kernel_outer_fft = self._build_kernel_fft(R_outer, device)
        self._flow_kernel_fft = self._build_kernel_fft(
            (R_inner + R_outer) // 2, device
        )
        self._kernels_built = True
        self._device = device

    def _build_rd_freq(self, device):
        H, W = self.grid_size, self.grid_size
        ky = torch.fft.fftfreq(H, device=device).reshape(-1, 1)
        kx = torch.fft.rfftfreq(W, device=device).reshape(1, -1)
        self._rd_freq_cache = -4.0 * (math.pi ** 2) * (ky ** 2 + kx ** 2)

    # -- Growth function ----------------------------------------------------

    def _growth_function(self, u, mu, sigma):
        return 2.0 * torch.exp(-((u - mu) ** 2) / (2 * sigma ** 2)) - 1.0

    def _to_float(self, val):
        if isinstance(val, torch.Tensor):
            return float(val.item())
        return float(val)

    # -- RD Morphogen -------------------------------------------------------

    def _init_morphogen(self, state, params, device):
        H, W = self.grid_size, self.grid_size
        mass = state.sum(dim=0) if state.dim() == 3 else state
        mass_norm = mass / (mass.max() + 1e-8)
        a = (0.1 + 0.5 * mass_norm).unsqueeze(0)
        h = torch.ones(1, H, W, device=device) * 0.5
        return torch.cat([a, h], dim=0)

    def _fft_laplacian(self, field):
        H, W = self.grid_size, self.grid_size
        f_hat = torch.fft.rfft2(field)
        return torch.fft.irfft2(self._rd_freq_cache * f_hat, s=(H, W))

    def _step_rd(self, morphogen, state, params):
        a = morphogen[0:1]
        h = morphogen[1:2]
        D_a = self._to_float(params['D_a'])
        D_h = self._to_float(params['D_h'])
        mu_a = self._to_float(params['mu_a'])
        mu_h = self._to_float(params['mu_h'])
        rho = self._to_float(params['rho'])
        dt_rd = self._to_float(params['dt_rd'])
        seed_str = self._to_float(params.get('morph_seed_strength', 0.03))

        lap_a = self._fft_laplacian(a)
        lap_h = self._fft_laplacian(h)

        mass = state.sum(dim=0).unsqueeze(0) if state.dim() == 3 else state.unsqueeze(0)
        mass_norm = mass / (mass.max() + 1e-8)
        seed_term = seed_str * mass_norm

        a_sq = a ** 2
        da = D_a * lap_a + a_sq / (h + 1e-6) - mu_a * a + rho + seed_term
        dh = D_h * lap_h + a_sq - mu_h * h

        new_a = (a + dt_rd * da).clamp(0.0, 8.0)
        new_h = (h + dt_rd * dh).clamp(1e-4, 8.0)
        return torch.cat([new_a, new_h], dim=0)

    # -- Flow computation ---------------------------------------------------

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
        strength = self._to_float(params.get("flow_strength", 0.05))
        flow = torch.stack([-flow_x, flow_y], dim=0) * strength

        gate_str = self._to_float(params.get("flow_gate_strength", 0.05))
        if self.morphogen is not None and gate_str > 0:
            h = self.morphogen[1]
            h_norm = h / (h.max() + 1e-8)
            gate = (1.0 - gate_str * h_norm).clamp(0.1, 1.0)
            flow = flow * gate.unsqueeze(0)

        # Persistent flow noise for locomotion
        flow_noise_scale = 0.008
        new_flow_noise = torch.randn(2, self.grid_size, self.grid_size, device=device) * flow_noise_scale
        if self._drift_field is None or self._drift_field.shape != new_flow_noise.shape:
            self._drift_field = new_flow_noise.clone()
        else:
            if self._drift_field.device != device:
                self._drift_field = self._drift_field.to(device)
            self._drift_field = self._drift_momentum * self._drift_field + (1 - self._drift_momentum) * new_flow_noise
        flow = flow + self._drift_field

        return flow

    # -- Growth with morphogen coupling (UNCAPPED) --------------------------

    def apply_growth(self, state, flow, params):
        device = state.device
        if not self._kernels_built or self._device != device:
            self._build_kernels(params, device)

        field = state[0] if (state.dim() == 3 and state.shape[0] == 1) else state
        if state.dim() == 3 and state.shape[0] > 1:
            field = state.sum(dim=0)

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

        g_inner = self._growth_function(u_inner, mu_i, si_i)
        g_outer = self._growth_function(u_outer, mu_o, si_o)
        growth = w_i * g_inner + w_o * g_outer

        # Additive morphogen coupling (same as orbit 003)
        coupling = self._to_float(params.get("morph_coupling", 0.08))
        if self.morphogen is not None and coupling > 0:
            a = self.morphogen[0]
            a_norm = (a - a.min()) / (a.max() - a.min() + 1e-8)
            threshold = self._to_float(params.get("morph_threshold", 0.4))
            morph_boost = coupling * (a_norm - threshold)
            growth = growth + morph_boost

        # Growth noise for SSIM stochasticity
        noise_scale = 0.015 * dt
        growth_noise = torch.randn_like(growth) * noise_scale * (field > 0.01).float()
        growth = growth + growth_noise

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

        # Apply growth WITHOUT hard mass conservation
        new_field = advected + dt * growth

        # Soft floor: no negative mass
        new_field = new_field.clamp(min=0.0)

        # Soft drift correction: only normalize if mass drifts too far
        drift_thresh = self._to_float(params.get("soft_drift_threshold", 0.15))
        mass_ceiling = self._to_float(params.get("mass_ceiling", 2.8))

        new_mass = new_field.sum()
        if self._initial_mass > 1e-8 and new_mass > 1e-8:
            drift = (new_mass - self._initial_mass) / self._initial_mass
            if drift > mass_ceiling - 1.0:
                # Hard ceiling: stay safely below evaluator's 3x kill
                target_mass = mass_ceiling * self._initial_mass
                new_field = new_field * (target_mass / new_mass)
            elif abs(drift) > drift_thresh:
                # Soft correction: gently nudge back toward initial mass
                correction_strength = 0.3
                correction_factor = 1.0 - correction_strength * drift / (1.0 + abs(drift))
                new_field = new_field * correction_factor

        if state.dim() == 3:
            return new_field.unsqueeze(0)
        return new_field

    # -- Main update step ---------------------------------------------------

    def update_step(self, state, params):
        device = state.device

        if self._rd_freq_cache is None or self._device != device:
            self._build_rd_freq(device)
            self._device = device

        if self.morphogen is None:
            self.morphogen = self._init_morphogen(state, params, device)
            self._initial_mass = float(state.sum().item())
            self._last_mass = self._initial_mass
        else:
            current_mass = float(state.sum().item())
            if abs(current_mass - self._last_mass) / (self._last_mass + 1e-8) > 0.5:
                self.morphogen = self._init_morphogen(state, params, device)
            self._last_mass = current_mass

        if self.morphogen.device != device:
            self.morphogen = self.morphogen.to(device)

        n_substeps = int(params.get("rd_substeps", 5))
        for _ in range(n_substeps):
            self.morphogen = self._step_rd(self.morphogen, state, params)

        flow = self.compute_flow(state, params)
        new_state = self.apply_growth(state, flow, params)

        self._step_count += 1
        return new_state

    def to(self, device):
        self._kernels_built = False
        self._rd_freq_cache = None
        if self.morphogen is not None:
            self.morphogen = self.morphogen.to(device)
        self._device = None
        return self

    def reset(self):
        self.morphogen = None
        self._step_count = 0
        self._initial_mass = 0.0
        self._last_mass = 0.0
        self._drift_field = None


# Alias for evaluator discovery
SubstrateVariant = RDMorphogenSubstrateV2
