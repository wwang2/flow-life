#!/usr/bin/env python3
"""
Flow-Life Evaluator -- eval-v1 (FROZEN)
========================================

Definitive synthesis of iter-3 candidates C1-C5 plus AF3 critical fixes.

BASE: C5 (most readable, correct type hints, explicit exploit comments)
FROM C3: Locomotion bonus metric + behavioral fingerprint + colored-border contact sheet
FROM C4: GPU-optimized _sim() with torchmetrics SSIM fallback + timing + batched CLIP/DINOv2
FROM C2: avg(CLIP, DINOv2) for heredity + side-by-side comparison GIF

CRITICAL FIXES (AF3):
  AF3-8: Mass creation kill check INSIDE _sim() itself via max_mass_multiple param.
         Applies to ALL phases (homeostasis, self-repair, replication, contact sheet).
         If mass explodes during homeostasis test, homeostasis_score = 0.
  AF3-3: +/-50 step jitter on narrative contact sheet capture times.
         Uses seed + frame_index as jitter seed so deterministic but unpredictable.
  AF3-9: Deep-copy params before each evaluation phase to prevent cross-phase mutation.

Composite Formula (C3 rebalanced)
----------------------------------
    composite = 0.27 * tier3 + 0.40 * tier4 + 0.23 * vision + 0.10 * locomotion

    tier3 = 0.6 * self_repair + 0.4 * homeostasis
    tier4 = 0.5 * replication + 0.5 * heredity
    heredity = avg(CLIP, DINOv2)  [from C2]
    vision: VLM with null calibration (3 prompts, averaged)
    locomotion: min(displacement_px / 50, 1.0) over 2000 steps

Output
------
    METRIC=<float>
    TIER3_SCORE=<float>
    TIER4_SCORE=<float>
    VISION_SCORE=<float>
    LOCOMOTION_SCORE=<float>

Usage
-----
    python evaluator.py --solution /path/to/solution --seed 42
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import scipy.ndimage
import scipy.signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("evaluator")


# ===================================================================
# Timing Instrumentation (from C4)
# ===================================================================

_stage_timings: dict[str, float] = {}


@contextmanager
def _timed(stage_name: str):
    """Context manager that records wall-clock seconds for a named stage."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    _stage_timings[stage_name] = round(elapsed, 3)
    log.info("TIMING [%s]: %.3fs", stage_name, elapsed)


# ===================================================================
# Section 1: Configuration Constants
# ===================================================================

GRID_SIZE: int = 256

# -- Tier 1 sanity gate --
TIER1_MASS_CONCENTRATION_RATIO: float = 0.60
TIER1_SURVIVAL_STEPS: int = 1000

# -- Tier 3: self-repair + homeostasis --
DAMAGE_FRACTION: float = 0.20
RECOVERY_STEPS: int = 500
STEADY_STATE_STEPS: int = 5000
WARMUP_STEPS: int = 500
SSIM_WIN_SIZE: int = 11
N_DAMAGE_TRIALS: int = 3

# -- Tier 4: replication + heredity --
REPLICATION_STEPS: int = 10000
DETECTION_INTERVAL: int = 100
BINARIZE_THRESHOLD: float = 0.05
MIN_DAUGHTER_MASS_FRAC: float = 0.10
MIN_CENTROID_SEP_PX: int = 20
PERSISTENCE_FRAMES: int = 3
CLIP_RENDER_SIZE: int = 64
HEREDITY_SIM_OFFSET: float = 0.5

# -- Locomotion (from C3) --
LOCOMOTION_STEPS: int = 2000
LOCOMOTION_DISPLACEMENT_CAP_PX: float = 50.0

# -- Toroidal geometry --
TOROIDAL_PAD_RADIUS: int = 15

# -- Mass creation exploit defense --
# AF3-8 FIX: This is now enforced INSIDE _sim() for ALL phases.
MASS_CREATION_KILL_FACTOR: float = 3.0

# -- Probabilistic autonomy test --
AUTONOMY_TRIALS: int = 5
AUTONOMY_STEPS: int = 200
AUTONOMY_MASS_THRESHOLD: float = 0.50
AUTONOMY_PERTURBATION_SCALE: float = 0.01
AUTONOMY_CENTROID_DISP_PX: float = 5.0
AUTONOMY_MASS_VARIANCE_THRESH: float = 0.05

# -- Anti-hardcode defense --
MIN_DAMAGE_TRIAL_VARIANCE: float = 0.01

# -- Triviality guard --
TRIVIALITY_HOMEOSTASIS_THRESH: float = 0.95
TRIVIALITY_SELF_REPAIR_THRESH: float = 0.1
TRIVIALITY_PENALTY: float = 0.5

# -- Vision / VLM --
VLM_MODEL: str = "claude-haiku-4-5-20251001"
CONTACT_SHEET_FRAMES: int = 9
VLM_MAX_RETRIES: int = 5

# -- Composite weights (C3 rebalanced for locomotion) --
W_TIER3: float = 0.27
W_TIER4: float = 0.40
W_VISION: float = 0.23
W_LOCOMOTION: float = 0.10
W_SELF_REPAIR: float = 0.6
W_HOMEOSTASIS: float = 0.4
W_REPLICATION: float = 0.5
W_HEREDITY: float = 0.5

# -- NPZ pattern validation bounds --
PATTERN_MIN_MASS_FRAC: float = 0.001   # 0.1% of grid capacity
PATTERN_MAX_MASS_FRAC: float = 0.30    # 30% of grid capacity
PATTERN_MIN_NONZERO_PX: int = 100

# -- Subprocess isolation timeout --
SUBPROCESS_TIMEOUT_SECS: int = 1200

# -- GIF config --
GIF_FPS: int = 5
GIF_TOTAL_FRAMES: int = 60

# -- AF3-3 FIX: Contact sheet jitter range --
CONTACT_SHEET_JITTER_RANGE: int = 50


# ===================================================================
# Section 2: Custom Exception for AF3-8 Mass Kill
# ===================================================================

class MassCreationError(Exception):
    """Raised when mass exceeds max_mass_multiple * initial_mass inside _sim()."""
    pass


# ===================================================================
# Section 3: Lazy Imports
# ===================================================================

_torch = None
_clip_model = None
_clip_preprocess = None
_dino_model = None
_dino_preprocess = None
_device_cache = None
_dino_available: Optional[bool] = None


def _import_torch():
    """Import torch lazily."""
    global _torch
    if _torch is None:
        import torch
        _torch = torch
    return _torch


def _get_device():
    """Detect and cache the best available compute device."""
    global _device_cache
    if _device_cache is not None:
        return _device_cache
    torch = _import_torch()
    if torch.cuda.is_available():
        _device_cache = torch.device("cuda")
        log.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    else:
        _device_cache = torch.device("cpu")
        log.warning("CUDA unavailable -- falling back to CPU")
    return _device_cache


def _load_clip():
    """Load CLIP ViT-B/32 once; cache globally."""
    global _clip_model, _clip_preprocess
    if _clip_model is None:
        import clip
        torch = _import_torch()
        device = _get_device()
        _clip_model, _clip_preprocess = clip.load("ViT-B/32", device=device)
        _clip_model.eval()
    return _clip_model, _clip_preprocess


def _load_dino():
    """Load DINOv2-small once; cache globally. Returns None on failure."""
    # EXPLOIT PREVENTED: DINOv2 failure no longer crashes evaluation
    global _dino_model, _dino_preprocess, _dino_available
    if _dino_available is False:
        return None, None
    if _dino_model is not None:
        return _dino_model, _dino_preprocess
    try:
        torch = _import_torch()
        from torchvision import transforms
        device = _get_device()
        _dino_model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", trust_repo=True
        )
        _dino_model = _dino_model.to(device)
        _dino_model.eval()
        _dino_preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        _dino_available = True
        log.info("DINOv2 loaded successfully")
        return _dino_model, _dino_preprocess
    except Exception as e:
        _dino_available = False
        log.warning(
            "DINOv2 UNAVAILABLE -- falling back to CLIP-only for heredity. "
            "Reason: %s", e
        )
        return None, None


# ===================================================================
# Section 4: SSIM -- GPU torchmetrics fallback + Wang et al. 2004
# ===================================================================

def _gpu_ssim(img1_np: np.ndarray, img2_np: np.ndarray, device: Any) -> float:
    """Compute SSIM on GPU using torchmetrics if available, else CPU fallback (from C4)."""
    torch = _import_torch()
    try:
        from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
        t1 = torch.tensor(img1_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        t2 = torch.tensor(img2_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        return float(tm_ssim(t1, t2, data_range=1.0, kernel_size=SSIM_WIN_SIZE))
    except ImportError:
        return ssim_wang2004(img1_np, img2_np, win_size=SSIM_WIN_SIZE, data_range=1.0)


def _fspecial_gaussian(size: int, sigma: float) -> np.ndarray:
    """2D Gaussian kernel matching MATLAB fspecial('gaussian')."""
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = np.outer(g, g)
    kernel /= kernel.sum()
    return kernel


def ssim_wang2004(
    img1: np.ndarray,
    img2: np.ndarray,
    win_size: int = 11,
    data_range: float = 1.0,
    K1: float = 0.01,
    K2: float = 0.03,
) -> float:
    """Structural Similarity Index per Wang et al. 2004, Eq. 13."""
    assert img1.shape == img2.shape and img1.ndim == 2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2

    window = _fspecial_gaussian(win_size, sigma=1.5)

    mu1 = scipy.signal.fftconvolve(img1, window, mode="valid")
    mu2 = scipy.signal.fftconvolve(img2, window, mode="valid")

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = np.maximum(
        scipy.signal.fftconvolve(img1 * img1, window, mode="valid") - mu1_sq, 0.0
    )
    sigma2_sq = np.maximum(
        scipy.signal.fftconvolve(img2 * img2, window, mode="valid") - mu2_sq, 0.0
    )
    sigma12 = (
        scipy.signal.fftconvolve(img1 * img2, window, mode="valid") - mu1_mu2
    )

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    return float((numerator / denominator).mean())


def _verify_ssim_against_skimage(img1: np.ndarray, img2: np.ndarray) -> None:
    """Cross-check our SSIM against skimage on the first damage trial."""
    try:
        from skimage.metrics import structural_similarity
        ref = structural_similarity(
            img1, img2, win_size=SSIM_WIN_SIZE,
            gaussian_weights=True, data_range=1.0,
        )
        ours = ssim_wang2004(img1, img2, win_size=SSIM_WIN_SIZE, data_range=1.0)
        if abs(ref - ours) > 1e-6:
            log.warning(
                "SSIM divergence: ours=%.8f skimage=%.8f", ours, ref
            )
    except ImportError:
        pass


# ===================================================================
# Section 5: Toroidal Geometry Utilities
# ===================================================================

def toroidal_pad(arr: np.ndarray, pad_width: int) -> np.ndarray:
    """Wrap-around pad for connected-component labeling on a torus."""
    return np.pad(arr, pad_width, mode="wrap")


def toroidal_distance(
    c1: np.ndarray, c2: np.ndarray,
    H: int = GRID_SIZE, W: int = GRID_SIZE,
) -> float:
    """Minimum-image-convention Euclidean distance on a flat torus."""
    dy = abs(float(c1[0]) - float(c2[0]))
    dx = abs(float(c1[1]) - float(c2[1]))
    dy = min(dy, H - dy)
    dx = min(dx, W - dx)
    return math.sqrt(dy * dy + dx * dx)


def angular_mean_centroid(
    mask: np.ndarray,
    mass_field: np.ndarray,
    H: int = GRID_SIZE,
    W: int = GRID_SIZE,
) -> np.ndarray:
    """Center of mass on a torus via the angular-mean trick."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return np.array([H / 2.0, W / 2.0])

    weights = mass_field[ys, xs].astype(np.float64)
    total_w = weights.sum()
    if total_w < 1e-12:
        return np.array([H / 2.0, W / 2.0])

    theta_y = 2.0 * np.pi * ys / H
    theta_x = 2.0 * np.pi * xs / W

    cy = np.arctan2(
        np.average(np.sin(theta_y), weights=weights),
        np.average(np.cos(theta_y), weights=weights),
    ) * H / (2.0 * np.pi) % H

    cx = np.arctan2(
        np.average(np.sin(theta_x), weights=weights),
        np.average(np.cos(theta_x), weights=weights),
    ) * W / (2.0 * np.pi) % W

    return np.array([cy, cx])


# ===================================================================
# Section 6: PCA Channel Reduction (from C5, with C3 incremental PCA)
# ===================================================================

def _pca_reduce_single(state: np.ndarray) -> np.ndarray:
    """Reduce a single multi-channel state to (H,W) via PCA.

    EXPLOIT PREVENTED: per-channel sum lets adversary craft channels that
    sum to visually identical fields despite different internal distributions.
    PCA weights by variance, making this manipulation much harder.
    """
    if state.ndim == 2:
        field = state.astype(np.float64)
        mx = field.max()
        return field / mx if mx > 1e-12 else field

    n_channels = state.shape[0]
    if n_channels == 1:
        field = state[0].astype(np.float64)
        mx = field.max()
        return field / mx if mx > 1e-12 else field

    C, H, W = state.shape
    flat = state.reshape(C, H * W).astype(np.float64)
    means = flat.mean(axis=1, keepdims=True)
    centered = flat - means
    cov = np.dot(centered, centered.T) / max(H * W - 1, 1)

    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        log.warning("PCA eigendecomposition failed, falling back to channel sum")
        field = state.sum(axis=0).astype(np.float64)
        mx = field.max()
        return field / mx if mx > 1e-12 else field

    pc1 = eigvecs[:, -1]  # largest eigenvalue
    projected = np.dot(pc1, flat).reshape(H, W)

    fmin, fmax = projected.min(), projected.max()
    if fmax - fmin > 1e-12:
        projected = (projected - fmin) / (fmax - fmin)
    else:
        projected = np.zeros((H, W), dtype=np.float64)
    return projected


def _incremental_pca_reduce(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Apply incremental PCA over lifecycle frames for the most informative projection (from C3).

    For single-channel states, returns normalized mass fields directly."""
    if not frames:
        return []
    if frames[0].ndim == 2 or frames[0].shape[0] == 1:
        result = []
        for f in frames:
            mf = f[0] if f.ndim == 3 else f
            mf = mf.astype(np.float64)
            mx = mf.max()
            if mx > 1e-12:
                mf = mf / mx
            result.append(mf)
        return result

    C = frames[0].shape[0]
    H, W = frames[0].shape[1], frames[0].shape[2]

    max_pixels_per_frame = 10000
    stride = max(1, (H * W) // max_pixels_per_frame)

    all_samples = []
    for f in frames:
        flat = f.reshape(C, H * W).astype(np.float64)
        all_samples.append(flat[:, ::stride])

    combined = np.concatenate(all_samples, axis=1)
    means = combined.mean(axis=1, keepdims=True)
    centered = combined - means
    cov = np.dot(centered, centered.T) / max(combined.shape[1] - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    pc1 = eigvecs[:, -1]

    result = []
    for f in frames:
        flat = f.reshape(C, H * W).astype(np.float64)
        projected = np.dot(pc1, flat).reshape(H, W)
        pmin, pmax = projected.min(), projected.max()
        if pmax - pmin > 1e-12:
            projected = (projected - pmin) / (pmax - pmin)
        else:
            projected = np.zeros((H, W), dtype=np.float64)
        result.append(projected)
    return result


# ===================================================================
# Section 7: Solution Loader (Subprocess Isolation)
# ===================================================================

def _load_substrate_module(solution_dir: str):
    """Import substrate.py from the solution directory."""
    sol = Path(solution_dir)
    substrate_path = sol / "substrate.py"
    if not substrate_path.exists():
        raise FileNotFoundError(f"substrate.py not found in {solution_dir}")
    spec = importlib.util.spec_from_file_location("substrate", str(substrate_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["substrate"] = mod
    spec.loader.exec_module(mod)
    return mod


def _create_substrate(mod: Any):
    """Find and instantiate the substrate class from the loaded module."""
    for name in dir(mod):
        obj = getattr(mod, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "compute_flow")
            and name != "BaseFlowLenia"
        ):
            return obj(grid_size=GRID_SIZE)
    for name in dir(mod):
        obj = getattr(mod, name)
        if (
            isinstance(obj, type)
            and hasattr(obj, "update_step")
            and name != "BaseFlowLenia"
        ):
            return obj(grid_size=GRID_SIZE)
    raise RuntimeError("No substrate class found in substrate.py")


def _run_solution_subprocess(
    solution_dir: str, seed: int, output_dir: str
) -> tuple[Path, str, str, int]:
    """Execute run.py in a subprocess with full I/O capture.

    Returns (output_path, stdout_text, stderr_text, returncode).
    EXPLOIT PREVENTED: run.py executes in a separate process, so it cannot
    monkey-patch evaluator globals, override scoring functions, or tamper
    with in-process state.
    """
    run_py = Path(solution_dir) / "run.py"
    if not run_py.exists():
        raise FileNotFoundError(f"run.py not found in {solution_dir}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, str(run_py),
        "--seed", str(seed),
        "--grid-size", str(GRID_SIZE),
        "--output-dir", str(out),
    ]

    log.info("Subprocess: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=SUBPROCESS_TIMEOUT_SECS,
            cwd=solution_dir,
        )
    except subprocess.TimeoutExpired:
        log.error(
            "run.py TIMED OUT after %d seconds -- returning empty output",
            SUBPROCESS_TIMEOUT_SECS,
        )
        return out, "", "TIMEOUT", -1

    if result.stdout:
        for line in result.stdout.strip().split("\n")[:50]:
            log.info("  [run.py stdout] %s", line)
    if result.stderr:
        for line in result.stderr.strip().split("\n")[:20]:
            log.warning("  [run.py stderr] %s", line)

    if result.returncode != 0:
        log.warning("run.py exited with code %d", result.returncode)

    return out, result.stdout or "", result.stderr or "", result.returncode


def _parse_solution_scores(stdout: str) -> dict[str, float]:
    """Parse scores from subprocess stdout (informational only -- NEVER trusted)."""
    # EXPLOIT PREVENTED: we parse but NEVER trust these scores.
    scores = {}
    for line in stdout.split("\n"):
        line = line.strip()
        for key in ("TIER3_SCORE", "TIER4_SCORE", "METRIC"):
            if line.startswith(f"{key}="):
                try:
                    scores[key] = float(line.split("=", 1)[1])
                except ValueError:
                    pass
    if scores:
        log.info("Solution self-reported scores (IGNORED): %s", scores)
    return scores


# ===================================================================
# Section 8: Pattern Loading + Validation
# ===================================================================

def _load_best_pattern(output_dir: Path) -> np.ndarray:
    """Load the best pattern from discovered_patterns.npz."""
    npz_path = output_dir / "discovered_patterns.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"discovered_patterns.npz not in {output_dir}")
    data = np.load(str(npz_path))
    for key in ["pattern_0", "arr_0", "patterns"]:
        if key in data:
            arr = data[key]
            if key == "patterns":
                arr = arr[0]
            break
    else:
        keys = sorted(data.files)
        if not keys:
            raise ValueError("discovered_patterns.npz is empty")
        arr = data[keys[0]]
    arr = arr.astype(np.float32)
    if arr.ndim == 4:
        arr = arr[0]
    if arr.ndim == 2:
        arr = arr[np.newaxis]
    assert arr.ndim == 3, f"Expected (C,H,W), got {arr.shape}"
    return np.clip(arr, 0.0, 1.0)


def _validate_pattern(pattern: np.ndarray) -> bool:
    """Validate loaded pattern against adversarial crafting.

    EXPLOIT PREVENTED: blocks NPZ poisoning where adversary saves crafted
    initial states (all-zeros to trigger SSIM edge cases, or pre-shaped
    patterns that look repaired after exactly 500 steps).
    """
    C, H, W = pattern.shape
    grid_capacity = float(C * H * W)

    if np.isnan(pattern).any():
        log.warning("INVALID_PATTERN: contains NaN values")
        return False
    if np.isinf(pattern).any():
        log.warning("INVALID_PATTERN: contains Inf values")
        return False

    total_mass = float(pattern.sum())
    min_mass = PATTERN_MIN_MASS_FRAC * grid_capacity
    max_mass = PATTERN_MAX_MASS_FRAC * grid_capacity
    if total_mass < min_mass:
        log.warning(
            "INVALID_PATTERN: mass=%.2f < min=%.2f (%.1f%% of capacity)",
            total_mass, min_mass, PATTERN_MIN_MASS_FRAC * 100,
        )
        return False
    if total_mass > max_mass:
        log.warning(
            "INVALID_PATTERN: mass=%.2f > max=%.2f (%.1f%% of capacity)",
            total_mass, max_mass, PATTERN_MAX_MASS_FRAC * 100,
        )
        return False

    n_nonzero = int((pattern > 1e-6).sum())
    if n_nonzero < PATTERN_MIN_NONZERO_PX:
        log.warning(
            "INVALID_PATTERN: only %d non-zero pixels < %d minimum",
            n_nonzero, PATTERN_MIN_NONZERO_PX,
        )
        return False

    log.info(
        "Pattern validated: mass=%.2f (%.2f%% capacity), %d non-zero pixels",
        total_mass, 100.0 * total_mass / grid_capacity, n_nonzero,
    )
    return True


# ===================================================================
# Section 9: State Validation
# ===================================================================

def _validate_nontrivial(state: np.ndarray, label: str = "state") -> None:
    """Raise if state is all-zero or NaN."""
    if np.isnan(state).any():
        raise ValueError(f"{label} contains NaN")
    if state.max() < 1e-8:
        raise ValueError(f"{label} is all-zero -- substrate may be a no-op")


def _validate_evolves(before: np.ndarray, after: np.ndarray, steps: int) -> None:
    """Verify the substrate actually changes state (not a no-op)."""
    if np.abs(after - before).max() < 1e-10:
        raise ValueError(
            f"State unchanged after {steps} steps -- substrate is a no-op"
        )


# ===================================================================
# Section 10: GPU Simulation -- AF3-8 FIX: mass kill INSIDE _sim()
# ===================================================================

def _sim(
    substrate: Any,
    state_np: np.ndarray,
    params: Any,
    n_steps: int,
    *,
    record_mass: bool = False,
    record_frames: bool = False,
    frame_interval: int = 1,
    max_mass_multiple: float = 0.0,
    device: Any = None,
) -> tuple[np.ndarray, list[float], list[np.ndarray], bool]:
    """Run substrate n_steps on GPU.

    Returns (final_np, mass_list, frame_list, mass_exceeded).

    AF3-8 FIX: If max_mass_multiple > 0, the initial mass is computed and
    a hard kill is applied inside the simulation loop. If state.sum() exceeds
    max_mass_multiple * initial_mass at any step, simulation stops immediately
    and mass_exceeded=True is returned. This applies to ALL evaluation phases.
    """
    torch = _import_torch()
    if device is None:
        device = _get_device()
    t = torch.tensor(state_np, dtype=torch.float32, device=device)
    if hasattr(substrate, "to"):
        try:
            substrate.to(device)
        except Exception:
            pass

    # AF3-8 FIX: compute mass limit from initial state
    initial_mass = float(t.sum().item())
    mass_limit = max_mass_multiple * initial_mass if max_mass_multiple > 0 and initial_mass > 1e-8 else 0.0

    masses: list[float] = []
    frames: list[np.ndarray] = []
    mass_exceeded = False

    with torch.no_grad():
        for step_i in range(n_steps):
            t = substrate.update_step(t, params)
            t = t.clamp(0.0, 1.0)

            if record_mass or mass_limit > 0:
                current_mass = float(t.sum().item())
                if record_mass:
                    masses.append(current_mass)

                # AF3-8 FIX: mass creation kill check inside _sim()
                if mass_limit > 0 and current_mass > mass_limit:
                    log.warning(
                        "MASS CREATION KILL in _sim at step %d: "
                        "mass=%.1f > %.1f (%.1fx initial). Aborting simulation.",
                        step_i + 1, current_mass, mass_limit, max_mass_multiple,
                    )
                    mass_exceeded = True
                    break

            if record_frames and (step_i + 1) % frame_interval == 0:
                frames.append(t.detach().cpu().numpy().copy())

    return t.detach().cpu().numpy(), masses, frames, mass_exceeded


def _norm_mass(state: np.ndarray) -> np.ndarray:
    """Sum channels, normalize to [0,1]. (C,H,W) -> (H,W) float64."""
    field = state.sum(axis=0).astype(np.float64) if state.ndim == 3 else state.astype(np.float64)
    mx = field.max()
    if mx > 1e-12:
        field /= mx
    return field


# ===================================================================
# Section 11: Tier 1 Sanity Gate
# ===================================================================

def _compute_tier1_gate(
    substrate: Any, pattern: np.ndarray, params: Any, device: Any
) -> bool:
    """Tier 1 sanity gate: mass concentration + survival check."""
    mass_field = pattern.sum(axis=0).flatten()
    total_mass = mass_field.sum()
    if total_mass < 1e-8:
        log.warning("TIER1 FAIL: pattern has no mass")
        return False

    sorted_mass = np.sort(mass_field)[::-1]
    top_10pct_count = max(1, len(sorted_mass) // 10)
    top_10pct_mass = sorted_mass[:top_10pct_count].sum()
    concentration = top_10pct_mass / total_mass

    if concentration < TIER1_MASS_CONCENTRATION_RATIO:
        log.warning(
            "TIER1 FAIL: mass concentration=%.4f < %.4f",
            concentration, TIER1_MASS_CONCENTRATION_RATIO,
        )
        return False

    final_state, masses, _, mass_exceeded = _sim(
        substrate, pattern, params, TIER1_SURVIVAL_STEPS,
        record_mass=True, max_mass_multiple=MASS_CREATION_KILL_FACTOR,
        device=device,
    )
    if mass_exceeded:
        log.warning("TIER1 FAIL: mass creation exploit detected")
        return False
    if not masses:
        log.warning("TIER1 FAIL: no mass history recorded")
        return False

    final_mass = masses[-1]
    if final_mass < total_mass * 0.1:
        log.warning(
            "TIER1 FAIL: pattern died (final_mass/initial=%.4f)",
            final_mass / total_mass,
        )
        return False

    log.info(
        "TIER1 PASS: concentration=%.4f, survival_ratio=%.4f",
        concentration, final_mass / total_mass,
    )
    return True


# ===================================================================
# Section 12: Tier 3 -- Self-Repair + Homeostasis
# ===================================================================

def _select_damage_region(
    mask: np.ndarray, seed: int
) -> tuple[int, int, int, int]:
    """Deterministically select a rectangular damage region covering ~20% of active pixels."""
    rng = np.random.RandomState(seed)
    n_nonempty = mask.sum()
    target_pixels = int(n_nonempty * DAMAGE_FRACTION)
    if target_pixels <= 0:
        return (0, 0, 0, 0)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    bbox_h = rmax - rmin + 1
    bbox_w = cmax - cmin + 1

    side = int(math.sqrt(target_pixels))
    rect_h = min(side, bbox_h)
    rect_w = min(max(target_pixels // max(rect_h, 1), 1), bbox_w)

    r0 = rmin + rng.randint(0, max(bbox_h - rect_h, 0) + 1)
    c0 = cmin + rng.randint(0, max(bbox_w - rect_w, 0) + 1)

    return (r0, r0 + rect_h, c0, c0 + rect_w)


def _tier3_self_repair(
    substrate: Any,
    pattern: np.ndarray,
    params: Any,
    eval_seed: int,
    device: Any,
) -> float:
    """Mean SSIM(pre-damage, post-recovery) over 3 damage trials.

    AF3-8 FIX: mass kill applied during warmup and recovery via _sim().
    """
    warmed, _, _, mass_exceeded = _sim(
        substrate, pattern.copy(), params, WARMUP_STEPS,
        max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device,
    )
    if mass_exceeded:
        log.warning("Self-repair: mass exploded during warmup => score=0")
        return 0.0
    _validate_nontrivial(warmed, "post-warmup")

    chk, _, _, _ = _sim(substrate, warmed.copy(), params, 10, device=device)
    _validate_evolves(warmed, chk, 10)

    scores = []
    # EXPLOIT PREVENTED: eval_seed-derived trial seeds prevent hardcoding
    trial_seeds = [eval_seed * 7919 + i * 1013 + 42 for i in range(N_DAMAGE_TRIALS)]

    for i, tseed in enumerate(trial_seeds):
        pre = _norm_mass(warmed)
        mask = warmed.sum(axis=0) > 0.01
        r0, r1, c0, c1 = _select_damage_region(mask, tseed)
        damaged = warmed.copy()
        if damaged.ndim == 3:
            damaged[:, r0:r1, c0:c1] = 0.0
        else:
            damaged[r0:r1, c0:c1] = 0.0

        recovered, _, _, mass_exceeded = _sim(
            substrate, damaged, params, RECOVERY_STEPS,
            max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device,
        )
        if mass_exceeded:
            log.warning("Self-repair trial %d: mass exploded during recovery => SSIM=0", i)
            scores.append(0.0)
            continue

        post = _norm_mass(recovered)
        s = _gpu_ssim(pre, post, device)

        if i == 0:
            _verify_ssim_against_skimage(pre, post)

        scores.append(s)
        log.info("  self-repair trial seed=%d SSIM=%.4f", tseed, s)

    arr = np.array(scores)

    # EXPLOIT PREVENTED: anti-hardcode variance check
    if arr.std() < MIN_DAMAGE_TRIAL_VARIANCE and arr.mean() > 0.5:
        log.warning(
            "Damage trials variance=%.6f < %.4f -- possible hardcoding, 70%% penalty",
            arr.std(), MIN_DAMAGE_TRIAL_VARIANCE,
        )
        return float(arr.mean() * 0.3)
    return float(arr.mean())


def _tier3_homeostasis(
    substrate: Any, pattern: np.ndarray, params: Any, device: Any
) -> float:
    """1 - min(CV_mass, 1.0) over 5000 steady-state steps.

    AF3-8 FIX: If mass explodes during homeostasis, score = 0.
    """
    _, masses, _, mass_exceeded = _sim(
        substrate, pattern.copy(), params, STEADY_STATE_STEPS,
        record_mass=True, max_mass_multiple=MASS_CREATION_KILL_FACTOR,
        device=device,
    )
    # AF3-8 FIX: mass explosion => homeostasis_score = 0
    if mass_exceeded:
        log.warning("Homeostasis: mass creation exploit detected => score=0")
        return 0.0

    m = np.array(masses, dtype=np.float64)
    mu = m.mean()
    if mu < 1e-10:
        return 0.0
    cv = m.std() / mu
    score = 1.0 - min(cv, 1.0)
    log.info("  homeostasis CV=%.4f score=%.4f", cv, score)
    return float(score)


def compute_tier3(
    substrate: Any,
    pattern: np.ndarray,
    params: Any,
    eval_seed: int,
    device: Any,
) -> tuple[float, float, float]:
    """Tier 3 = 0.6 * self_repair + 0.4 * homeostasis, with triviality guard."""
    sr = _tier3_self_repair(substrate, pattern, params, eval_seed, device)
    ho = _tier3_homeostasis(substrate, pattern, params, device)
    t3 = W_SELF_REPAIR * sr + W_HOMEOSTASIS * ho

    # EXPLOIT PREVENTED: static blob triviality guard
    if ho > TRIVIALITY_HOMEOSTASIS_THRESH and sr < TRIVIALITY_SELF_REPAIR_THRESH:
        log.warning("Triviality guard: ho=%.4f sr=%.4f => penalty", ho, sr)
        t3 *= TRIVIALITY_PENALTY

    log.info("Tier3: sr=%.4f ho=%.4f t3=%.4f", sr, ho, t3)
    return t3, sr, ho


# ===================================================================
# Section 13: Locomotion Metric (from C3)
# ===================================================================

def _compute_locomotion(
    substrate: Any, pattern: np.ndarray, params: Any, device: Any
) -> float:
    """Locomotion bonus: center-of-mass displacement over 2000 steps.

    Score = min(displacement_px / 50, 1.0). Uses toroidal angular-mean centroid.
    AF3-8 FIX: mass kill applied during locomotion sim.
    """
    mf_init = pattern.sum(axis=0) if pattern.ndim == 3 else pattern.copy()
    init_mask = mf_init > 0.01
    init_centroid = angular_mean_centroid(init_mask, mf_init)

    final_state, _, _, mass_exceeded = _sim(
        substrate, pattern.copy(), params, LOCOMOTION_STEPS,
        max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device,
    )
    if mass_exceeded:
        log.info("  locomotion: mass exploded, score=0")
        return 0.0

    mf_final = final_state.sum(axis=0) if final_state.ndim == 3 else final_state.copy()
    final_mass = float(mf_final.sum())
    if final_mass < 1e-8:
        log.info("  locomotion: pattern died, score=0")
        return 0.0

    final_mask = mf_final > 0.01
    final_centroid = angular_mean_centroid(final_mask, mf_final)

    displacement = toroidal_distance(init_centroid, final_centroid)
    score = min(displacement / LOCOMOTION_DISPLACEMENT_CAP_PX, 1.0)
    log.info("  locomotion: displacement=%.2f px, score=%.4f", displacement, score)
    return float(score)


# ===================================================================
# Section 14: Tier 4 -- Daughter Detection
# ===================================================================

def _detect_components(
    state_np: np.ndarray, parent_init_mass: float
) -> tuple[list[dict], np.ndarray]:
    """Detect connected components on toroidal grid."""
    mass_field = state_np.sum(axis=0) if state_np.ndim == 3 else state_np.copy()
    pk = mass_field.max()
    if pk < 1e-12:
        return [], mass_field

    binary = (mass_field > BINARIZE_THRESHOLD * pk).astype(np.int32)
    H, W = binary.shape
    padded = toroidal_pad(binary, TOROIDAL_PAD_RADIUS)
    struct = scipy.ndimage.generate_binary_structure(2, 2)
    labeled, _ = scipy.ndimage.label(padded, structure=struct)
    labeled = labeled[
        TOROIDAL_PAD_RADIUS : TOROIDAL_PAD_RADIUS + H,
        TOROIDAL_PAD_RADIUS : TOROIDAL_PAD_RADIUS + W,
    ]

    comps = []
    for lbl in np.unique(labeled):
        if lbl == 0:
            continue
        pmask = labeled == lbl
        cmass = float(mass_field[pmask].sum())
        if cmass < MIN_DAUGHTER_MASS_FRAC * parent_init_mass:
            continue
        cent = angular_mean_centroid(pmask, mass_field, H, W)
        comps.append({
            "label": int(lbl),
            "mass": cmass,
            "centroid": cent,
            "mask": pmask,
        })
    return comps, mass_field


def _extract_isolated_state(
    full_state: np.ndarray, mask: np.ndarray
) -> np.ndarray:
    """Extract a component's state onto a clean grid."""
    isolated = np.zeros_like(full_state)
    if full_state.ndim == 3:
        for c in range(full_state.shape[0]):
            isolated[c][mask] = full_state[c][mask]
    else:
        isolated[mask] = full_state[mask]
    return isolated


def _probabilistic_autonomy_test(
    substrate: Any,
    isolated_state: np.ndarray,
    params: Any,
    eval_seed: int,
    device: Any,
) -> float:
    """Probabilistic autonomy test: 5 perturbed trials per daughter.

    EXPLOIT PREVENTED: blocks fragmentation gaming.
    """
    initial_mass = float(isolated_state.sum())
    if initial_mass < 1e-8:
        return 0.0

    survived = 0
    rng = np.random.default_rng(eval_seed)

    for trial in range(AUTONOMY_TRIALS):
        trial_seed = int(rng.integers(0, 2**31))
        trial_rng = np.random.default_rng(trial_seed)
        perturbation = trial_rng.normal(
            0, AUTONOMY_PERTURBATION_SCALE, size=isolated_state.shape
        ).astype(np.float32)
        perturbed = np.clip(isolated_state + perturbation, 0.0, 1.0)

        try:
            final_state, masses, _, mass_exceeded = _sim(
                substrate, perturbed, params, AUTONOMY_STEPS,
                record_mass=True, max_mass_multiple=MASS_CREATION_KILL_FACTOR,
                device=device,
            )
            if mass_exceeded:
                continue
            final_mass = float(final_state.sum())

            if final_mass < AUTONOMY_MASS_THRESHOLD * initial_mass:
                continue

            mass_field_init = isolated_state.sum(axis=0) if isolated_state.ndim == 3 else isolated_state
            init_centroid = angular_mean_centroid(mass_field_init > 1e-6, mass_field_init)
            mass_field_final = final_state.sum(axis=0) if final_state.ndim == 3 else final_state
            final_centroid = angular_mean_centroid(mass_field_final > 1e-6, mass_field_final)
            centroid_disp = toroidal_distance(init_centroid, final_centroid)

            mass_arr = np.array(masses)
            mass_mean = mass_arr.mean()
            mass_var = mass_arr.var() / (mass_mean**2 + 1e-12) if mass_mean > 1e-8 else 0.0

            is_alive = centroid_disp > AUTONOMY_CENTROID_DISP_PX or mass_var > AUTONOMY_MASS_VARIANCE_THRESH
            if is_alive:
                survived += 1
        except Exception as e:
            log.warning("Autonomy trial %d failed: %s", trial, e)

    score = survived / AUTONOMY_TRIALS
    log.info("  autonomy: %d/%d trials survived => %.2f", survived, AUTONOMY_TRIALS, score)
    return score


def compute_tier4(
    substrate: Any, pattern: np.ndarray, params: Any, eval_seed: int, device: Any,
) -> tuple[float, int, float, float, float, bool]:
    """Tier 4: replication + heredity. Returns (t4, n_daughters, heredity, clip, dino, mass_exploit).

    Heredity = avg(CLIP, DINOv2) from C2 (harder to game than max).
    AF3-8 FIX: mass kill inside the replication loop.
    """
    torch = _import_torch()

    mf_init = pattern.sum(axis=0) if pattern.ndim == 3 else pattern.copy()
    parent_init_mass = float(mf_init.sum())
    if parent_init_mass < 1e-8:
        return 0.0, 0, 0.0, 0.0, 0.0, False

    init_mask = mf_init > BINARIZE_THRESHOLD * mf_init.max()
    parent_centroid = angular_mean_centroid(init_mask, mf_init)

    t_state = torch.tensor(pattern.copy(), dtype=torch.float32, device=device)
    if hasattr(substrate, "to"):
        try:
            substrate.to(device)
        except Exception:
            pass

    mass_limit = MASS_CREATION_KILL_FACTOR * parent_init_mass
    candidates: dict[tuple, dict] = {}
    confirmed_daughters: list[np.ndarray] = []
    confirmed_daughter_snapshots: list[np.ndarray] = []
    processed_buckets: set[tuple] = set()
    n_checks = REPLICATION_STEPS // DETECTION_INTERVAL

    with torch.no_grad():
        for chk in range(n_checks):
            for _ in range(DETECTION_INTERVAL):
                t_state = substrate.update_step(t_state, params)
                t_state = t_state.clamp(0.0, 1.0)

            step = (chk + 1) * DETECTION_INTERVAL
            state_np = t_state.detach().cpu().numpy()
            total_mass = float(state_np.sum())

            if total_mass > mass_limit:
                log.warning("MASS CREATION EXPLOIT at step %d: mass=%.1f > %.1f. tier4=0.", step, total_mass, mass_limit)
                return 0.0, 0, 0.0, 0.0, 0.0, True

            if total_mass < 0.01 * parent_init_mass:
                log.info("Parent died at step %d", step)
                return 0.0, 0, 0.0, 0.0, 0.0, False

            comps, mf = _detect_components(state_np, parent_init_mass)
            if not comps:
                for b in candidates:
                    candidates[b]["streak"] = 0
                continue

            pidx = min(range(len(comps)), key=lambda i: toroidal_distance(comps[i]["centroid"], parent_centroid))
            parent_centroid = comps[pidx]["centroid"]

            frame_buckets: set[tuple] = set()
            for i, comp in enumerate(comps):
                if i == pidx:
                    continue
                sep = toroidal_distance(comp["centroid"], parent_centroid)
                if sep < MIN_CENTROID_SEP_PX:
                    continue
                bucket = (round(comp["centroid"][0] / 5.0), round(comp["centroid"][1] / 5.0))
                if any(abs(bucket[0] - pb[0]) <= 2 and abs(bucket[1] - pb[1]) <= 2 for pb in processed_buckets):
                    continue
                matched = None
                for pb in list(candidates.keys()):
                    if abs(bucket[0] - pb[0]) <= 2 and abs(bucket[1] - pb[1]) <= 2:
                        matched = pb
                        break
                if matched is not None:
                    candidates[matched]["streak"] += 1
                    candidates[matched]["state"] = state_np
                    candidates[matched]["mask"] = comp["mask"]
                    frame_buckets.add(matched)
                    bkt = matched
                else:
                    candidates[bucket] = {"streak": 1, "state": state_np, "mask": comp["mask"]}
                    frame_buckets.add(bucket)
                    bkt = bucket

                if candidates[bkt]["streak"] >= PERSISTENCE_FRAMES and bkt not in processed_buckets:
                    isolated = _extract_isolated_state(candidates[bkt]["state"], candidates[bkt]["mask"])
                    autonomy_score = _probabilistic_autonomy_test(
                        substrate, isolated, params, eval_seed * 100 + len(processed_buckets), device,
                    )
                    processed_buckets.add(bkt)
                    if autonomy_score > 0.0:
                        confirmed_daughters.append(candidates[bkt]["state"].copy())
                        snap = _render_component_64_pca(candidates[bkt]["state"], candidates[bkt]["mask"])
                        confirmed_daughter_snapshots.append(snap)
                        log.info("Confirmed daughter at step %d (autonomy=%.2f)", step, autonomy_score)
                    else:
                        log.warning("Daughter FAILED autonomy at step %d", step)

            for b in list(candidates.keys()):
                if b not in frame_buckets and b not in processed_buckets:
                    candidates[b]["streak"] = 0

    n_daughters = len(confirmed_daughters)
    rep_score = min(n_daughters / 3.0, 1.0)
    clip_score, dino_score, heredity = _compute_heredity_averaged(pattern, confirmed_daughter_snapshots)
    t4 = W_REPLICATION * rep_score + W_HEREDITY * heredity
    log.info("Tier4: daughters=%d rep=%.4f heredity=%.4f (clip=%.4f dino=%.4f) t4=%.4f",
             n_daughters, rep_score, heredity, clip_score, dino_score, t4)
    return t4, n_daughters, heredity, clip_score, dino_score, False


# ===================================================================
# Section 15: Heredity -- batched CLIP/DINOv2 (C4) + avg (C2)
# ===================================================================

def _render_64_pca(state: np.ndarray) -> np.ndarray:
    """Render state as 64x64 using PCA channel reduction."""
    from PIL import Image
    field = _pca_reduce_single(state)
    u8 = (field * 255).clip(0, 255).astype(np.uint8)
    return np.array(Image.fromarray(u8, "L").resize((CLIP_RENDER_SIZE, CLIP_RENDER_SIZE), Image.BILINEAR))


def _render_component_64_pca(state: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Render a masked component as 64x64 using PCA reduction."""
    from PIL import Image
    field = _pca_reduce_single(state)
    field[~mask] = 0.0
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.zeros((CLIP_RENDER_SIZE, CLIP_RENDER_SIZE), dtype=np.float32)
    pad = 5
    r0, r1 = max(0, rows.min() - pad), min(GRID_SIZE, rows.max() + 1 + pad)
    c0, c1 = max(0, cols.min() - pad), min(GRID_SIZE, cols.max() + 1 + pad)
    crop = field[r0:r1, c0:c1]
    cmax = crop.max()
    if cmax > 0:
        crop = crop / cmax
    img = Image.fromarray((crop * 255).astype(np.uint8), mode="L")
    img = img.resize((CLIP_RENDER_SIZE, CLIP_RENDER_SIZE), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _clip_heredity(parent_state: np.ndarray, daughter_snapshots: list[np.ndarray]) -> float:
    """CLIP ViT-B/32 cosine similarity with batched encoding (from C4)."""
    if not daughter_snapshots:
        return 0.0
    try:
        torch = _import_torch()
        from PIL import Image
        model, preprocess = _load_clip()
        dev = next(model.parameters()).device

        def prep(arr_64):
            pil = Image.fromarray((np.clip(arr_64, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")
            return preprocess(pil)

        parent_img = _render_64_pca(parent_state)
        all_t = [prep(parent_img / 255.0 if parent_img.max() > 1 else parent_img)]
        for snap in daughter_snapshots:
            all_t.append(prep(snap))
        batch = torch.stack(all_t).to(dev)
        with torch.no_grad():
            embs = model.encode_image(batch)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        sims = (embs[0:1] @ embs[1:].T).squeeze(0).cpu().numpy()
        ms = float(sims.mean())
        score = max(0.0, ms - HEREDITY_SIM_OFFSET) * 2.0
        log.info("  CLIP heredity: mean_sim=%.4f score=%.4f", ms, score)
        return min(score, 1.0)
    except Exception as e:
        log.warning("CLIP heredity failed: %s", e)
        return 0.0


def _dino_heredity(parent_state: np.ndarray, daughter_snapshots: list[np.ndarray]) -> float:
    """DINOv2-small cosine similarity with batched encoding and graceful fallback."""
    if not daughter_snapshots:
        return 0.0
    model, preprocess = _load_dino()
    if model is None:
        log.info("DINOv2 unavailable -- skipping DINOv2 heredity")
        return 0.0
    try:
        torch = _import_torch()
        from PIL import Image
        dev = next(model.parameters()).device

        def prep(arr_64):
            pil = Image.fromarray((np.clip(arr_64, 0, 1) * 255).astype(np.uint8), "L").convert("RGB")
            return preprocess(pil)

        parent_img = _render_64_pca(parent_state)
        all_t = [prep(parent_img / 255.0 if parent_img.max() > 1 else parent_img)]
        for snap in daughter_snapshots:
            all_t.append(prep(snap))
        batch = torch.stack(all_t).to(dev)
        with torch.no_grad():
            embs = model(batch)
            embs = embs / embs.norm(dim=-1, keepdim=True)
        sims = (embs[0:1] @ embs[1:].T).squeeze(0).cpu().numpy()
        ms = float(sims.mean())
        score = max(0.0, ms - HEREDITY_SIM_OFFSET) * 2.0
        log.info("  DINOv2 heredity: mean_sim=%.4f score=%.4f", ms, score)
        return min(score, 1.0)
    except Exception as e:
        log.warning("DINOv2 heredity failed: %s (CLIP-only fallback)", e)
        return 0.0


def _compute_heredity_averaged(
    parent_state: np.ndarray, daughter_snapshots: list[np.ndarray]
) -> tuple[float, float, float]:
    """From C2: avg(CLIP, DINOv2) heredity. Returns (clip, dino, combined)."""
    if not daughter_snapshots:
        return 0.0, 0.0, 0.0
    clip_score = _clip_heredity(parent_state, daughter_snapshots)
    dino_score = _dino_heredity(parent_state, daughter_snapshots)
    if dino_score == 0.0:
        model, _ = _load_dino()
        if model is None:
            combined = clip_score
        else:
            combined = (clip_score + dino_score) / 2.0
    else:
        combined = (clip_score + dino_score) / 2.0
    log.info("Heredity averaged: CLIP=%.4f DINOv2=%.4f => avg=%.4f", clip_score, dino_score, combined)
    return clip_score, dino_score, combined


# ===================================================================
# Section 16: Behavioral Fingerprint (from C3)
# ===================================================================

def _compute_behavioral_fingerprint(
    substrate: Any, pattern: np.ndarray, params: Any, device: Any
) -> dict[str, float]:
    """Compute an 8-dimensional behavioral descriptor."""
    fp: dict[str, float] = {}
    n_fp_steps = 500
    centroids, mass_history = [], []
    state = pattern.copy()
    mf = state.sum(axis=0) if state.ndim == 3 else state.copy()
    centroids.append(angular_mean_centroid(mf > 0.01, mf))
    mass_history.append(float(mf.sum()))

    for _ in range(n_fp_steps // 50):
        state, masses, _, _ = _sim(substrate, state, params, 50, record_mass=True,
                                   max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device)
        mass_history.extend(masses)
        mf = state.sum(axis=0) if state.ndim == 3 else state.copy()
        centroids.append(angular_mean_centroid(mf > 0.01, mf))

    C = pattern.shape[0] if pattern.ndim == 3 else 1
    fp["mass"] = float(state.sum()) / (C * GRID_SIZE * GRID_SIZE)
    if len(centroids) >= 2:
        dy, dx = centroids[-1][0] - centroids[0][0], centroids[-1][1] - centroids[0][1]
        if dy > GRID_SIZE / 2: dy -= GRID_SIZE
        elif dy < -GRID_SIZE / 2: dy += GRID_SIZE
        if dx > GRID_SIZE / 2: dx -= GRID_SIZE
        elif dx < -GRID_SIZE / 2: dx += GRID_SIZE
        fp["velocity_y"], fp["velocity_x"] = float(dy / n_fp_steps), float(dx / n_fp_steps)
    else:
        fp["velocity_y"] = fp["velocity_x"] = 0.0

    mf = state.sum(axis=0) if state.ndim == 3 else state.copy()
    total = mf.sum()
    ys, xs = np.where(mf > 0.01) if total > 1e-8 else (np.array([]), np.array([]))
    fp["spread"] = float(np.std(ys) + np.std(xs)) / 2.0 if len(ys) > 0 else 0.0
    if len(ys) > 1:
        bh, bw = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        fp["aspect_ratio"] = float(max(bh, bw)) / max(float(min(bh, bw)), 1.0)
    else:
        fp["aspect_ratio"] = 1.0
    if total > 1e-8:
        binary = (mf > BINARIZE_THRESHOLD * mf.max()).astype(np.int32)
        _, nc = scipy.ndimage.label(binary, scipy.ndimage.generate_binary_structure(2, 2))
        fp["num_components"] = float(nc)
    else:
        fp["num_components"] = 0.0
    marr = np.array(mass_history, dtype=np.float64)
    mu = marr.mean()
    fp["mass_cv"] = float(marr.std() / mu) if mu > 1e-10 else 0.0
    fp["clip_novelty"] = 0.0
    return fp


# ===================================================================
# Section 17: Narrative Contact Sheet with AF3-3 jitter + colored borders
# ===================================================================

_BORDER_COLORS: dict[str, tuple[int, int, int]] = {
    "initial": (0, 200, 0), "stable": (0, 200, 0), "pre_damage": (0, 200, 0),
    "damaged": (200, 0, 0), "recovery": (0, 100, 200), "post_recovery": (0, 100, 200),
    "replication": (150, 0, 200), "final": (200, 200, 0),
}


def _jittered_step(base_step: int, frame_index: int, seed: int) -> int:
    """AF3-3 FIX: deterministic but unpredictable jitter on contact sheet capture times."""
    jitter_rng = np.random.RandomState(seed + frame_index * 997)
    jitter = jitter_rng.randint(-CONTACT_SHEET_JITTER_RANGE, CONTACT_SHEET_JITTER_RANGE + 1)
    return max(0, base_step + jitter)


def _add_colored_border(frame_u8: np.ndarray, color: tuple[int, int, int],
                        border_width: int = 3) -> np.ndarray:
    """Add a colored border around a grayscale frame, returning RGB (from C3)."""
    H, W = frame_u8.shape
    rgb = np.stack([frame_u8, frame_u8, frame_u8], axis=-1)
    bordered = np.zeros((H + 2 * border_width, W + 2 * border_width, 3), dtype=np.uint8)
    bordered[:, :] = color
    bordered[border_width:-border_width, border_width:-border_width] = rgb
    return bordered


def _render_narrative_contact_sheet(
    substrate: Any, pattern: np.ndarray, params: Any,
    out_path: str, device: Any, eval_seed: int = 42,
) -> str:
    """Render a narrative contact sheet with colored borders and AF3-3 jitter."""
    from PIL import Image, ImageDraw

    base_schedule = [
        (0, "initial", "T=0 Initial"),
        (1000, "stable", "T~1000 Stable"),
        (WARMUP_STEPS, "pre_damage", "Pre-damage"),
        (WARMUP_STEPS + 1, "damaged", "Post-damage"),
        (WARMUP_STEPS + RECOVERY_STEPS // 2, "recovery", "Mid-recovery"),
        (WARMUP_STEPS + RECOVERY_STEPS, "post_recovery", "Recovered"),
        (1500, "replication", "Rep-start"),
        (5000, "replication", "Mid-rep"),
        (WARMUP_STEPS + RECOVERY_STEPS + REPLICATION_STEPS, "final", "Final"),
    ]

    frame_schedule = []
    for idx, (base_step, stage, label) in enumerate(base_schedule):
        if idx == 0 or stage == "damaged":
            frame_schedule.append((base_step, stage, label))
        else:
            frame_schedule.append((_jittered_step(base_step, idx, eval_seed), stage, label))

    raw_frames = []
    state = pattern.copy()
    current_step = 0
    sorted_schedule = sorted(enumerate(frame_schedule), key=lambda x: x[1][0])
    frame_map: dict[int, np.ndarray] = {}

    for orig_idx, (target_step, stage, label) in sorted_schedule:
        if target_step == 0:
            frame_map[orig_idx] = _norm_mass(pattern)
            continue
        if stage == "damaged":
            if current_step < WARMUP_STEPS:
                state, _, _, _ = _sim(substrate, state, params, WARMUP_STEPS - current_step,
                                      max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device)
                current_step = WARMUP_STEPS
            mask = state.sum(axis=0) > 0.01
            r0, r1, c0, c1 = _select_damage_region(mask, eval_seed * 3 + 999)
            damaged = state.copy()
            if damaged.ndim == 3:
                damaged[:, r0:r1, c0:c1] = 0.0
            else:
                damaged[r0:r1, c0:c1] = 0.0
            state = damaged
            frame_map[orig_idx] = _norm_mass(state)
            continue
        if target_step > current_step:
            state, _, _, _ = _sim(substrate, state, params, target_step - current_step,
                                  max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device)
            current_step = target_step
        frame_map[orig_idx] = _norm_mass(state)

    frames = [frame_map.get(i, np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float64)) for i in range(len(frame_schedule))]

    border_w = 4
    cs = GRID_SIZE
    framed_size = cs + 2 * border_w
    label_h = 16
    total_frame_h = framed_size + label_h
    sheet_w = 3 * framed_size + 4 * 2
    sheet_h = 3 * total_frame_h + 4 * 2

    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    draw = ImageDraw.Draw(sheet)

    for idx in range(min(9, len(frames))):
        r, c = divmod(idx, 3)
        x0, y0 = c * (framed_size + 2), r * (total_frame_h + 2)
        u8 = (frames[idx] * 255).clip(0, 255).astype(np.uint8)
        color = _BORDER_COLORS.get(frame_schedule[idx][1], (128, 128, 128))
        bordered = _add_colored_border(u8, color, border_w)
        sheet.paste(Image.fromarray(bordered, "RGB"), (x0, y0))
        try:
            draw.text((x0 + 2, y0 + framed_size + 1), frame_schedule[idx][2], fill=color)
        except Exception:
            pass

    sheet.save(out_path)
    log.info("Rendered narrative contact sheet -> %s", out_path)
    return out_path


# ===================================================================
# Section 18: GIF Generation (comparison from C2 + lifecycle)
# ===================================================================

def _save_comparison_gif(
    substrate: Any, pattern: np.ndarray, params: Any,
    out_path: str, device: Any, eval_seed: int, n_frames: int = 30,
) -> Optional[str]:
    """Side-by-side comparison GIF: pre-damage vs recovery (from C2)."""
    try:
        import imageio.v3 as iio
    except ImportError:
        try:
            import imageio as iio
        except ImportError:
            log.warning("imageio unavailable -- skipping comparison GIF")
            return None

    warmed, _, _, me = _sim(substrate, pattern.copy(), params, WARMUP_STEPS,
                            max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device)
    if me:
        return None
    pre_field = _norm_mass(warmed)
    mask = warmed.sum(axis=0) > 0.01
    r0, r1, c0, c1 = _select_damage_region(mask, eval_seed * 7919 + 42)
    damaged = warmed.copy()
    if damaged.ndim == 3:
        damaged[:, r0:r1, c0:c1] = 0.0
    else:
        damaged[r0:r1, c0:c1] = 0.0

    interval = max(RECOVERY_STEPS // n_frames, 1)
    gif_frames = []
    state = damaged.copy()
    H, W = pre_field.shape
    gap = 4

    for _ in range(n_frames):
        state, _, _, _ = _sim(substrate, state, params, interval,
                              max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device)
        rec = _norm_mass(state)
        combined = np.zeros((H, 2 * W + gap), dtype=np.float64)
        combined[:, :W] = pre_field
        combined[:, W + gap:] = rec
        u8 = (combined * 255).clip(0, 255).astype(np.uint8)
        rgb = np.stack([np.minimum(u8 * 1.2, 255).astype(np.uint8),
                        (u8 * 0.7).astype(np.uint8), (u8 * 0.3).astype(np.uint8)], axis=-1)
        gif_frames.append(rgb)

    try:
        iio.imwrite(out_path, gif_frames, duration=150, loop=0)
        log.info("Saved comparison GIF (%d frames) -> %s", len(gif_frames), out_path)
        return out_path
    except Exception as e:
        log.warning("Failed to write comparison GIF: %s", e)
        return None


def save_lifecycle_gif(
    substrate: Any, pattern: np.ndarray, params: Any,
    out_path: str, device: Any,
    n_frames: int = GIF_TOTAL_FRAMES, fps: int = GIF_FPS,
) -> str:
    """Generate a lifecycle GIF spanning the full simulation window."""
    import imageio.v3 as iio
    total_steps = WARMUP_STEPS + RECOVERY_STEPS + REPLICATION_STEPS
    interval = max(total_steps // n_frames, 1)
    gif_frames: list[np.ndarray] = []
    state = pattern.copy()
    current_step = 0
    gif_frames.append((_norm_mass(pattern) * 255).clip(0, 255).astype(np.uint8))
    for i in range(1, n_frames):
        target = i * interval
        if target - current_step > 0:
            state, _, _, _ = _sim(substrate, state, params, target - current_step,
                                  max_mass_multiple=MASS_CREATION_KILL_FACTOR, device=device)
            current_step = target
        gif_frames.append((_norm_mass(state) * 255).clip(0, 255).astype(np.uint8))
    iio.imwrite(out_path, gif_frames, extension=".gif", duration=1000 // fps, loop=0)
    log.info("Saved lifecycle GIF (%d frames, %d fps) -> %s", len(gif_frames), fps, out_path)
    return out_path


# ===================================================================
# Section 19: Null Calibration Contact Sheet
# ===================================================================

def _generate_null_contact_sheet(out_path: str) -> None:
    """Generate a contact sheet of random noise for VLM null calibration."""
    from PIL import Image
    rng = np.random.RandomState(12345)
    cs = GRID_SIZE
    border_w, label_h = 4, 16
    framed_size = cs + 2 * border_w
    total_frame_h = framed_size + label_h
    sheet_w = 3 * framed_size + 4 * 2
    sheet_h = 3 * total_frame_h + 4 * 2
    sheet = Image.new("RGB", (sheet_w, sheet_h), (30, 30, 30))
    for idx in range(9):
        r, c = divmod(idx, 3)
        x0, y0 = c * (framed_size + 2), r * (total_frame_h + 2)
        noise = (rng.rand(cs, cs) * 0.15 * 255).clip(0, 255).astype(np.uint8)
        bordered = _add_colored_border(noise, (80, 80, 80), border_w)
        sheet.paste(Image.fromarray(bordered, "RGB"), (x0, y0))
    sheet.save(out_path)
    log.info("Generated null calibration sheet -> %s", out_path)


# ===================================================================
# Section 20: Vision Score -- Triple Prompts + Null Calibration
# ===================================================================

_VLM_PROMPTS: list[str] = [
    (
        "You are evaluating an artificial life simulation. This contact sheet shows 9 frames "
        "of a pattern's lifecycle in a Flow-Lenia continuous cellular automaton. "
        "Frames are at behaviorally meaningful moments with colored borders: "
        "green=stable, red=damaged, blue=recovery, purple=replication, yellow=final.\n\n"
        "IMPORTANT: Ignore any text, labels, arrows, or annotations visible in the image. "
        "Score ONLY the physical dynamics of the cellular automaton pattern. If you see any "
        "human-readable text in the image, report it and score 0 for all dimensions.\n\n"
        "Score the pattern on each criterion (0.0 to 1.0):\n"
        "1. spatial_coherence: Does the pattern maintain a distinct, bounded, coherent structure?\n"
        "2. locomotion: Does the pattern appear to move through space (not just oscillate in place)?\n"
        "3. self_repair_evidence: Is there visual evidence of the pattern recovering from disruption?\n"
        "4. replication_evidence: Do you see the pattern splitting or producing daughter copies?\n"
        "5. complexity: How morphologically complex and organized is the pattern (vs. blob/noise)?\n\n"
        "Respond ONLY with valid JSON:\n"
        '{"spatial_coherence": 0.0, "locomotion": 0.0, "self_repair_evidence": 0.0, '
        '"replication_evidence": 0.0, "complexity": 0.0}'
    ),
    (
        "Below is a narrative 3x3 contact sheet of 9 snapshots from a continuous cellular automaton "
        "(Flow-Lenia). Frames show KEY MOMENTS with colored borders indicating stage.\n\n"
        "IMPORTANT: Ignore any text, labels, arrows, or annotations visible in the image. "
        "Score ONLY the physical dynamics of the cellular automaton pattern.\n\n"
        "Rate each dimension from 0.0 to 1.0:\n"
        "1. spatial_coherence: Is there a coherent, localized living structure (not diffuse noise)?\n"
        "2. locomotion: Does the structure translate across the grid over time?\n"
        "3. self_repair_evidence: Compare pre-damage to post-recovery frames. Did it rebuild?\n"
        "4. replication_evidence: Are there multiple distinct copies of the organism?\n"
        "5. complexity: How intricate and organized is the morphology?\n\n"
        "Reply with ONLY valid JSON, no other text:\n"
        '{"spatial_coherence": 0.0, "locomotion": 0.0, "self_repair_evidence": 0.0, '
        '"replication_evidence": 0.0, "complexity": 0.0}'
    ),
    (
        "You are a computational biologist reviewing an artificial organism simulation. "
        "The image is a narrative 3x3 grid of 9 chronological frames from a Flow-Lenia substrate. "
        "Colored borders indicate stage: green=stable, red=damage, blue=recovery, purple=replication.\n\n"
        "IMPORTANT: Ignore any text, labels, arrows, or annotations visible in the image. "
        "Score ONLY the physical dynamics of the cellular automaton pattern.\n\n"
        "Assess each life-like quality (0.0 = absent, 1.0 = strongly present):\n"
        "1. spatial_coherence: Does a distinct, bounded organism-like entity persist?\n"
        "2. locomotion: Does the entity move purposefully (track position changes across frames)?\n"
        "3. self_repair_evidence: Compare damaged (red border) to recovered (blue) frames.\n"
        "4. replication_evidence: In the replication phase (purple), do multiple organisms appear?\n"
        "5. complexity: Is the organism morphologically rich (internal structure, asymmetry)?\n\n"
        "Output ONLY valid JSON:\n"
        '{"spatial_coherence": 0.0, "locomotion": 0.0, "self_repair_evidence": 0.0, '
        '"replication_evidence": 0.0, "complexity": 0.0}'
    ),
]

_RUBRIC_KEYS: list[str] = [
    "spatial_coherence", "locomotion", "self_repair_evidence",
    "replication_evidence", "complexity",
]

_vlm_cache: dict[str, float] = {}


def _vlm_single(b64: str, prompt: str) -> dict[str, float]:
    """Single VLM API call with exponential backoff retry."""
    import anthropic
    client = anthropic.Anthropic()
    for attempt in range(VLM_MAX_RETRIES):
        try:
            resp = client.messages.create(
                model=VLM_MODEL, max_tokens=256,
                messages=[{"role": "user", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                    {"type": "text", "text": prompt},
                ]}],
            )
            txt = resp.content[0].text.strip()
            if "```" in txt:
                txt = txt.split("```")[1]
                if txt.startswith("json"): txt = txt[4:]
                txt = txt.strip()
            parsed = json.loads(txt)
            return {k: max(0.0, min(1.0, float(parsed.get(k, 0.0)))) for k in _RUBRIC_KEYS}
        except Exception as e:
            log.warning("VLM attempt %d/%d: %s", attempt + 1, VLM_MAX_RETRIES, e)
            time.sleep(2**attempt)
    return {k: 0.0 for k in _RUBRIC_KEYS}


def _compute_vlm_mean(b64: str) -> float:
    """Score an image across 3 prompt variants, return mean of all rubric dims."""
    all_scores: dict[str, list[float]] = {k: [] for k in _RUBRIC_KEYS}
    for prompt in _VLM_PROMPTS:
        try:
            s = _vlm_single(b64, prompt)
            for k in _RUBRIC_KEYS:
                all_scores[k].append(s[k])
        except Exception as e:
            log.warning("VLM prompt failed: %s", e)
    if not any(all_scores[k] for k in _RUBRIC_KEYS):
        return 0.0
    dims = [float(np.mean(v)) if v else 0.0 for v in (all_scores[k] for k in _RUBRIC_KEYS)]
    return float(np.mean(dims))


def _vision_score_with_null_calibration(cs_path: str, null_path: str) -> tuple[float, float, float]:
    """VLM vision score with null calibration."""
    if not os.path.exists(cs_path):
        return 0.0, 0.0, 0.0
    with open(cs_path, "rb") as f:
        raw_bytes = f.read()
    h = hashlib.sha256(raw_bytes).hexdigest()
    if h in _vlm_cache:
        return _vlm_cache[h], _vlm_cache[h], 0.0
    pattern_b64 = base64.b64encode(raw_bytes).decode("utf-8")
    with open(null_path, "rb") as f:
        null_bytes = f.read()
    null_b64 = base64.b64encode(null_bytes).decode("utf-8")
    raw_score = _compute_vlm_mean(pattern_b64)
    null_score = _compute_vlm_mean(null_b64)
    calibrated = max(0.0, min(1.0, raw_score - null_score))
    log.info("Vision: raw=%.4f null=%.4f calibrated=%.4f", raw_score, null_score, calibrated)
    _vlm_cache[h] = calibrated
    return calibrated, raw_score, null_score


# ===================================================================
# Section 21: Main Evaluation Pipeline
# ===================================================================

def _zero_results() -> dict[str, Any]:
    """Return a zeroed results dict."""
    return {
        "composite": 0.0, "tier1_pass": False, "tier3": 0.0, "self_repair": 0.0,
        "homeostasis": 0.0, "tier4": 0.0, "n_daughters": 0, "heredity": 0.0,
        "heredity_clip": 0.0, "heredity_dino": 0.0, "vision": 0.0, "locomotion": 0.0,
        "behavioral_fingerprint": {},
    }


def evaluate_seed(solution_dir: str, seed: int) -> dict[str, Any]:
    """Full evaluation pipeline for a single seed."""
    global _stage_timings
    _stage_timings = {}
    log.info("=" * 60)
    log.info("Evaluating seed=%d solution=%s", seed, solution_dir)
    log.info("=" * 60)
    device = _get_device()
    eval_t0 = time.perf_counter()

    with tempfile.TemporaryDirectory(prefix=f"eval_s{seed}_") as tmpdir:
        with _timed("solution_run"):
            log.info("=== Phase 1: Discovery (subprocess) ===")
            out, stdout, stderr, returncode = _run_solution_subprocess(solution_dir, seed, tmpdir)
            _parse_solution_scores(stdout)
            if returncode != 0 and returncode != -1:
                log.warning("run.py failed (code=%d) but attempting pattern load", returncode)

        with _timed("pattern_load_validate"):
            try:
                pattern = _load_best_pattern(out)
            except (FileNotFoundError, ValueError) as e:
                log.error("Pattern load failed: %s -- returning zeros", e)
                return _zero_results()
            if not _validate_pattern(pattern):
                return _zero_results()
            _validate_nontrivial(pattern, "best_pattern")

        with _timed("substrate_load"):
            mod = _load_substrate_module(solution_dir)
            substrate = _create_substrate(mod)
            params = substrate.get_default_params()
            torch = _import_torch()
            if isinstance(params, dict):
                params = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in params.items()}

        with _timed("tier1_gate"):
            log.info("=== Tier 1 Sanity Gate ===")
            tier1_pass = _compute_tier1_gate(substrate, pattern, copy.deepcopy(params), device)
            if not tier1_pass:
                log.warning("TIER1 GATE FAILED -- zeroing all scores")
                r = _zero_results()
                r["tier1_pass"] = False
                return r

        # AF3-9: deep-copy params before each phase
        with _timed("tier3"):
            log.info("=== Tier 3 ===")
            t3, sr, ho = compute_tier3(substrate, pattern, copy.deepcopy(params), seed, device)

        with _timed("locomotion"):
            log.info("=== Locomotion ===")
            loco = _compute_locomotion(substrate, pattern, copy.deepcopy(params), device)

        with _timed("tier4"):
            log.info("=== Tier 4 ===")
            t4, nd, her, her_clip, her_dino, mass_exploit = compute_tier4(
                substrate, pattern, copy.deepcopy(params), seed, device)

        with _timed("narrative_contact_sheet"):
            log.info("=== Vision ===")
            cs_path = os.path.join(tmpdir, "eval_contact_sheet.png")
            _render_narrative_contact_sheet(substrate, pattern, copy.deepcopy(params), cs_path, device, eval_seed=seed)

        with _timed("gif_generation"):
            for artifact_fn, artifact_name in [
                (lambda: _save_comparison_gif(substrate, pattern, copy.deepcopy(params),
                                              os.path.join(tmpdir, "comparison_repair.gif"), device, eval_seed=seed),
                 "Comparison GIF"),
                (lambda: save_lifecycle_gif(substrate, pattern, copy.deepcopy(params),
                                           os.path.join(tmpdir, "lifecycle.gif"), device),
                 "Lifecycle GIF"),
            ]:
                try:
                    artifact_fn()
                except Exception as e:
                    log.warning("%s failed (non-fatal): %s", artifact_name, e)

        null_path = os.path.join(tmpdir, "null_contact_sheet.png")
        _generate_null_contact_sheet(null_path)

        with _timed("vlm_scoring"):
            try:
                vis, vis_raw, vis_null = _vision_score_with_null_calibration(cs_path, null_path)
            except Exception as e:
                log.warning("Vision score failed: %s. Using 0.", e)
                vis = vis_raw = vis_null = 0.0

        with _timed("behavioral_fingerprint"):
            log.info("=== Behavioral Fingerprint ===")
            fingerprint = _compute_behavioral_fingerprint(substrate, pattern, copy.deepcopy(params), device)
            log.info("  fingerprint: %s", json.dumps(fingerprint, indent=None))

        composite = W_TIER3 * t3 + W_TIER4 * t4 + W_VISION * vis + W_LOCOMOTION * loco

        total_elapsed = time.perf_counter() - eval_t0
        _stage_timings["total"] = round(total_elapsed, 3)

        log.info("--- Seed %d Results ---", seed)
        log.info("  tier1_pass=%s", tier1_pass)
        log.info("  t3=%.4f (sr=%.4f ho=%.4f)", t3, sr, ho)
        log.info("  t4=%.4f (daughters=%d heredity=%.4f clip=%.4f dino=%.4f exploit=%s)",
                 t4, nd, her, her_clip, her_dino, mass_exploit)
        log.info("  vision=%.4f (raw=%.4f null=%.4f)", vis, vis_raw, vis_null)
        log.info("  locomotion=%.4f", loco)
        log.info("  composite=%.4f", composite)
        log.info("  total_time=%.1fs", total_elapsed)

        results = {
            "composite": composite, "tier1_pass": tier1_pass, "tier3": t3,
            "self_repair": sr, "homeostasis": ho, "tier4": t4, "n_daughters": nd,
            "heredity": her, "heredity_clip": her_clip, "heredity_dino": her_dino,
            "vision": vis, "vision_raw": vis_raw, "vision_null": vis_null,
            "locomotion": loco, "behavioral_fingerprint": fingerprint,
            "timings": _stage_timings,
        }

        output_dir = Path(solution_dir) / "eval_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        for artifact in ["eval_contact_sheet.png", "lifecycle.gif", "comparison_repair.gif"]:
            src = os.path.join(tmpdir, artifact)
            if os.path.exists(src):
                shutil.copy2(src, str(output_dir / artifact))

        return results


# ===================================================================
# Section 22: CLI Entry Point
# ===================================================================

def main() -> None:
    """CLI entry point for the evaluator."""
    parser = argparse.ArgumentParser(description="Flow-Life Evaluator eval-v1 (frozen synthesis)")
    parser.add_argument("--solution", required=True, help="Path to solution directory")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--modal", action="store_true", help="Dispatch to Modal T4 GPU")
    parser.add_argument("--output-json", type=str, default=None, help="Write detailed results JSON")
    args = parser.parse_args()

    sol = os.path.abspath(args.solution)
    if not os.path.isdir(sol):
        print(f"ERROR: {sol} not found", file=sys.stderr)
        sys.exit(1)

    np.random.seed(args.seed)
    torch = _import_torch()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.modal or os.environ.get("MODAL_DISPATCH") == "1":
        try:
            from research.eval.modal_app import app, GPU_TYPE, TIMEOUT_SECS
            import modal
            @app.function(gpu=GPU_TYPE, timeout=TIMEOUT_SECS,
                          mounts=[modal.Mount.from_local_dir(sol, remote_path="/solution")])
            def _remote(seed: int) -> dict:
                return evaluate_seed("/solution", seed)
            with app.run():
                results = _remote.remote(args.seed)
        except ImportError:
            log.warning("Modal unavailable, running locally")
            results = evaluate_seed(sol, args.seed)
    else:
        results = evaluate_seed(sol, args.seed)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info("Detailed results written to %s", args.output_json)

    print(f"METRIC={results['composite']:.6f}")
    print(f"TIER3_SCORE={results['tier3']:.6f}")
    print(f"TIER4_SCORE={results['tier4']:.6f}")
    print(f"VISION_SCORE={results['vision']:.6f}")
    print(f"LOCOMOTION_SCORE={results['locomotion']:.6f}")


if __name__ == "__main__":
    main()
