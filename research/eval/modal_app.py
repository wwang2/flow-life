"""
Shared Modal app definition for the flow-life campaign.

Single source of truth for:
  - Modal image (built from pyproject.toml via uv)
  - Hardware config (T4 GPU, 4 CPUs, 16 GB RAM)
  - App name

Both evaluator.py and orbit solution run.py import from here.
To add a package: `uv add <package>` at the project root.
"""

import modal
from pathlib import Path

# ── Hardware config (from research/config.yaml compute.*) ─────────────────────
GPU_TYPE     = "T4"    # T4 for 256×256 Flow-Lenia + CLIP + VLM scoring
CPU_COUNT    = 4
MEMORY_MB    = 16384
TIMEOUT_SECS = 1200    # 20 min per seed — matches eval.timeout in config.yaml

# ── Image built from pyproject.toml via uv ────────────────────────────────────
_repo_root = Path(__file__).parent.parent.parent  # campaign repo root

image = (
    modal.Image.debian_slim(python_version="3.11")
    .run_commands("pip install uv --quiet")
    .add_local_file(str(_repo_root / "pyproject.toml"), "/app/pyproject.toml", copy=True)
    .run_commands("cd /app && uv pip install --system .")
    # Install PyTorch with CUDA for GPU-accelerated FFT convolutions
    .run_commands(
        "pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet"
    )
    # Install clip for heredity scoring
    .run_commands("pip install git+https://github.com/openai/CLIP.git --quiet")
    # Install scikit-image for SSIM, scipy for connected-component labeling
    .run_commands("pip install scikit-image scipy imageio --quiet")
    # Install anthropic SDK for VLM vision scoring
    .run_commands("pip install anthropic --quiet")
)

# ── Modal app ──────────────────────────────────────────────────────────────────
app = modal.App("flow-life-eval", image=image)
