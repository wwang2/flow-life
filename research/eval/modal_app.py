"""
Shared Modal app definition for the flow-life campaign.

Single source of truth for:
  - Modal image (pip_install-based, no uv on remote)
  - Hardware config (T4 GPU, 4 CPUs, 16 GB RAM)
  - App name

Both evaluator.py and orbit solution run.py import from here.
"""

import modal

# ── Hardware config (from research/config.yaml compute.*) ─────────────────────
GPU_TYPE     = "T4"    # T4 for 256×256 Flow-Lenia + CLIP + VLM scoring
CPU_COUNT    = 4
MEMORY_MB    = 16384
TIMEOUT_SECS = 1200    # 20 min per seed — matches eval.timeout in config.yaml

# ── Image ──────────────────────────────────────────────────────────────────────
# Build in layers so only changed layers rebuild.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # System deps for imageio / PIL / scipy
    .run_commands("apt-get update -qq && apt-get install -y git libglib2.0-0 --no-install-recommends -qq")
    # PyTorch with CUDA 12.1 (supported on T4)
    .pip_install(
        "torch==2.3.0",
        "torchvision==0.18.0",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    # Scientific stack
    .pip_install(
        "numpy>=1.26",
        "scipy>=1.13",
        "scikit-image>=0.22",
        "imageio>=2.34",
        "imageio[ffmpeg]",
        "matplotlib>=3.8",
        "Pillow>=10.0",
        "pyyaml>=6",
    )
    # CLIP (openai) for heredity scoring
    .run_commands("pip install git+https://github.com/openai/CLIP.git --quiet")
    # Anthropic SDK for VLM vision scoring
    .pip_install("anthropic>=0.25")
    # ftfy / regex needed by CLIP
    .pip_install("ftfy", "regex")
)

# ── Modal app ──────────────────────────────────────────────────────────────────
app = modal.App("flow-life-eval", image=image)
