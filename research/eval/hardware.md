# Hardware Inference

## Problem signals
> "Run discovery for up to 20,000 timesteps on a 256×256 grid" — Flow-Lenia simulation uses
> FFT-based convolutions; 256×256 × 20k steps is GPU-amenable but not large-scale.
> "Print TIER3_SCORE and TIER4_SCORE" — requires CLIP embedding calls (heredity metric) and
> SSIM computation; both are lightweight on GPU.

## Inferred needs
- Evaluation: T4 — FFT convolutions + CLIP embeddings at 256×256/20k steps fit well in T4
  VRAM; CPU alone would be too slow for 20-min timeout across 3 seeds
- Experiments: T4 — orbit agents run discovery in the same scale as eval; GPU speeds
  CMA-ES/gradient search over substrates significantly
- Estimated eval duration: 5–12 minutes per seed on T4 (×3 seeds = 15–36 min per solution)

## Config
compute.gpu: T4
