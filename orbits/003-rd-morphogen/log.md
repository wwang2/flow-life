---
issue: 4
parents: []
eval_version: eval-v1
metric: 0.325592
---

# RD Morphogen + Flow-Lenia Substrate

## Hypothesis

Couple a Gierer-Meinhardt reaction-diffusion (activator-inhibitor) morphogen layer to Flow-Lenia. The activator modulates growth rate, creating spatially structured activity patterns. The two-kernel Turing instability provides locomotion and structural stability, while the RD system provides an additional self-organizing "blueprint" that aids self-repair (the activator pattern re-establishes after damage).

## Approach

- **RD layer**: Gierer-Meinhardt activator (fast diffusion A) + inhibitor (slow diffusion H) — classic Turing pattern generator
- **Coupled growth**: `growth_rate = growth_fn(U) * (1 + alpha * A)` where A is the activator field
- **Two-kernel Flow-Lenia backbone**: inner repulsive (R~10) + outer attractive (R~25) kernels for Turing fission

## Seed 1 Results (CPU, eval-v1)

| Metric | Value |
|--------|-------|
| Tier 1 | PASS (concentration=0.723, survival=1.000) |
| Tier 3 | **0.9041** (sr=0.8402, ho=**1.0000**) |
| Locomotion | **0.8148** (displacement=40.74px) |
| Tier 4 | 0.0000 (daughters=0, no replication detected) |
| Vision | 0.0000 (VLM auth missing) |
| **METRIC** | **0.3256** — best of orbit batch 1 |

**Analysis:** Excellent self-repair (SSIM 0.840) and perfect homeostasis (CV≈0, mass perfectly regulated). Good locomotion (40.74px). The RD morphogen coupling creates strong structural memory — the activator pattern guides recovery after damage. However, t4=0: the same stability that enables repair prevents fission. Discovery score=1.000 was achieved by pattern `random_12` suggesting the RD coupling is robustly stable regardless of IC.

**Key insight for child orbits:** Homeostasis=1.0 + strong self-repair is a proven win. The missing piece is fission. A child should decouple the stability/repair mechanism from the fission trigger — e.g., use a second RD mode that activates fission when total mass exceeds threshold.
