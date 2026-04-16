---
issue: 2
parents: []
eval_version: eval-v1
metric: 0.230683
---

# Multi-Channel Flow-Lenia with Inter-Channel Competitive Coupling

## Hypothesis

Standard single-channel Flow-Lenia produces stable solitons (Tier 1-2) but lacks the internal redundancy needed for self-repair (Tier 3) or self-replication (Tier 4). The key insight: biological organisms use multiple interacting chemical/molecular channels as mutual "template memories." When one channel is damaged, the cross-channel coupling drives reconstruction from the intact channels.

## Approach

Building on the two-kernel fission substrate from `good.py`, which already demonstrates reliable fission via Turing instability (short-range repulsion + long-range attraction). The two-kernel approach is the strongest known baseline for Tier 4 (reproduction).

### Key design decisions

1. **Two-kernel architecture**: Inner kernel (short-range R~10, repulsive at high density) + outer kernel (long-range R~25, attractive at medium density) creates Turing instability for fission
2. **Semi-Lagrangian advection**: Toroidal bilinear interpolation for mass-conserving flow
3. **Growth function**: Gaussian bell curve mapping potential to growth rate
4. **CMA-ES optimization**: Search over kernel parameters, growth parameters, and flow strength to maximize a proxy fitness combining mass stability, self-repair, and fission tendency

## Research Notes

## Seed 1 Results (CPU, eval-v1)

| Metric | Value |
|--------|-------|
| Tier 1 | PASS (concentration=1.000, survival=1.068) |
| Tier 3 | **0.8527** (sr=0.9687, ho=0.6785) |
| Locomotion | 0.0047 (displacement=0.23px — nearly static) |
| Tier 4 | 0.0000 (daughters=0, no replication detected) |
| Vision | 0.0000 (VLM skipped — `anthropic` not installed at run time) |
| **METRIC** | **0.2307** |

**Analysis:** Excellent self-repair (SSIM 0.97 across 3 trials) confirms the two-kernel Turing substrate robustly restores structure after 20% damage. However, t4=0 — the 10,000-step simulation found no daughter blobs. The substrate is stable but not replicating. Homeostasis CV=0.32 is moderate; mass oscillates rather than being tightly regulated.

**Next steps for a child orbit:** Reduce pattern mass or increase flow strength to push toward fission regime; try asymmetric perturbation to trigger splitting.
