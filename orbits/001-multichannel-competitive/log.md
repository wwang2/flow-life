---
issue: 2
parents: []
eval_version: eval-v1
metric: null
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

(Results appended after each eval iteration)
