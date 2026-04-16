---
issue: 3
parents: []
eval_version: eval-v1
metric: 0.252581
---

# Chemotaxis-Fission Flow-Lenia

## Hypothesis

The flow field is computed as the gradient of a self-secreted diffused concentration field (chemotaxis). The pattern secretes a chemoattractant that it follows via gradient ascent. When local density exceeds a threshold, the gradient reverses locally (repulsion), driving fission into two daughter patterns.

## Approach

The substrate combines two mechanisms:

1. **Chemotaxis flow**: A wide Gaussian kernel convolves the state to produce a chemoattractant field `c`. The flow is `grad(c)`, normalized and scaled. This creates self-trapping (the pattern follows its own concentration gradient) and locomotion (asymmetric gradients cause directed motion).

2. **Density-dependent fission**: When local density exceeds `rho_split`, the flow direction reverses locally. High-density centers experience outward flow (repulsion), while the periphery experiences inward flow (attraction). This pinches the pattern and drives binary fission.

3. **Growth**: Two-kernel Lenia growth (inner repulsive, outer attractive) provides the Turing-instability backbone that the chemotaxis flow amplifies.

## Key Design Decisions

- Combined the chemotaxis flow mechanism with the proven two-kernel Turing instability from the reference good.py
- The chemotaxis gradient provides locomotion (weight 0.10 in metric) essentially for free
- Fission is driven by BOTH the density-dependent flow reversal AND the inner-kernel repulsion
- Semi-Lagrangian advection with toroidal wrapping for mass conservation

## Seed 1 Results (CPU, eval-v1)

| Metric | Value |
|--------|-------|
| Tier 1 | PASS (concentration=0.771, survival=2.800) |
| Tier 3 | 0.5651 (sr=0.2930, ho=0.9733) |
| Locomotion | **1.0000** (displacement=90.91px — exceptional!) |
| Tier 4 | 0.0000 (daughters=0, no replication detected) |
| Vision | 0.0000 (VLM auth missing at run time) |
| **METRIC** | **0.2526** |

**Analysis:** The chemotaxis flow achieves near-maximum locomotion (90.91px CoM displacement) — the pattern travels continuously. However SSIM self-repair is low (0.29), indicating the diffuse multi-blob topology doesn't recover spatially after 20% damage. Homeostasis is strong (CV=0.027). No daughters in 10k Tier 4 steps — fission threshold too high or pattern too fragmented.

**Next steps for a child orbit:** Tighten the spatial coherence (reduce diffusion radius, increase concentration threshold) to get a compact moving blob; then tune fission split ratio. The locomotion mechanism is proven — exploit it.
