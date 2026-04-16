---
issue: 3
parents: []
eval_version: eval-v1
metric: null
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
