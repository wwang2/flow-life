# Multi-Channel Competitive Flow-Lenia

Two-kernel Flow-Lenia substrate with Turing instability for fission and self-repair.

## Architecture

- **Inner kernel** (R~10): Short-range shell-shaped kernel. When local density exceeds mu_inner, growth becomes negative (repulsive), pushing mass outward.
- **Outer kernel** (R~25): Long-range shell-shaped kernel. When neighborhood density is near mu_outer, growth is positive (attractive), pulling mass together.
- **Flow**: Gradient of smoothed potential field with semi-Lagrangian advection for mass conservation.

## Fission Mechanism

When the organism grows beyond a critical mass, inner repulsion overcomes outer attraction at the center. This Turing-instability-like mechanism causes the organism to pinch and divide into daughter organisms.

## Search Strategy

1. Random search over 80 parameter sets x 3 initial conditions
2. (1+1)-ES refinement of top-3 candidates
3. Fitness = weighted combination of stability, self-repair, and fission potential
