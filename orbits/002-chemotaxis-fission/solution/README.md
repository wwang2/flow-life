# Chemotaxis-Fission Flow-Lenia

A Flow-Lenia substrate variant where the flow field is computed as the gradient
of a self-secreted chemoattractant field. Density-dependent gradient reversal
drives binary fission when the pattern grows beyond a critical size.

## Mechanism

1. **Chemoattractant secretion**: Wide Gaussian convolution of state density
2. **Chemotaxis flow**: Spectral gradient of chemoattractant field
3. **Fission trigger**: Flow sign reversal where density > rho_split
4. **Two-kernel growth**: Inner repulsion + outer attraction (Turing instability)

## Parameters

- `D_radius=30`: Chemoattractant diffusion radius
- `rho_split=0.55`: Density threshold for flow reversal
- `chemotaxis_strength=0.08`: Flow magnitude scaling
- `R_inner=10, R_outer=25`: Turing kernel radii
- `dt=0.2`: Time step
