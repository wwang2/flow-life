# RD Morphogen Flow-Lenia

Two-kernel Flow-Lenia augmented with a Turing activator-inhibitor reaction-diffusion
morphogen layer for morphogenetic self-repair.

## Design

1. **Two-kernel base**: Inner kernel (R=10, repulsive) + outer kernel (R=25, attractive)
   creates Turing-pattern instability driving fission.
2. **RD morphogen**: Gierer-Meinhardt activator-inhibitor system running alongside the
   main state. Provides a "target pattern" that guides growth.
3. **Growth coupling**: Activator concentration modulates the growth function via sigmoid
   gating — high activator promotes growth, low activator suppresses it.
4. **Flow gating**: Inhibitor concentration dampens flow magnitude, creating barriers
   that prevent mass dissipation.
5. **Self-repair mechanism**: After damage, the RD system re-establishes the target
   morphogen gradient, driving regrowth in damaged areas.
