# RD Morphogen V2 -- Uncapped Growth

Builds on orbit 003's RD morphogen backbone (t3=0.904, homeostasis=1.0) 
with the key fix: removing strict mass conservation that killed daughters.

## Changes from orbit 003

1. **No strict mass rescaling** -- replaced with soft drift correction
2. **No hard clamp(0,1)** -- mass can accumulate for fission
3. **Fission-detecting searcher** -- replaces stability-only searcher
4. **Parameter grid search** -- over repulsion strength, coupling, drift tolerance

## Hypothesis

Orbit 003's strict mass conservation (rescale every step + clamp to [0,1]) 
eroded fragment mass below viability threshold. Without it, the inner 
repulsion kernel can pinch blobs into daughters that persist independently.
