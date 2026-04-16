---
issue: 6
parents: [003-rd-morphogen]
eval_version: eval-v1
metric: null
---

# Research Notes

## Hypothesis

Orbit 003 (RD morphogen substrate) achieved strong self-repair via Gierer-Meinhardt morphogen
coupling but scored t4=0 because strict mass conservation eroded fission fragments below viability.
This orbit removes the strict mass rescaling, replacing it with soft drift correction (only normalize
if mass drifts >15% from initial, with a hard ceiling at 2.8x to stay below the evaluator's 3x kill).

The key idea: the RD morphogen provides self-repair memory (tier 3), while uncapped growth allows the
inner repulsion kernel to drive genuine fission (tier 4). We combine the best of both worlds.

## Substrate: RDMorphogenSubstrateV2

Same two-kernel Flow-Lenia backbone as good.py (R_inner=10, R_outer=25, dt=0.2), plus:
- Gierer-Meinhardt reaction-diffusion morphogen (activator-inhibitor) for self-repair memory
- Morphogen coupling: growth is modulated by the activator field, creating pattern memory
- Persistent drift field for locomotion
- Soft drift correction: mass can grow beyond initial (enabling fission) but is gently nudged
  back if it drifts >15%, with hard ceiling at 2.8x (safely below 3x kill)

## Searcher: Grid search over fission parameters

Search over 3 key axes:
- w_inner: [-0.6, -0.8, -1.0, -1.2] (repulsion strength -- stronger drives pinching)
- morph_coupling: [0.05, 0.08, 0.12] (morphogen influence on growth)
- soft_drift_threshold: [0.10, 0.15, 0.25] (mass drift tolerance)
- Blob shapes: 5 presets from elongated to circular

Two-phase fitness: Phase 1 survival filter (500 steps), Phase 2 fission detection
(2000-6000 steps) matching evaluator's Tier 4 logic (connected components, persistence tracking).

## Iteration 1: Initial implementation

Local test on 64x64 grid (2 candidates): fission detected (3-5 daughters), pipeline functional.
Running full eval next.
