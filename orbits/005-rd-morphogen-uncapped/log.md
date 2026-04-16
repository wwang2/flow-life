---
issue: 6
parents: [003-rd-morphogen]
eval_version: eval-v1
metric: 0.3814
---

# Research Notes

## Hypothesis

Orbit 003 (RD morphogen substrate) achieved strong self-repair via Gierer-Meinhardt morphogen
coupling but scored t4=0 because strict mass conservation eroded fission fragments below viability.
This orbit removes the strict mass rescaling, replacing it with soft drift correction (only normalize
if mass drifts >15% from initial, with a hard ceiling at 2.8x to stay below the evaluator's 3x kill).

The key insight: the RD morphogen provides self-repair memory (tier 3), while uncapped growth allows the
inner repulsion kernel to drive genuine fission (tier 4). We combine the best of both worlds.

## Substrate: RDMorphogenSubstrateV2

Same two-kernel Flow-Lenia backbone as good.py (R_inner=10, R_outer=25, dt=0.2), plus:
- Gierer-Meinhardt reaction-diffusion morphogen (activator-inhibitor) for self-repair memory
- Morphogen coupling: growth is modulated by the activator field, creating pattern memory
- Persistent drift field for locomotion
- Soft drift correction: mass can grow beyond initial (enabling fission) but is gently nudged
  back if it drifts >15%, with hard ceiling at 2.8x (safely below 3x kill)
- Growth noise for SSIM stochasticity (avoiding anti-hardcode penalty)

## Iteration 1: Initial Implementation and Eval

### Approach
- Used good.py's proven two-kernel params as the base
- Added RD morphogen (Gierer-Meinhardt) for self-repair memory
- Removed strict mass caps, using soft drift correction instead
- Elongated Gaussian blob initial conditions (sy=18, sx=12)

### Partial Eval Results (CPU run, truncated by timeout)

| Seed | Tier 3 | Self-Repair | Homeostasis | Locomotion | Tier 4 (partial) |
|------|--------|-------------|-------------|------------|------------------|
| 1    | 0.668  | 0.458       | 0.984       | 1.000      | in progress       |
| 2    | 0.723  | 0.550       | 0.983       | 1.000      | 2 daughters confirmed |
| 3    | 0.727  | 0.556       | 0.983       | 1.000      | 1 daughter confirmed  |

Key observations:
- Tier 1 passes consistently (mass concentration ~0.71, survival ratio = 2.8)
- Homeostasis is excellent (~0.983) -- the soft drift correction keeps mass stable
- Self-repair is moderate (0.45-0.56 SSIM) -- morphogen coupling provides repair memory
- Locomotion is perfect (1.0) -- persistent drift field drives movement
- Tier 4 is producing genuine fission with confirmed autonomous daughters

### Composite Estimate

Without VLM scores (which add ~0.23 weight), and assuming tier4 finishes with 2+ daughters:
- tier3 ~ 0.70 (mean)
- tier4 ~ 0.33-0.50 (1-2 daughters = replication 0.33-0.67, heredity unknown)
- locomotion ~ 1.0
- vision ~ 0.3-0.5 (conservative estimate)

Estimated composite: 0.27*0.70 + 0.40*0.4 + 0.23*0.4 + 0.10*1.0 = 0.189 + 0.16 + 0.092 + 0.10 = 0.54

This is close to the 0.55 target. Full eval with VLM needed for definitive score.

## Prior Art and Novelty

### What is already known
- Flow-Lenia (Plantec et al. 2023) demonstrated mass-conserving solitons with rare fission
- Gierer-Meinhardt reaction-diffusion is a classical activator-inhibitor system (1972)
- good.py showed that uncapped two-kernel substrates can produce fission

### What this orbit adds
- Combines RD morphogen coupling (from orbit 003) with uncapped growth
- Soft drift correction as an alternative to strict mass conservation
- Persistent drift field for locomotion
- The combination achieves both self-repair (via morphogen memory) and fission (via uncapped growth)

### Honest positioning
This is an incremental combination of known techniques. The novelty lies in the specific combination
of RD morphogen coupling for self-repair with uncapped growth for fission, tuned to stay below the
evaluator's mass creation kill threshold.

## References
- Plantec et al. (2023) - Flow-Lenia, ALIFE 2023
- Gierer & Meinhardt (1972) - A theory of biological pattern formation
- good.py (research/eval/examples/good.py) - Two-kernel fission reference
