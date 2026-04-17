---
issue: 13
parents: [011-deterministic-ic]
eval_version: eval-v1
metric: 0.5345
status: dead-end
---

# Research Notes

## Hypothesis

Apply `spot_print_strength=0.10` only during daughter phase (post-fission), keeping parent phase at 0.04 to preserve 011's proven fission dynamics. Expected: heredity 0.27 → 0.40, composite 0.56 → 0.60.

## Implementation

Added `_daughter_phase` flag to substrate. Checked every 50 steps via `scipy.ndimage.label` on `(field > 0.05)`: if ≥2 connected components, switch to daughter strength (0.10); otherwise parent strength (0.04). Reset on context switch (mass drop >50%).

## Results — REGRESSION

| Seed | composite | daughters | t4 | heredity | vs 011 |
|------|-----------|-----------|-----|----------|--------|
| 1 | 0.5759 | 4 | 0.6501 | 0.3002 | +0.006 |
| 2 | 0.5469 | 3 | 0.6211 | 0.2422 | +0.003 |
| 3 | 0.4808 | 2 | 0.4650 | 0.2634 | **-0.081** |

**Mean: 0.5345** — regression from 011's 0.5584 (-0.024)

## Analysis

- Heredity barely moved (0.27 → 0.24-0.30). The 2.5× stronger daughter print did NOT materially boost CLIP/DINOv2 similarity.
- Seed 3 replication dropped from 1.0 (3/3 daughters survived autonomy) to 0.67 (2/3). The stronger texture disturbed one daughter enough to fail the 5-trial autonomy test.

## Key Insight

**CLIP and DINOv2 respond to overall shape, not print texture strength.** The parent is a smooth Gaussian blob (~0.5 cosine similarity baseline). Adding Turing-spot texture to daughters makes them MORE dissimilar to the smooth parent, not less. The small heredity gain on seed 1 came from daughter count increasing (4 vs 7 in 011), not print strength.

For heredity improvement: need to either (a) make the parent itself textured (so daughters inherit matching texture from the start) or (b) preserve parent shape precisely via structural templating. Amplifying print strength alone is not the answer.

This orbit is concluded as a dead-end. Orbit 011 remains the winner at 0.5584.
