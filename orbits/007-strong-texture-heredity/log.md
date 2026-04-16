---
issue: 8
parents: [006-heredity-morphogen]
eval_version: eval-v1
metric: 0.2766
---

# Research Notes

## Final Evaluation Results (3 seeds)

| Seed | composite | tier3  | tier4  | daughters | heredity | loco  |
|------|-----------|--------|--------|-----------|----------|-------|
| 1    | 0.2807    | 0.6693 | 0.0000 | 0         | 0.000    | 1.000 |
| 2    | 0.2691    | 0.6745 | 0.0000 | 0         | 0.000    | 0.869 |
| 3    | 0.2798    | 0.6660 | 0.0000 | 0         | 0.000    | 1.000 |

**Mean composite: 0.2766** — regression from orbit 006 (0.4349)

## Root Cause

`spot_print_strength=0.25` disrupts fission dynamics in the full evaluator run (10,000 steps).
The run.py internal check (shorter simulation) finds daughters=2-3, but the evaluator confirms
daughters=0 across all 3 seeds.

The stronger Turing coupling competes with the two-kernel fission instability — once the
spot pattern becomes dominant, the large-scale density gradient that drives fission is suppressed.

## Key Finding

Cannot improve heredity by printing Turing texture stronger — any texture strength that creates
visually distinctive daughters also prevents fission from occurring.

**The correct insight for orbit 008:** The parent IC is always the discovered Gaussian blob.
For heredity, need parent and daughters to look similar at 64×64 render.
Solution: BOOTSTRAP the IC from a daughter snapshot — simulate until fission, capture a daughter,
use it as the IC. Then parent (daughter-shaped) ≈ daughters (also daughter-shaped) → high CLIP sim.
