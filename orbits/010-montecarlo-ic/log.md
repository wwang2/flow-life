---
issue: 11
parents: [009-seed1-fission-fix]
eval_version: eval-v1
metric: 0.5103
---

# Research Notes

## Orbit Goal

Improve consistency across seeds by using Monte Carlo IC verification: run each blob IC 3× with different torch seeds, accept only if daughters≥2 in 2/3 runs. This should prevent selecting ICs that produce daughters only due to lucky random drift.

## Approach

- 9 blob shapes (3 from orbit 009 + 6 new)
- `verify_daughters_mc`: 3 MC runs per IC with `torch.manual_seed(mc_run * 42 + 1234)`
- Accept IC if passes≥2/3; fallback to best-by-daughters if none verified
- Early stop for blob1 (sy=18, sx=12) if it passes MC verification (preserves locomotion)
- No strict per-daughter autonomy pre-check (orbit 009 approach dropped)
- Time limit: 750s per run.py execution

## Results

| Seed | composite | daughters | t4     | heredity | vs orbit 009 |
|------|-----------|-----------|--------|----------|--------------|
| 1    | 0.5701    | 7         | 0.6498 | 0.2997   | =0.5701 (same blob2) |
| 2    | 0.5437    | 3         | 0.6194 | 0.2388   | =0.5437 (blob1 passes 3/3 MC) |
| 3    | 0.4171    | 1         | 0.3023 | 0.2712   | =0.4171 (no blob passes MC) |

**Mean composite: 0.5103** (identical to orbit 009)

## Key Findings

- Seed 1: blob2 (sy=20, sx=10) passes MC 3/3 → consistent daughters=7 in evaluator ✓
- Seed 2: blob1 (sy=18, sx=12) passes MC 3/3 → early stop → consistent daughters=3 ✓
- Seed 3: **no blob passes MC threshold** — best blob1 gets 0/3 (daughters=0,1,0); all others also fail
  - MC correctly identifies that seed 3 dynamics are unreliable for fission
  - Fallback to best-by-daughters selects blob1 (best_daughters=1), evaluator gets daughters=1
- The seed 3 problem is not IC selection — it's that the substrate itself doesn't fission reliably for seed 3

## Root Cause Analysis for Seed 3

The stochastic `torch.randn` drift field (in `substrate.py`) produces fundamentally different fission trajectories depending on the global random state at run start. Orbit 006 seed 3 was lucky (daughters=3). Both orbit 009 and 010 consistently get daughters≤1 for seed 3.

MC verification confirms: none of the 9 tested blob shapes reliably fission for seed 3 at 2/3 threshold. Increasing blob variety or shape search won't solve this — the fix needs to be in the substrate dynamics or in parameterizing the stochastic drift field per-seed.
