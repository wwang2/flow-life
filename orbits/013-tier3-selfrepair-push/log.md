---
issue: 14
parents: [011-deterministic-ic]
eval_version: eval-v1
metric: 0.5373
status: dead-end
---

# Research Notes

## Hypothesis

Tighten `soft_drift_threshold` (0.15 → 0.08) and strengthen `morph_coupling` (0.08 → 0.15) to boost tier 3 self-repair from ~0.60 to ~0.80 while keeping 011's fission. Expected composite 0.56 → 0.58.

## Results — REGRESSION

| Seed | composite | t3 (sr, ho) | t4 | daughters | heredity | vs 011 |
|------|-----------|-------------|-----|-----------|----------|--------|
| 1 | 0.5678 | 0.760 (0.611, 0.982) | 0.657 | 5 | 0.314 | -0.002 |
| 2 | 0.4997 | 0.712 (0.531, 0.983) | 0.519 | 2 | **0.371** | **-0.044** |
| 3 | 0.5444 | 0.761 (0.613, 0.982) | 0.646 | 3 | 0.291 | -0.017 |

**Mean: 0.5373** (vs 011: 0.5584, **-0.021**)

## Analysis

- Tier 3 self-repair did NOT improve — sr stayed at 0.53-0.61 (same range as 011). Homeostasis was already maxed (ho=0.98) in 011, so the homeostasis lever had no headroom.
- Heredity DID improve notably on seed 2 (0.371 vs 0.239 in 011) — stronger morph_coupling helps visual inheritance.
- But seed 2's replication collapsed (daughters 3 → 2, rep 1.0 → 0.67) because tighter mass control disturbed the fission → settling dynamics.
- Net: heredity gain < replication loss.

## Cross-Orbit Pattern (012, 013)

Two consecutive "push a knob" orbits both regressed:
- 012: stronger daughter print → +heredity on s1/s2, -replication on s3
- 013: stronger morph coupling + tighter drift → +heredity on s2, -replication on s2

**Orbit 011 sits in a narrow Pareto frontier.** Any lever pushed in either direction trades one component for another, with the net effect being negative. Further substrate tuning at this level is unlikely to yield >0.02 composite gain.

## Strategic Implication

The biggest remaining lever is **vision scoring (weight=0.23, currently =0)** because `ANTHROPIC_API_KEY` is not set. Enabling VLM scoring could add ~0.10 to composite without any substrate changes. Second-biggest: systematic multi-knob search (e.g., evolutionary/CMA-ES over params), not single-knob pushes.

Orbit 011 remains the winner at 0.5584.
