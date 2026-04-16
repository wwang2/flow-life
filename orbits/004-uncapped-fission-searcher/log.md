---
issue: 5
parents: [001-multichannel-competitive]
eval_version: eval-v1
metric: null
status: dead-end
---

# Research Notes

## Hypothesis

Use good.py's uncapped two-kernel Flow-Lenia as a substrate, search over initial conditions with
CMA-ES to find patterns that produce 2+ daughter organisms post-fission. No mass cap → Turing
instability drives fission.

## Outcome: Dead-End

Three sequential evaluator failures revealed a fundamental incompatibility between good.py dynamics
and the evaluator's Tier 1 concentration check.

### Failure 1: 3× mass-creation kill
Raw Gaussian IC blob triples in 6 steps. Evaluator kills at step 6 with
`MASS CREATION KILL: mass=8177 > 7298 (3x initial)`.

**Fix attempted:** Run 800 warmup steps to settle pattern to equilibrium, save that as IC.

### Failure 2: 30% capacity rejection
800-step settled state has mass=26881 > 19660 (30% of 256×256 grid).
Evaluator rejects pattern before eval even starts.

**Fix attempted:** Add 20% capacity mass ceiling to substrate (rescale if over threshold).

### Failure 3: Tier 1 concentration check failure
With mass ceiling, pattern mass=13107 (20% capacity), but:
`TIER1 FAIL: mass concentration=0.3933 < 0.6000`

Root cause: good.py growth dynamics flood the grid uniformly. Any ceiling-based rescaling
preserves the uniform distribution — mass stays spread across ~29000 pixels regardless of
normalization factor. Concentration = (top 10% of cells by value) / total ≈ 0.30-0.41 at
ANY settle step (verified 10-500 steps). The Tier 1 check can never pass.

### Conclusion

The good.py substrate produces uniform-density blobs that inherently fail the Tier 1 concentration
check (`top_10pct_mass / total_mass ≥ 0.60`). This is a fundamental dynamics mismatch, not a
tunable parameter. No further iteration warranted.

The key insight passed to orbit 005: use RD morphogen coupling (orbit 003) + soft drift correction
instead of strict mass conservation to enable fission while maintaining concentrated patterns.
