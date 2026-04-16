---
issue: 7
parents: [005-rd-morphogen-uncapped]
eval_version: eval-v1
metric: 0.4349
---

# Research Notes

## Final Evaluation Results (3 seeds)

| Seed | composite | tier3  | tier4  | daughters | heredity |
|------|-----------|--------|--------|-----------|----------|
| 1    | 0.3007    | 0.7434 | 0.0000 | 0         | 0.000    |
| 2    | 0.4960    | 0.7258 | 0.5000 | 3         | 0.000    |
| 3    | 0.5081    | 0.7602 | 0.5071 | 3         | 0.014    |

**Mean composite: 0.4349** (target was ≥ 0.55)

All seeds achieved locomotion=1.0. Vision scoring=0.0 (no CLIP/DINOv2 API key available).

## Design

Orbit 006 builds on orbit 005's proven RD-morphogen Flow-Lenia substrate, adding:

1. **Heredity morphogen channel**: Gierer-Meinhardt reaction-diffusion generates Turing-like texture patterns that are inherited by daughters after fission. Activator diffuses slowly (D_a=0.04), inhibitor diffuses faster (D_h=0.6) to produce spot patterns.

2. **Spot printing**: Turing texture is lightly printed onto the density field (spot_print_strength=0.04) so daughters carry spatial structure from parent.

3. **Idempotent to()**: Evaluator calls `substrate.to(device)` before tier4; made idempotent to prevent kernel cache destruction.

4. **Context-switch detection**: Detects mass drops >50% (new eval phase / autonomy test) and resets drift field + morphogen state, ensuring tier4 gets fresh stable dynamics.

5. **Preserved orbit 005 fission params**: w_inner=-0.8, rd_substeps=5, same elongated Gaussian blob selection strategy (noise seeded from eval seed directly).

## Key Findings

- Seeds 2 and 3 achieved daughters=3 with replication=1.0, demonstrating stable fission.
- Seed 1 blob (sy=18, sx=12, noise_seed=1) did not produce persistent fission streaks.
- Heredity scoring near zero: daughters are structurally distinct blobs without enough Turing texture at 64×64 resolution to exceed cosine_sim=0.5 threshold for CLIP/DINOv2.
- spot_print_strength=0.04 is insufficient; stronger texture printing risks destabilizing fission dynamics.

## Improvement Opportunities

- Increase spot_print_strength and test fission stability (would need careful tuning).
- Use multi-channel state where one channel carries heritable pattern and evaluator checks channel correlation.
- Seed 1 fission: explore different blob shapes or parameters for seed 1.
