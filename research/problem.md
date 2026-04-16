# Can novel Flow-Lenia substrate variants produce self-repairing, self-replicating artificial life?

## Problem Statement

Automated Search for Artificial Life (ASAL, 2024) demonstrated that foundation model embeddings
of static frames can guide discovery of visually interesting patterns across five substrates
(Boids, Particle Life, Game of Life, Lenia, NCA). However, ASAL's metrics are fundamentally
frame-level: they capture appearance novelty but cannot detect whether a pattern repairs itself
after damage, maintains homeostasis under perturbation, or produces daughter copies that inherit
parental structure. The highest behavioral tiers — Tier 3 (robustness) and Tier 4 (reproduction)
— remain undiscovered in the continuous CA literature via automated search.

Flow-Lenia (Best Paper, ALIFE 2023) introduced two key substrate innovations over classical Lenia:
mass conservation (patterns cannot appear from nothing) and parameter localization (each spatial
region carries its own update-rule parameters). These properties make living patterns the default
rather than the exception, and enable multi-species coexistence. Yet Flow-Lenia's own search has
relied on manual parameter sweeps and gradient-based optimization targeting specific known
behaviors — it has never been systematically searched for Tier 3-4 behaviors across a broad
substrate design space.

This campaign explores **substrate design space**: orbit agents propose new Flow-Lenia variant
update rules (new kernel families, interaction terms, energy channels, asymmetry mechanisms, or
multi-channel extensions) alongside a discovery pipeline that searches for the highest-tier
patterns achievable in that substrate. Each solution delivers the substrate definition, the
search algorithm, the discovered patterns, and GIF visualizations. The eval harness
automatically scores discovered patterns on a composite Tier 3+4 life-likeness metric using
behavioral tests: damage-recovery fidelity for self-repair and daughter-pattern detection for
self-replication.

## Solution Interface

Each solution is a **full pipeline** implemented as a Python directory with:

```
solution/
  substrate.py        # Flow-Lenia variant: implements update_step(state, params) -> state
                      # Must inherit BaseFlowLenia and implement compute_flow() and apply_growth()
  searcher.py         # Discovery algorithm: implements search(substrate, seed, n_steps) -> patterns
  run.py              # Entry point: produces discovered_patterns.npz + gifs/ directory
  README.md           # Brief description of substrate design choices
```

**substrate.py** must implement:
```python
class SubstrateVariant(BaseFlowLenia):
    def compute_flow(self, state: np.ndarray, params: dict) -> np.ndarray:
        """Return divergence-free flow field (mass-conserving)."""
        ...
    def apply_growth(self, state: np.ndarray, flow: np.ndarray, params: dict) -> np.ndarray:
        """Apply growth mapping with optional novel interaction terms."""
        ...
    def get_default_params(self) -> dict:
        """Return substrate-specific parameter dictionary."""
        ...
```

**run.py** must:
- Accept `--seed <int>` and `--grid-size 256` flags
- Run discovery for up to 20,000 timesteps on a 256×256 grid
- Save top-5 discovered patterns to `discovered_patterns.npz`
- Save GIFs for each discovered pattern (birth → stable dynamics) to `gifs/`
- Print `TIER3_SCORE=<float>` and `TIER4_SCORE=<float>` to stdout

## Success Metric

**Composite Tier 3+4 Life-Likeness Score** — maximize.

```
composite = 0.4 * tier3_score + 0.6 * tier4_score
```

**Tier 3 score** (robustness, 0–1):
- `self_repair`: After applying structured damage (zero out 20% of pattern mass), measure
  pattern reconstruction fidelity after 500 recovery steps — Structural Similarity Index (SSIM)
  between pre-damage and post-recovery state. Average over 3 damage trials.
- `homeostasis`: Coefficient of variation of total pattern mass over 5,000 steady-state steps.
  Score = 1 - min(CV, 1). Lower variance = higher score.
- `tier3_score = 0.6 * self_repair + 0.4 * homeostasis`

**Tier 4 score** (reproduction, 0–1):
- `replication_count`: Number of spatially distinct daughter patterns detected within 10,000
  steps of isolating a single stable parent. Score = min(count / 3, 1).
- `heredity`: Cosine similarity in CLIP embedding space between parent pattern and each
  daughter pattern, averaged over detected daughters. Score = max(0, similarity - 0.5) * 2.
- `tier4_score = 0.5 * replication_count + 0.5 * heredity`

**Evaluation procedure:** averaged over seeds 1, 2, 3. Timeout per seed: 20 minutes on T4 GPU (Modal).
Composite includes a vision model score: `composite = 0.30*tier3 + 0.45*tier4 + 0.25*vision_score`
where vision_score is from `claude-haiku-4-5-20251001` scoring a 9-frame contact sheet GIF.

**Baseline (standard Flow-Lenia, no substrate modification):** expected composite ≈ 0.15–0.20
(mass-stable solitons with decent homeostasis but no reliable self-repair or replication).

**Target:** composite ≥ 0.55 (reliable self-repair + at least one reproducible daughter pattern + VLM-confirmed life-likeness).

**Direction:** maximize.

## Known Approaches

- **Flow-Lenia** (Plantec et al., 2023): mass conservation + parameter localization. Basis for
  all substrate variants in this campaign. Demonstrated locomotion, chemotaxis, obstacle
  navigation, and rare self-division. No systematic Tier 3-4 search performed.
- **ASAL** (Kumar et al., 2024): foundation model (CLIP/DINOv2) embeddings + CMA-ES search.
  Tier 1-2 behaviors discovered reliably; Tier 3-4 not targeted.
- **LeniaBreeder** (2024): MAP-Elites + AURORA QD search in Lenia. Targets locomotion (Tier 2).
- **IMGEP-HOLMES** (Etcheverry et al., 2020): intrinsically motivated goal exploration in Lenia.
  Automated discovery of novel solitons (Tier 1-2).
- **Growing NCA** (Mordvintsev et al., 2020): gradient-trained NCAs demonstrating self-repair
  (Tier 3) in a fixed target-image paradigm — not open-ended discovery.
