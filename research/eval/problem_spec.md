# Problem Specification: Flow-Life

## Research Question
Can we design novel Flow-Lenia substrate variants that produce patterns exhibiting Tier 3-4
life-like behaviors (self-repair + self-replication), as measured by a composite algorithmic +
vision-model score — discovering species qualitatively richer than those found by ASAL's
static-frame search?

## Metric

### Composite Life-Likeness Score (maximize)

```
composite = 0.30 * tier3_score + 0.45 * tier4_score + 0.25 * vision_score
```

### Tier 3 Score — Robustness (0–1)

**self_repair** (weight 0.6):
- Isolate a stable discovered pattern on a 256×256 toroidal grid
- Apply structured damage: zero out a contiguous rectangular region covering 20% of non-empty
  pixels (seeded deterministically per trial)
- Run 500 recovery steps
- Score = `skimage.metrics.structural_similarity(pre_damage, post_recovery,
    win_size=11, gaussian_weights=True, data_range=1.0)` computed on
  sum-over-channels field normalized to [0,1] on the full 256×256 grid
- Average over 3 damage trials with seeds 1, 2, 3

**homeostasis** (weight 0.4):
- Run 5,000 steady-state steps on isolated stable pattern
- Score = `1 - min(CV, 1)` where CV = std(total_mass) / mean(total_mass)
- Higher = more mass-stable

```python
tier3_score = 0.6 * self_repair + 0.4 * homeostasis
```

### Tier 4 Score — Reproduction (0–1)

**Setup**: Isolate a single stable parent on a 256×256 toroidal grid; run up to 10,000 steps.

**Daughter detection algorithm**:
1. Binarize: `mask = state.sum(axis=0) > 0.05 * state.sum(axis=0).max()`
2. Toroidal pad-wrap before labeling: pad by kernel_radius, apply `scipy.ndimage.label`
   with 8-connectivity, then clip label array back to [0:256, 0:256]
3. Filter: discard components with total mass < 0.10 × parent_initial_mass
4. Track parent: nearest centroid to t=0 parent centroid position (handles locomotion)
5. Confirm daughter: any non-parent component persisting for ≥ 3 consecutive detection frames
   (detection runs every 100 steps) and centroid > 20px from parent centroid

**replication_count** (weight 0.5):
- `score = min(confirmed_daughter_count / 3, 1.0)`

**heredity** (weight 0.5):
- Render each confirmed daughter as a 64×64 normalized image
- Compute CLIP ViT-B/32 embedding (openai/clip-vit-base-patch32) for parent and each daughter
- `score = max(0, mean_cosine_similarity(parent, daughters) - 0.5) * 2`
  (maps [0.5, 1.0] → [0.0, 1.0]; anything below 0.5 similarity scores 0)

```python
tier4_score = 0.5 * replication_count + 0.5 * heredity
```

### Vision Score — VLM Semantic Assessment (0–1)

**Render a contact sheet**: 3×3 grid of 9 frames sampled at equal intervals across the full
run (birth → stable dynamics → damage → recovery → replication window). Each frame is 256×256
rendered as a grayscale or RGB heatmap of the sum-over-channels state.

**VLM prompt** (sent to `claude-haiku-4-5-20251001` with the contact sheet image):

```
You are evaluating an artificial life simulation. This contact sheet shows 9 frames
of a pattern's lifecycle in a Flow-Lenia continuous cellular automaton.

Score the pattern on each criterion (0.0 to 1.0):
1. spatial_coherence: Does the pattern maintain a distinct, bounded, coherent structure?
2. locomotion: Does the pattern appear to move through space (not just oscillate in place)?
3. self_repair_evidence: Is there visual evidence of the pattern recovering from disruption?
4. replication_evidence: Do you see the pattern splitting or producing daughter copies?
5. complexity: How morphologically complex and organized is the pattern (vs. blob/noise)?

Respond ONLY with valid JSON:
{"spatial_coherence": 0.0, "locomotion": 0.0, "self_repair_evidence": 0.0,
 "replication_evidence": 0.0, "complexity": 0.0}
```

**vision_score** = mean of the 5 VLM scores.

**Implementation note**: Use `anthropic` Python SDK with `claude-haiku-4-5-20251001`.
Cache VLM calls by image hash to avoid redundant API calls across seeds.

## Solution Interface

Each solution is a directory:

```
solution/
  substrate.py    # Flow-Lenia variant inheriting BaseFlowLenia
  searcher.py     # Discovery algorithm
  run.py          # Entry point (see below)
  README.md       # Brief substrate design notes (1 page)
```

### BaseFlowLenia (provided by eval harness)

```python
class BaseFlowLenia(ABC):
    def __init__(self, grid_size: int = 256, n_channels: int = 1):
        self.H = self.W = grid_size
        self.n_channels = n_channels
        # Pre-computed FFT frequency grids (for subclasses to use)
        ky = torch.fft.fftfreq(grid_size).reshape(-1, 1)
        kx = torch.fft.fftfreq(grid_size).reshape(1, -1)
        self.k2 = (kx**2 + ky**2)  # shape (H, W)

    def update_step(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Non-overridable. Calls compute_flow → _advect → apply_growth."""
        flow = self.compute_flow(state, params)          # (2, H, W)
        advected = self._advect(state, flow)             # (C, H, W)
        new_state = self.apply_growth(advected, flow, params)  # (C, H, W)
        if abs(new_state.sum() - state.sum()) / (state.sum() + 1e-8) > 0.05:
            # Soft warning — do not hard-rescale (corrupts substrate physics)
            pass
        return new_state.clamp(0, 1)

    def _advect(self, state: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """Non-overridable. Mass-conserving semi-Lagrangian advection (toroidal)."""
        ...  # provided by harness

    @abstractmethod
    def compute_flow(self, state: torch.Tensor, params: dict) -> torch.Tensor:
        """Return divergence-free flow field. Shape: (2, H, W)."""
        ...

    @abstractmethod
    def apply_growth(self, state: torch.Tensor, flow: torch.Tensor,
                     params: dict) -> torch.Tensor:
        """Apply growth mapping. Shape in/out: (C, H, W)."""
        ...

    @abstractmethod
    def get_default_params(self) -> dict:
        """Return substrate-specific parameter dictionary."""
        ...
```

### run.py interface

```bash
uv run python solution/run.py --seed <int> --grid-size 256 --output-dir <path>
```

Must write to `--output-dir`:
- `discovered_patterns.npz`: top-5 patterns, each as float32 array shape `(C, 256, 256)`
- `gifs/pattern_{i}.gif`: lifecycle GIF for each discovered pattern
- `contact_sheet.png`: 3×3 grid of 9 frames from best pattern (for VLM scoring)

Must print to stdout:
```
TIER3_SCORE=<float>
TIER4_SCORE=<float>
```
(vision_score is computed by the eval harness from contact_sheet.png)

## Evaluation Procedure

1. Run `solution/run.py` with seeds 1, 2, 3 on T4 GPU (Modal)
2. For each seed: compute tier3_score, tier4_score, call VLM for vision_score
3. `composite = mean over seeds of (0.30 * t3 + 0.45 * t4 + 0.25 * v)`
4. Timeout: 20 minutes per seed (wall-clock kill)

## Baselines and Targets

| Level | Composite | Notes |
|---|---|---|
| Trivial (random noise) | ~0.02 | No coherent structure |
| Standard Flow-Lenia | ~0.15–0.20 | Stable solitons, no repair/replication |
| **Target** | **≥ 0.55** | Reliable Tier 3 + at least one replicating daughter |
| ASAL upper bound | ~0.25 | Estimated from ASAL results mapped to this metric |

**Direction**: maximize

**Metric name**: `composite_life_likeness`

## Known Pitfalls and Safeguards

| Pitfall | Safeguard |
|---|---|
| Edge effects in daughter detection | Toroidal pad-wrap before `scipy.ndimage.label` |
| Homeostasis trivially high for static blobs | Self-repair score de-weights static non-repairing patterns |
| VLM hallucination / inconsistency | 3 independent VLM calls per seed, average; cache by image hash |
| Substrate that cheats by memorizing seeds | Seeds only control initial mass distribution, not rule parameters |
| CLIP heredity gaming (structurally identical daughters) | Replication count must be ≥ 1 (independent of heredity) |
| Mass drift in creative substrates | Soft 5% warning logged; hard clamp(0,1) prevents NaN propagation |

## Boundary Conditions

**Toroidal** (wrap-around) everywhere: convolutions, advection, damage application, daughter
detection. This is the standard in Lenia/Flow-Lenia literature and prevents edge artifacts.

## Hardware

- Grid: 256×256, toroidal
- Steps: up to 20,000 discovery + behavioral tests
- Backend: PyTorch with CUDA (T4 GPU via Modal); harness warns and falls back to CPU if CUDA unavailable
- Estimated wall-clock: 5–10 min per seed on T4

## Citation

Primary references for metric components:
- Flow-Lenia substrate: Plantec et al. (2023), arXiv:2212.07906
- SSIM: Wang et al. (2004), IEEE Trans. Image Processing 13(4):600–612
- CLIP embeddings: Radford et al. (2021), arXiv:2103.00020
- Connected-component daughter detection: standard `scipy.ndimage.label` (SciPy 1.x)
