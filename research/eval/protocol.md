# Evaluation Protocol

## What We Measure

We measure how life-like a discovered pattern is across two behavioral tiers:

**Tier 3 — Robustness**: Does the pattern maintain its structure under perturbation and recover
from damage? A truly robust pattern should persist over thousands of timesteps (homeostasis) and
reconstruct its morphology after localized mass removal (self-repair).

**Tier 4 — Reproduction**: Does the pattern produce spatially distinct copies of itself that
inherit its structural character? Genuine reproduction requires both a replication event (daughter
count) and hereditary fidelity (parent-daughter structural similarity in embedding space).

**Semantic validation**: A vision model reads a 9-frame contact sheet spanning the pattern's
lifecycle and scores it on five dimensions: spatial coherence, locomotion, self-repair evidence,
replication evidence, and morphological complexity. This provides a human-aligned check orthogonal
to the algorithmic scores.

## How to Measure

### Step 1: Load the solution

```bash
uv run python solution/run.py --seed <int> --grid-size 256 --output-dir /tmp/eval_output
```

The solution runs its discovery pipeline and writes:
- `discovered_patterns.npz`: top-5 patterns as float32 arrays (C, 256, 256)
- `gifs/pattern_{i}.gif`: lifecycle GIF for each pattern
- `contact_sheet.png`: 3×3 grid of 9 equally-spaced frames from best pattern
- Prints `TIER3_SCORE=<float>` and `TIER4_SCORE=<float>` to stdout

### Step 2: Compute Tier 3 score (self-repair + homeostasis)

**Homeostasis test** (runs during discovery):
1. Load the best discovered pattern from `discovered_patterns.npz`
2. Place it on a fresh 256×256 toroidal grid
3. Run 5,000 steady-state steps using the solution's substrate
4. Record total mass at each step: `M(t) = state.sum()`
5. `homeostasis = 1 - min(std(M) / mean(M), 1.0)`

**Self-repair test**:
1. Run pattern for 500 warm-up steps to confirm stability
2. Identify non-empty pixels: `mask = state.sum(axis=0) > 0.01`
3. Select a contiguous rectangular region covering ~20% of `mask` pixels
   (deterministic: use seed to pick top-left corner from non-empty region)
4. Zero out all channels within that region: `state[:, r0:r1, c0:c1] = 0`
5. Run 500 recovery steps
6. Compute `self_repair = SSIM(pre_damage_state, post_recovery_state,
   win_size=11, gaussian_weights=True, data_range=1.0)` on sum-over-channels
   field normalized to [0,1]
7. Repeat for seeds 1, 2, 3 and average

`tier3_score = 0.6 * mean_self_repair + 0.4 * homeostasis`

### Step 3: Compute Tier 4 score (replication + heredity)

**Replication test**:
1. Isolate best pattern on clean 256×256 toroidal grid
2. Run up to 10,000 steps, checking every 100 steps for daughter patterns:
   - Sum state over channels → mass field (256×256)
   - Binarize: `mask = mass_field > 0.05 * mass_field.max()`
   - Toroidal pad-wrap by `kernel_radius` pixels, apply `scipy.ndimage.label`
   - Clip back to 256×256; filter out components < 10% of parent initial mass
   - Track parent as nearest centroid to t=0 position
   - Flag any non-parent component with centroid > 20px from parent, persisting ≥ 3 consecutive frames
3. `replication_count_score = min(n_confirmed_daughters / 3, 1.0)`

**Heredity test** (if daughters > 0):
1. Render parent and each confirmed daughter as 64×64 grayscale image
   (normalize mass field to [0,1], resize with bilinear interpolation)
2. Compute CLIP ViT-B/32 embeddings via `openai/clip-vit-base-patch32`
3. `heredity = max(0, mean_cosine_similarity(parent_emb, daughter_embs) - 0.5) * 2`
   (If no daughters: `heredity = 0.0`)

`tier4_score = 0.5 * replication_count_score + 0.5 * heredity`

### Step 4: Compute vision score (VLM assessment)

1. Read `contact_sheet.png` (3×3 grid of 9 frames)
2. Call `claude-haiku-4-5-20251001` with the image and structured rubric prompt
3. Parse JSON response for 5 scores in [0,1]
4. `vision_score = mean(spatial_coherence, locomotion, self_repair_evidence,
   replication_evidence, complexity)`
5. Cache by SHA256 of image bytes to avoid redundant calls

### Step 5: Compute composite

`composite = 0.30 * tier3_score + 0.45 * tier4_score + 0.25 * vision_score`

Repeat steps 1–5 for seeds 1, 2, 3. Report mean composite.

## Acceptance Criteria

- **Metric direction**: maximize
- **Deterministic**: same solution + same seed → same tier3/tier4 scores (VLM may vary ±0.05)
- **Seeds**: 3 seeds per evaluation
- **Timeout**: 20 minutes per seed wall-clock (hard kill)
- **Backend**: T4 GPU via Modal; PyTorch required for substrate simulation

## What Counts as a Solution

A directory containing:
- `substrate.py`: class inheriting `BaseFlowLenia` implementing `compute_flow`,
  `apply_growth`, `get_default_params`
- `searcher.py`: discovery algorithm calling `substrate.update_step`
- `run.py`: entry point accepting `--seed` and `--output-dir` flags
- `README.md`: description of substrate design choices

The eval harness provides `BaseFlowLenia` and all evaluation logic — the solution only
needs to implement the substrate variant and discovery algorithm.

## Known Pitfalls

- **Toroidal boundary**: All convolutions and daughter detection must use wrap-around padding
- **CUDA availability**: Harness warns at startup; falls back to CPU (slower, may approach timeout)
- **VLM variance**: Vision scores have ±0.05 noise; averaged over 3 seeds to reduce variance
- **Mass conservation**: Substrate variants may drift; soft 5% warning logged, hard clamp applied
- **Phase boundary fragility**: Patterns near phase boundary may die before behavioral tests;
  searcher should ensure ≥500-step persistence before reporting a pattern as stable
