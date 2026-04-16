# Metric Provenance

## Papers Reviewed

- **[Chan (2019)]** "Lenia: Biology of Artificial Life" — *Complex Systems* 28(3), arXiv:1812.05433
  Introduced continuous CA with 400+ species; defines mass, velocity, spread as primary phenotypic
  metrics. First systematic life-like pattern taxonomy. Basis for all downstream work.
  URL: https://arxiv.org/abs/1812.05433

- **[Chan (2020)]** "Lenia and Expanded Universe" — ALIFE 2020, arXiv:2005.03742
  Extends to multi-channel (recurrent ConvNet analogy), 3D/4D; reports self-replication in
  multi-channel variants. Defines geometric, metameric, fuzzy, resilient, adaptive, rule-generic
  behavioral descriptors.
  URL: https://arxiv.org/abs/2005.03742

- **[Plantec et al. (2023)]** "Flow-Lenia: Towards open-ended evolution in cellular automata
  through mass conservation and parameter localization" — Best Paper ALIFE 2023, arXiv:2212.07906
  Introduces mass conservation + parameter localization. Demonstrates locomotion, chemotaxis,
  obstacle navigation, self-division at 1024×1024 scale. Primary substrate for this campaign.
  URL: https://arxiv.org/abs/2212.07906

- **[Kumar et al. (2024)]** "Automating the Search for Artificial Life with Foundation Models"
  (ASAL) — arXiv:2412.17799, *Artificial Life* 31(3) 2025
  Uses CLIP/DINOv2 static-frame embeddings for substrate-agnostic search. Three modes: supervised,
  open-endedness, illumination. Explicitly notes video-language FM as future work (our gap).
  URL: https://arxiv.org/abs/2412.17799

- **[Etcheverry et al. (2020)]** "Hierarchically Organized Latent Modules for Exploratory Search
  in Morphogenetic Systems" (IMGEP-HOLMES) — NeurIPS 2020
  Intrinsically motivated goal exploration in Lenia; discovers solitons from soup in <15k steps.
  Defines behavioral characterization space via unsupervised VAE latent representations.
  URL: https://proceedings.neurips.cc/paper/2020/hash/33a5435d4f945aa6154b31a73bab3b73-Abstract.html

- **[Mordvintsev et al. (2020)]** "Growing Neural Cellular Automata" — *Distill* 2020
  Gradient-trained NCA demonstrating self-repair (Tier 3) in fixed target-image paradigm.
  Uses pixel-wise L2 loss and SSIM for reconstruction fidelity — directly informs our self_repair
  metric design.
  URL: https://distill.pub/2020/growing-ca/

- **[Wang et al. (2004)]** "Image quality assessment: from error visibility to structural
  similarity" — IEEE Transactions on Image Processing 13(4):600–612
  Defines SSIM formula with 11×11 Gaussian window (equation 13). This is the exact implementation
  adopted for self_repair scoring.
  URL: https://doi.org/10.1109/TIP.2003.819861

- **[Radford et al. (2021)]** "Learning Transferable Visual Models from Natural Language
  Supervision" (CLIP) — arXiv:2103.00020, ICML 2021
  Defines ViT-B/32 CLIP embedding. Used for heredity scoring (parent-daughter cosine similarity)
  and ASAL's baseline FM approach.
  URL: https://arxiv.org/abs/2103.00020

- **[Calcaterra & Boldt (2022)]** "Existence of Life in Lenia" — arXiv:2203.14390
  Mathematically proves Lenia's convergence to a well-defined integro-differential equation;
  validates simulation correctness. Provides rigorous basis for the substrate.
  URL: https://arxiv.org/abs/2203.14390

- **[Kerg et al. (2024)]** "Looking for Complexity at Phase Boundaries in Continuous CA"
  arXiv:2402.17848
  Phase Transition Finder for multi-channel Lenia; sampling near phase boundaries yields 2.3×
  more interesting patterns. Informs our searcher.py design (search near phase boundary).
  URL: https://arxiv.org/abs/2402.17848

## Existing Implementations / Repos

- **SakanaAI/asal** (github.com/SakanaAI/asal, ★2.8k) — reference ASAL implementation
  Status: ADAPT — reuse CLIP embedding utilities and CMA-ES search scaffolding; replace
  single-frame scoring with contact-sheet VLM scoring.

- **Chakazul/Lenia** (github.com/Chakazul/Lenia, ★4.1k) — canonical Lenia implementation
  Status: ADAPT — reuse kernel/growth function implementations; adapt to Flow-Lenia base class.

- **google/evojax** (github.com/google/evojax) — JAX-based evolutionary algorithms
  Status: SKIP — JAX dependency conflicts with our PyTorch substrate; evolutionary search logic
  will be reimplemented with scipy.

- **skimage.metrics.structural_similarity** — scikit-image SSIM implementation
  Status: USE — exact citation-compliant implementation with win_size=11, gaussian_weights=True.

## Existing Benchmarks / Datasets

- **Lenia species archive** (Chan 2019–2020): 400+ named species as parameter configurations
  Size: 400 configs, JSON format, available in Chakazul/Lenia repo
  Status: ADOPT — use as baseline patterns for behavioral tests and as diversity comparison.

- **ASAL discovered patterns**: Supplementary material of arXiv:2412.17799
  Size: ~50 patterns across 5 substrates
  Status: ADOPT — use Lenia subset as ASAL baseline for comparison.

## Chosen Metric

- **Name**: `composite_life_likeness`
- **Formula**:
  ```
  composite = 0.30 * tier3_score + 0.45 * tier4_score + 0.25 * vision_score

  tier3_score = 0.6 * SSIM(pre_damage, post_recovery) + 0.4 * (1 - min(CV_mass, 1))
  tier4_score = 0.5 * min(n_daughters/3, 1) + 0.5 * max(0, mean_clip_sim - 0.5) * 2
  vision_score = mean(spatial_coherence, locomotion, self_repair_evidence,
                      replication_evidence, complexity)  [from VLM rubric]
  ```
- **Direction**: maximize
- **Why this metric**: Captures three orthogonal evidence channels — algorithmic behavioral
  testing (Tier 3), replication detection (Tier 4), and semantic VLM assessment (vision).
  Addresses ASAL's acknowledged gap of temporal/behavioral evaluation.
- **Alternatives rejected**:
  - Pure FM novelty (ASAL): captures appearance diversity, not behavioral capability
  - Lyapunov exponents (Asymptotic Lenia): computationally expensive PDE reformulation
    required, not compatible with PyTorch substrate variants
  - Entropy/complexity measures: substrate-dependent, hard to normalize across variants

## Known Pitfalls

- **Daughter detection edge effects** → safeguard: toroidal pad-wrap before ndimage.label
- **Homeostasis trivially high for static blobs** → safeguard: self_repair de-weights blobs
  (a static blob won't repair because it doesn't deform under the CA dynamics)
- **CLIP heredity gaming** (daughter identical to parent) → safeguard: replication_count
  and heredity are independent; both must contribute for high tier4_score
- **VLM hallucination** → safeguard: 3 independent calls per seed, average; cache by hash
- **Mass drift in expressive substrates** → safeguard: soft 5% warning + hard clamp(0,1)
- **Phase boundary fragility** → expected: patterns near phase boundary may be unstable;
  searcher must persist candidates for ≥500 steps before scoring

## Baseline References

- Random / trivial: ~0.02 (incoherent noise, no structure)
- Standard Flow-Lenia (Orbium-class soliton): ~0.15–0.20 (to be measured empirically)
- ASAL upper bound (estimated, Lenia substrate): ~0.25
- **Target**: 0.55
