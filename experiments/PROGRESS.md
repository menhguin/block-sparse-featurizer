# Distribution-Aware Block Selection for BSF — Progress Notes

**Last updated: 2026-07-10 02:24 SGT (UTC+8)**
**Author:** Minh Nguyen (min-p author) + agent-assisted exploration
**Branch:** `distribution-aware-block-selection` (fork of `goodfire-ai/block-sparse-featurizer`)
**Related prior work:** min-p sampling — paper [arXiv:2407.01082](https://arxiv.org/abs/2407.01082) (ICLR 2025 oral); replication/iteration repo (min-p → min-z lineage): [github.com/menhguin/top_nsigma](https://github.com/menhguin/top_nsigma)
**Status:** exploratory / one evening's work. Numbers are on a non-gated timm mirror of DINOv3 ViT-B/16 (identical Meta checkpoint, hash 73cec8be) pending a bit-for-bit re-run on the now-approved `facebook/dinov3-*` weights. Shape/ordering conclusions are robust to this; treat absolute numbers as provisional.

---

## TL;DR

Port the **min-p / top-nσ** adaptive-truncation idea from LLM sampling onto **block selection** in a Block-Sparse Featurizer. Stock BSF uses **block top-k** (fixed count k=8 blocks per patch) — exactly analogous to top-k *sampling*, and subject to the same critique min-p was built for: a fixed count over-selects on simple inputs (admits noise) and truncates on complex ones (drops real concepts), because the true per-input concept count is a random variable.

**Headline result (honest, includes the negatives):**
1. Distribution-aware selection gives **NO win on in-distribution reconstruction R²** (parity with top-k) or on operating-point robustness. Two clean negatives.
2. But it IS **materially more selection-*correct***: at matched mean-L0, a noise-referenced (χ) threshold tracks each patch's *true* support, cutting total selection error ~21% vs fixed-k — and fixed-k's errors are **entirely on the complexity tails** (over-selects simple patches, truncates complex ones), which **cancel in the aggregate R²-average**. So the value is a *complexity-robustness / generalization* property, not a reconstruction win — exactly as min-p/top-nσ's real value was robustness (temperature-invariance), not higher greedy accuracy.
3. **noise-referenced ≫ max-referenced.** The min-p-style `p·max` port ("min-z") works poorly here because the max-norm block is often a generic/positional block — an untrustworthy anchor, the *opposite* of LLM sampling where the max logit is the model's best guess. A noise-floor (χ) reference is the right transplant.

---

## Why the analogy (the core mapping)

| LLM sampling | BSF block selection |
|---|---|
| logits over vocab | pre-gate L2 norms over 256 candidate blocks (per patch) |
| truncation sampler picks which tokens survive | selector picks which blocks are "active" |
| **top-k** = fixed count | **block top-k** (stock BSF, k=8) = fixed count |
| min-p / top-nσ = variable count from distribution shape | this work: variable count from per-patch norm distribution |
| per-token concept count varies | per-patch concept count varies (measured CV ≈ 0.31–0.68) |

---

## Experiments (chronological — the reasoning arc matters)

All on DINOv3 ViT-B/16 patch activations of the 300 rabbit images shipped with the repo (58,800 patches × 768-d), 256 blocks × group_size 3. Scripts in `experiments/scripts/`, outputs + findings in `experiments/results/`.

### Step 0 — distribution shape (`02_measure_distributions.py`, `FINDINGS.md`)
- Block-norm distributions are **NOT cleanly bimodal** (small-pool effect: 256 blocks, unlike a 128k-token vocab).
- BUT **inactive-block norms fit χ₃ near-perfectly (KS-D = 0.0019)**. group_size=3 → norm of a 3-dim near-Gaussian → χ₃. This means "is block g active?" is a well-posed hypothesis test against a characterized null, with a false-positive-rate (α) interpretation. Training *manufactures* the clean noise floor the top-nσ sampling paper could only assume.
- k=8 is **unjustified in the repo** (no rationale; README uses 16, notebooks use 8) and cuts a **smooth continuum** (rank-8 vs rank-9 norm differ ~5%).

### Step 1 (E1) — Pareto on frozen top-k magnitudes (`03_pareto_e1.py`, `FINDINGS_E1.md`)
- All adaptive selectors **lose** to top-k at matched L0 — **but the test is biased** (magnitudes were trained *with* top-k in the loop; away game for other selectors).
- Signal despite bias: **perpatch_chi ties top-k (−0.003); min_z loses 10× more (−0.03)** → noise-ref ≫ max-ref confirmed. min_z dropped from contention.

### Step 2 (E2) — co-adapted training + operating-point robustness (`04_robustness_e2.py`, `FINDINGS_E2.md`)
- Trained chi-floor selection *in the loop*; swept train × eval operating points.
- **top-k(8) is the most robust single model** (sags only ~0.05 eval-swept, incl. up to L0=16). No chi robustness advantage on this axis.
- Why the temperature-invariance analogy breaks here: sparsity budget is a **train-time** property (baked into the learned dictionary), not an **eval-time** transform. No invariance for chi to exploit. Also found: chi's q→L0 map is unstable (chaotic budget targeting) — a real practical downside.

### Step 3 (fork a) — monosemanticity at matched L0 (`05b_monosemanticity_matched.py`, `FINDINGS_monosemanticity.md`)
- Passenger-firing rate, feature selectivity, top-firing input coherence. chi wins all four **but every margin is within noise.** No material monosemanticity win on vision-BSF.

### Step 4 (Test A) — selection-correctness vs per-patch TRUE support (`06_testA_complexity.py`, `FINDINGS_testA.md`) ← **the result that flips the verdict**
- Defined a rule-agnostic per-patch "true support" k* = min #blocks reaching 95% of that patch's **own achievable peak R²** (peak, not full-256 — the overcomplete non-orthogonal 768×768 decoder makes per-patch R² *peak ~k10 then decline*, a finding in itself).
- **k*: mean 5.4, CV 0.31** (real per-patch support variance, p5=3, p95=8).
- At matched mean-selected count: **chi-floor tracks k* (corr 0.58, total selection error 0.99); fixed-k is flat (corr 0, error 1.26 — 21% worse).**
- **fixed-k's errors are entirely on the tails**: over-selects simple patches by +1.47 blocks (admits noise), truncates complex patches by −1.84 (drops real concepts), ~0 error at the mean. chi spreads small errors evenly. This is the top-nσ mechanism made visible.
- **Reconciliation:** tail errors cancel in the aggregate R²-average (why E1/E2 saw parity) but not in selection-correctness → the win is robustness across complexity, measured on the right axis.

---

## Current best hypothesis

On vision-BSF, distribution-aware block selection is **not** a reconstruction improvement, but **is** a *complexity-robustness / selection-correctness* improvement: a **noise-floor-referenced (χ) threshold** matches per-patch true support where fixed-k is structurally blind. Whether this translates into a downstream benefit that a practitioner cares about is the open question — the strongest next test is **cross-distribution transfer** (train on one image distribution, evaluate selection-correctness on a different-complexity distribution without re-tuning — does one χ setting hold where fixed-k needs re-tuning?).

## Method-porting menu

`experiments/results/sampling_methods_catalogue.md` — 20 min-p-lineage sampling methods classified by *reference point* (max / cumulative-mass / entropy / noise-floor / feedback / local-curvature) and mapped to BSF ports. Top candidates to prototype next: **η-sampling** (entropy-referenced dual floor), **REAL-sampling-style learned input-conditioned floor** (noise-floor v2), **Mirostat-style feedback control** (fixes χ's unstable hyperparameter), **XTC-style dominant-block exclusion** (the only method in the literature that treats "most probable ≠ trustworthy" — directly targets our max-block problem).

## Reproduce

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e . && uv pip install timm
# activations (one-time; needs HF access to a DINOv3 ViT-B/16 or the timm mirror):
python experiments/scripts/00_dino_activations.py
python experiments/scripts/01_train_baselines.py
python experiments/scripts/02_measure_distributions.py   # step 0 shape
python experiments/scripts/03_pareto_e1.py               # E1
python experiments/scripts/04_robustness_e2.py           # E2
python experiments/scripts/06_testA_complexity.py        # Test A (the key one)
```
`.npy` activations and `.pt` model weights are gitignored (regenerable from the scripts above).

## Honest caveats
- Numbers on the timm DINOv3 mirror, not `facebook/dinov3-*` (now approved; re-run pending). Shape/ordering robust.
- All within-rabbit (single object class). Cross-distribution transfer not yet tested — that's the load-bearing next experiment for the robustness claim.
- k* is built from norm-ranked greedy addition → mildly sympathetic to norm-based selection generally, but neutral for the fixed-vs-adaptive comparison (both rank by norm).
- One evening's exploration; not a paper. Sharing to get a second researcher's read on whether the complexity-robustness framing is worth pushing.
