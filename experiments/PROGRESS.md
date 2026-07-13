---
title: "BSF distribution-aware block selection — MASTER PROGRESS"
created: 2026-07-11
updated: 2026-07-11
type: concept
tags: [ai-research, programmable-models, bsf, min-p, fdr, interpretability]
status: active exploration — FDR-v1 works, Test B (transfer) is the load-bearing next test
supersedes_as_entrypoint: [bsf-shape-findings, bsf-e1-pareto-findings, bsf-e2-robustness-findings, bsf-monosemanticity-findings, bsf-testA-findings, bsf-testA-extended-findings, bsf-fdr-v1-findings]
---

# BSF Distribution-Aware Block Selection — Master Progress

**Single entry point** for the min-z/χ-floor/FDR investigation (task #756). The dated
`bsf-*-findings-*.md` files are the detailed audit trail per experiment; this doc is
the narrative + current state. Experiment code + data: `~/local/ai-research/2026-07-09-bsf-distribution-aware-thresholds/`.
Public branch: `github.com/menhguin/block-sparse-featurizer/tree/distribution-aware-block-selection`.

## The one-line thesis (evolved over the session)

Started: "port min-p/top-nσ adaptive truncation onto BSF block selection (replace fixed
top-k=8 with a distribution-aware threshold)." Ended: **"bring null-referenced,
α-controlled (FDR-style) variable-size selection to block-sparse featurizers, using an
empirically-MEASURED χ₃ noise null — a principled, transferable alternative to the
arbitrary fixed-k, whose payoff is complexity-robustness/generalisation, not
in-distribution reconstruction."**

## The problem

Stock BSF picks the fixed top-k=8 largest-norm blocks per input patch — analogous to
top-k *sampling*. But the true number of concepts per patch is a latent random variable
(measured CV ≈ 0.31). Fixed-k over-selects on simple patches (admits noise) and truncates
on complex ones (drops real concepts). We want ONE rule that adapts the count per input
and transfers across distribution shift without re-tuning.

## Experiment ledger (chronological — the reasoning arc)

| # | Experiment | Finding | Detail file |
|---|---|---|---|
| 0 | Distribution shape | NOT bimodal, BUT inactive-block norms fit **χ₃ near-perfectly (KS-D=0.0019)** → clean measured noise null exists. k=8 unjustified, cuts a smooth continuum. | `bsf-shape-findings-2026-07-09` |
| E1 | Pareto on frozen top-k magnitudes | All adaptive lose at matched L0 (biased test — magnitudes co-adapted to top-k). Signal: **noise-ref (χ) ≫ max-ref (min-z)** by 10×. min-z dropped. | `bsf-e1-pareto-findings-2026-07-09` |
| E2 | Co-adapted training + operating-point robustness | top-k(8) most robust single model; no χ advantage on THIS axis. Temp-invariance analogy breaks (sparsity budget is train-time, not eval-time). χ has unstable q→L0 targeting. | `bsf-e2-robustness-findings-2026-07-09` |
| fork-a | Monosemanticity at matched L0 | χ wins all 4 metrics but **within noise**. No material monosemanticity win. | `bsf-monosemanticity-findings-2026-07-09` |
| **A** | **Selection-correctness vs per-patch true support k*** | **THE REFRAME (Minh's pool-variance instinct).** k* mean 5.4, CV 0.31. χ-floor tracks k* (corr 0.58, err 0.99); **fixed-k flat (corr 0, err 1.26)**; fixed-k errors ENTIRELY on complexity tails (over-select simple +1.47, truncate complex −1.84). E1/E2 saw parity because tail errors CANCEL in the R²-average. Value = complexity-robustness, measured on the right axis. | `bsf-testA-findings-2026-07-09` |
| A-ext | p-less / DiffSampling / Top-H vs k* | **p_less_L1 best support-tracker (corr 0.695)** but mis-scaled/over-selects 2.6×. DiffSampling ~random (no cliffs, confirms smooth per-patch profile). Top-H ANTI-correlated (−0.53; entropy≠support in BSF). | `bsf-testA-extended-findings-2026-07-10` |
| DR | Exa deep research | Verdict: **FDR (Benjamini-Hochberg) IS the canonical answer** to "α-referenced threshold from a null." Conformal prediction = the transfer-guarantee frame. Prior art: **Enkhbayar 2025** (Model-X knockoffs for SAE FDR) — nearest neighbor, but supervised/dataset-level/manufactured-null vs our unsupervised/per-input/measured-null. | `deep-research-generalisable-selection-2026-07-10` |
| **v1** | **Per-patch BH on χ₃ p-values** | **UNTUNED α=0.10 → corr 0.62 with k*** (2nd best, beats tuned χ-floor). Principled dimensionless knob (target FDR) matches benchmark-tuned heuristics WITHOUT tuning → answers "sound scale-setting." Solves per-sample threshold. BH≡BY here (dependence not distorting). | `bsf-fdr-v1-findings-2026-07-11` |

## Current landscape: which reference-point families work in BSF

- **FDR / null-referenced (BH on χ₃)** — LIVE. Principled α knob, competitive corr, THE current lead.
- **Participation-ratio (p-less)** — best raw ranking (0.695) but needs α-calibration for scale.
- **Noise-floor heuristic (χ-floor)** — works, but its q is an arbitrary multiplier (FDR supersedes it with a meaningful knob).
- **Max-referenced (min-z)** — DEAD (max block untrustworthy/generic).
- **Gap/curvature (DiffSampling, Min-k)** — DEAD (no cliffs, per-patch or aggregate).
- **Entropy (Top-H, η, ACS, GUARD)** — DEAD/suspect (Shannon entropy anti-correlates with support here).

## Answers to Minh's two theory questions
- **Q: sound way to set the scale (non-arbitrary knob)?** → α as target FDR against the measured χ₃ null. Dimensionless, same meaning across distributions. DEMONSTRATED: untuned α=0.10 matches tuned heuristics.
- **Q: do knockoffs (Enkhbayar) make sense here?** → NO. Knockoffs manufacture a null (we measured one) and are supervised/dataset-level (we're unsupervised/per-input). BH-on-measured-χ is the matched tool; only borrow = dependence-robustness (BY variant, tested, no diff).

## Datasets (reference)
- Our experiments: DINOv3 ViT-B/16 (timm mirror) on the 300 rabbit images shipped with BSF repo.
- BSF paper's actual evals: DINOv3 (rabbits + shadow/lighting manifolds), **InceptionV1** (curve detectors), **SDXL** (diffusion steering). Rabbits = demo only.
- Enkhbayar 2025: Pythia-70M SAE latents on SST sentiment (supervised).

## Test B (attempted 2026-07-11): FAILED + CONFOUNDED
Froze rabbit knobs, applied rabbit dictionary to Imagenette. Everything failed (adaptive rules under-selected badly: needed k*=10.1, BH α=0.10 gave 2.1). BUT the test is CONFOUNDED — it conflates "does the selection RULE transfer" with "does the rabbit-trained DICTIONARY transfer." Dictionary domain-shift swamps the signal: the χ₃ null measured on rabbits doesn't describe Imagenette inactive-block norms, so σ̂ is mis-estimated. ROOT CAUSE confirmed: rabbit dict reconstructs Imagenette at R²=0.15 vs 0.81 native — dictionary literally can't represent the images, norm contrast collapses 11.8→3.4, so nothing clears noise floor. Not a selection failure. Honest null+confound, NOT a clean disconfirmation. Detail: `bsf-testB-findings-2026-07-11`. FDR-v1's in-distribution result STANDS (Test B doesn't touch it). **Transfer claim remains UNPROVEN — do not claim it.**

## Test B'' (2026-07-11): CLEAN TRANSFER WIN
Within-rabbit complexity split (LOW k*=4.2 vs HIGH k*=6.8), same dictionary (zero domain shift). Calibrate knob on one half, freeze, test on other. **Adaptive rules cut error 40-67% vs fixed-k and auto-adapt the count in the right direction with a FROZEN knob** (LOW→HIGH: BH widens 4→5.5; HIGH→LOW: BH narrows 7→4.9, corr 0.60). fixed-k stuck at calibration count, wrong by ~2.8 on the other half. First clean evidence the α-knob transfers across a complexity gap where fixed-k structurally can't. Caveats: BH≈chi (FDR's edge is the principled knob, not raw perf); realized-FDR proxy saturated/uninformative (needs real calibration test); LOW→HIGH corr weak (compressed variance) but error win holds. Detail: `bsf-testB2-findings-2026-07-11`.

## Test B''' (next): domain transfer with competent dictionary

Freeze α on rabbits, apply the χ₃-BH rule to a DIFFERENT image distribution (Imagenette,
multi-class — downloading), check realized FDR stays ≈ α while fixed-k's effective error
rate swings. B′: train dictionary on a BROAD set (Imagenette or rabbits+Imagenette), calibrate α on one held-out subset, test on another of similar dictionary-competence — isolates rule transfer from dictionary transfer. B″ (cheapest honest version): within-rabbit low- vs high-complexity split, calibrate α on one half, test realized FDR on the other — zero dictionary shift. Where α's dimensionless-transfer property should beat fixed-k — converting FDR-v1 from "principled option at parity" to "provably robust
where fixed-k breaks." The whole robustness thesis rides on this.

## Caveats carried throughout
- All numbers on timm DINOv3 mirror (identical weights, non-gated) — facebook/dinov3 now approved, bit-for-bit re-run pending for any pub numbers.
- k* is norm-ranked-greedy-defined → neutral for fixed-vs-adaptive comparison but not a norm-independent oracle.
- All within-rabbit until Test B lands.
