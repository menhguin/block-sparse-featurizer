# Generalisable Selection Across Unbounded Pool Size — Exa Deep Research

**Task:** r_01kx6p7svx588jshz22qbdhp43 · model exa-research-pro · completed 2026-07-11 03:02 +08

---

# Direct answer to the deep question

Short answer: the problem is partially solved in existing literatures but the exact framing — a per-input, null-referenced threshold that (a) uses a well-specified noise null for inactive blocks (here a chi / χ distribution), (b) exposes a single, dimensionless free parameter interpreted as a false-positive / miscoverage rate α, and (c) provably transfers across inputs and some classes of distribution shift — does not have a single canonical off‑the‑shelf method that covers all desiderata for sparse dictionary / SAE latents. Two near-canonical approaches should be borrowed (and combined) as the best candidates: (1) FDR-style procedures (Benjamini–Hochberg / Storey / Model‑X knockoffs) applied to per-block p-values computed from the chi noise-null; and (2) conformal-style calibrated set construction (split or weighted conformal) adapted to per-input selection via likelihood-ratio weighting or group/Mondrian calibration. The known direct prior-art applying a per-input, null-referenced FDR threshold to SAE latents is limited to a single recent work (Model‑X knockoffs for SAE features) with finite-sample FDR guarantees; otherwise the SAE/interpretability literature mostly uses fixed-k or learned global thresholds, not null-referenced per-input α-controls [Benjamini & Hochberg (1995)](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf) [Storey (2002)](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf) [Barber & Candès (Model‑X / knockoffs literature)](https://statweb.stanford.edu/~candes/knockoffs/), and the direct SAE knockoffs application [Enkhbayar (2025)](https://arxiv.org/html/2511.11711v1).


# Per-frontier canonical methods, exact references, and what they buy you

- Classical multiple testing / shrinkage (canonical borrow):
  - Benjamini–Hochberg FDR (BH): order per-item p-values and reject those p ≤ (k/m)·α; guarantees control of expected FDR at level α under independence (and certain positive-dependence conditions) [Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf). BH therefore produces a variable-size selection (k depends on the dataset/input) with α interpreted as target FDR.
  - Storey’s q-value / pFDR estimation: estimate π0 (proportion of true nulls) and compute q-values, then select items with q ≤ α; typically more powerful than BH while still targeting FDR/pFDR in expectation [Storey 2002](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf).
  - Model‑X knockoffs / knockoff+: creates knockoff variables to form a valid null for each feature and selects a variable set with finite-sample FDR control at target q, without strong assumptions on feature dependence when Model‑X holds [Candès etal.; Barber & Candès literature; SAE application: Enkhbayar 2025]. The SAE-specific application demonstrates how to apply a knockoff FDR threshold to latent codes with finite-sample guarantees when the latent distribution (Model‑X) can be estimated [Enkhbayar 2025](https://arxiv.org/html/2511.11711v1). 
  - Shrinkage / universal thresholds (VisuShrink): threshold λ = σ̂·√(2 log n) on coefficient norms gives a dimensionless tail control that adapts to n and σ; VisuShrink is a classical null-referenced threshold with near-minimax risk guarantees under Gaussian noise [Donoho & Johnstone 1994](https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf) and is directly analogous to setting a χ-null exceedance threshold for L2 norms.

- Conformal prediction / calibrated set-valued prediction (canonical borrow):
  - Split conformal (distribution-free marginal coverage 1−α under exchangeability): build per-input sets by comparing a per-input nonconformity score to the (1−α) empirical quantile of calibration scores; output set sizes vary per input, and α is miscoverage [Lei etal. 2018](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf). 
  - Weighted conformal for covariate shift: re-weight calibration scores by estimated likelihood ratios to obtain valid coverage under covariate shift if the likelihood ratios are known or well-estimated [Tibshirani etal. 2020](https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf). This gives a route to transfer α across certain shifts. 
  - Mondrian / class-conditional conformal: groupwise calibration to obtain exact coverage conditional on pre-specified groups; useful when per-input groups (e.g., input difficulty strata) are available [Angelopoulos & Bates (APS/RAPS literature), Mondrian variants]. APS/RAPS provide finite-sample marginal coverage, and Mondrian variants provide group-conditional coverage under exchangeability within groups [Angelopoulos & Bates 2020/2021](https://openreview.net/pdf?id=eNdiU_DbM9) [Mondrian variants].

- Compressed sensing / sparse-coding (canonical borrow):
  - Lasso / group-Lasso with λ chosen via information criteria or stability selection can adaptively select sparsity and thereby variable-size supports; guarantees (consistency or exact recovery) hold under restricted eigenvalue / RIP or incoherence assumptions [Tibshirani 1996; Candès & Tao 2006]. See phase-transition analyses for sharp recovery regimes [Donoho & Tanner 2009](https://doi.org/10.1098/rsta.2009.0157). 
  - Stability Selection (Meinshausen & Bühlmann 2010) uses subsampling and selection-frequency thresholds to control expected number of false positives and provides a direct error-rate interpretation (a bound on expected false positives) with robustness to correlated designs under stated assumptions [Meinshausen & Bühlmann 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x). 
  - Knockoff variants for support recovery in CS (recent work) bring explicit FDR guarantees for support selection under Model‑X-style assumptions, and have been applied to compressive sensing setups [Knockoff-CS literature].

- Neural‑interpretability / SAE literature (what exists and what doesn’t):
  - Most SAE/dictionary-latent work uses fixed-k, batch-top-k, learned global thresholds, or L0 targets (TopK/BatchTopK, JumpReLU, L0 constrained training) to control sparsity per input or per batch; these do not provide a principled per-input α that guarantees per-input false positive control under a null model [BatchTopK, JumpReLU, Huben etal.]. Recent targeted prior-art that explicitly implements null-referenced FDR control for SAE latents is one arXiv contribution applying Model‑X knockoffs to SAE features and reporting finite‑sample FDR control for latent-feature selection [Enkhbayar 2025](https://arxiv.org/html/2511.11711v1). Other SAE papers either tune k or L0 or learn thresholds but do not present per-input α-style null guarantees.


# Does a null-referenced, per-input α threshold provably generalize across samples/different support sizes?

- Short interpretation: Yes in principled statistical frameworks, but only under their respective assumptions. Each canonical method gives a transferable, dimensionless tuning parameter α with provable guarantees — but only under the method’s assumptions (exchangeability or Model‑X, correct likelihood-ratio estimation, independence/allowed dependence classes, RIP/compatibility conditions, etc.). Therefore, a one‑stop guarantee that a single α will provably control false positives across arbitrary distribution shifts and arbitrary, unbounded per-input support is not available; instead you must choose a framework whose assumptions match the regime of expected shifts and noise.

  - FDR methods: BH/Storey control expected FDR at α under independence or certain dependence classes; Model‑X knockoffs deliver finite‑sample FDR control at α when the latent distribution (Model‑X) is correctly specified or can be simulated [Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf) [Storey 2002](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf) [Enkhbayar 2025](https://arxiv.org/html/2511.11711v1). FDR control is an expectation guarantee (over sampling or randomness in data or knockoff construction) rather than per-input guaranteed control.

  - Conformal methods: split / full conformal guarantee marginal coverage 1−α in finite samples under exchangeability; weighted conformal provides corrected coverage under covariate shift if the test/training likelihood ratio is known/estimated [Lei etal. 2018](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) [Tibshirani etal. 2020](https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf). Conformal methods are therefore probably the closest formalism that gives a dimensionless α with finite-sample transfer guarantees — but the core assumptions (exchangeability or accurate likelihood-ratio estimation / weighting) must hold.

  - Compressed-sensing guarantees: Lasso/Stability selection/phase-transition results give high-probability support-recovery guarantees that scale with sparsity, sensing matrix properties, and noise magnitude; these are not single-α universal-transfer guarantees across arbitrary shifts, but do provide asymptotic/finite-sample conditions under which support recovery (hence false positive control) holds [Candès etal. 2006](https://doi.org/10.1109/TIT.2005.862083) [Meinshausen & Bühlmann 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x) [Donoho & Tanner 2009](https://doi.org/10.1098/rsta.2009.0157).


# Explicit mapping: which literature explains your five empirical findings

1) Inactive-item norms fit a χ distribution (df = block dimensionality ≈ 3) almost perfectly (KS D = 0.0019).
- Explanation / literature: Classical null modeling and wavelet‑shrinkage / VisuShrink-style thresholding rely on a well-specified noise null for coefficients (often Gaussian-derived leading to χ/χ² distributions for norms or squared norms). Donoho & Johnstone’s universal thresholding and the general χ/χ² theory explain why inactive-block L2 norms (sum of squared standard-normal-like components) are expected to follow a χ (or χ²) null when inactive components are Gaussian/noise-like [Donoho & Johnstone 1994](https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf) and classical chi-square references (goodness-of-fit theory) explain the distributional fit [chi-square theory](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900058/). This directly supports computing per-block p-values from the χ-null.
  - Key refs: Donoho & Johnstone (VisuShrink), classical χ/χ² theory [Donoho & Johnstone 1994](https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf) [Chi-square test exposition](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900058/).

2) Max‑referenced thresholds (keep block if score ≥ p·max_score) do poorly because the max is often a generic/dominant/positional artifact rather than reliable signal.
- Explanation / literature: The empirically poor behaviour of max-referenced methods is consistent with observations in interpretability and generative‑model sampling literatures that metrics tied to the single largest coordinate are brittle when that coordinate is dominated by systematic artifacts (positional/dominant features) rather than contentful signal; top‑p (nucleus) and min‑p style samplers were invented for different trust regimes in token sampling where the top probability is meaningful (LLM logits) [Holtzman etal., nucleus sampling 2019] while SAE latents can have a dominant artifact that makes any rule referencing the single max unreliable [Holtzman etal. 2019](https://arxiv.org/abs/1904.09751). Work on SAE failure modes also documents dominant/artifactual latents affecting naive thresholding [Huben etal. 2024; BatchTopK discussions].
  - Key refs: Holtzman etal. (nucleus sampling) for contrast with where max-based rules make sense [https://arxiv.org/abs/1904.09751], SAE literature noting artifacts and the need for batch-aware or more principled selection [BatchTopK SAE; Huben etal.].

3) Collision-probability / participation-ratio (p‑less sampler: sum of squared normalized scores = exp(−Rényi‑2 entropy)) ranks inputs by true support best of all methods tried (Pearson r ≈ 0.70) but mis-scales (over-selects ≈ 2.5×).
- Explanation / literature: The participation ratio / Rényi‑2 based effective-support estimator appears in physics and signal-analysis literature as an estimator of effective dimensionality (effective number of nonzero modes). In statistics, Higher Criticism and effective-sparsity concepts relate aggregated second-moment/energy-based metrics to true sparsity and are theoretically motivated for ranking/detecting sparse signals [Donoho & Jin 2004]. Participation ratio gives excellent rank-ordering (sensitivity to effective support) but lacks a calibrated null tail for absolute selection count; this is why it ranks well but over-selects absent a null‑referenced α correction [Donoho & Jin 2004](https://arxiv.org/abs/math/0410072). 
  - Key refs: Donoho & Jin (Higher Criticism) and the general literature on participation ratio / Rényi entropies and effective support [Donoho & Jin 2004](https://arxiv.org/abs/math/0410072).

4) Shannon entropy anti‑correlates with true support (corr ≈ −0.53) — entropy is not a proxy for "how many items are present" here.
- Explanation / literature: Shannon entropy measures uncertainty/spread, not the count of active supports; high entropy indicates distributed, low-contrast mass rather than a few strong activations. Information-theoretic reasoning therefore predicts anti-correlation between Shannon entropy and the number of high-norm, strongly present items in situations where activity is concentrated on a small number of blocks [Shannon 1948]. Rényi-2 (collision) focuses on squared-mass concentration and so correlates positively with support in this setting, while Shannon entropy can move oppositely [Shannon 1948]. 
  - Key refs: Shannon’s original information theory [https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf].

5) Knee/gap detection fails — the sorted-score curve is a smooth continuum with no knee per input.
- Explanation / literature: Knee/cliff detection algorithms assume a clear separation between signal and noise coefficients (a gap). Under weak or heterogeneous signals and smooth score decay (common in overcomplete dictionaries and many high-dimensional latent representations), the ordered-score curve has no sharp gap and knee detectors fail; this is precisely the regime where universal shrinkage (VisuShrink/SURE) or HC-style aggregated tests are designed to operate rather than gap heuristics [Donoho & Johnstone 1994; Donoho & Jin 2004]. SURE and universal thresholding assume a noise null and perform continuous shrinkage rather than hard-knee truncation [Donoho & Johnstone 1994] [Donoho & Jin 2004].
  - Key refs: Donoho & Johnstone (wavelet shrinkage / VisuShrink) and Donoho & Jin (Higher Criticism) provide frameworks for smooth-decay regimes [Donoho & Johnstone 1994](https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf) [Donoho & Jin 2004](https://arxiv.org/abs/math/0410072).


# Ranked shortlist of concrete threshold rules to try next (each with alpha interpretation and their transfer guarantee)

1. Model‑X knockoff FDR on χ-null per‑block p-values (best canonical match for SAE latents)
   - Algorithm: For each input, compute per-block statistic (e.g., L2 norm). Use the χ (or χ²) null (df = block-dim) to obtain p-values for each block under inactivity. Construct Model‑X knockoff latents that mimic joint latent distribution; compute feature importance statistics on (original, knockoff) pairs and apply the knockoff+ selection rule that guarantees FDR ≤ q. 
   - Free parameter: q interpreted as target FDR (false-discovery rate). 
   - Guarantee: Finite-sample FDR control at level q under Model‑X (i.e., when the latent distribution used to construct knockoffs is correct or when knockoffs satisfy the Model‑X condition); transfers across inputs because FDR is controlled in finite samples for the chosen q [Barber & Candès / knockoffs literature; SAE application Enkhbayar 2025] [Enkhbayar 2025](https://arxiv.org/html/2511.11711v1). 
   - Why try: directly leverages your measured χ null and gives a rigorous α interpretation with finite-sample guarantees for support selection in latent spaces where Model‑X is plausible.

2. Benjamini–Hochberg FDR on per-block χ p-values (simple, low-cost)
   - Algorithm: Compute per-block p-values from χ (df=3) null; apply BH across the ~256 blocks per input and select rejections at level α. 
   - Free parameter: α interpreted as target expected FDR. 
   - Guarantee: Expected FDR ≤ α under independence (or certain positive dependence regimes); in practice BH is often robust but may be anti‑conservative under arbitrary dependence [Benjamini & Hochberg 1995] [Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf). 
   - Why try: minimal machinery, directly uses your χ-null fit, produces variable-size selections per input, and is straightforward to implement and benchmark.

3. Weighted split-Conformal thresholding on normalized scores (for covariate shift)
   - Algorithm: Use a nonconformity score per block (e.g., p-value or normalized norm). Calibrate on held-out data and, if test/training covariate distribution differs, apply likelihood-ratio weighting to calibration residuals per the weighted conformal recipe; select blocks for which nonconformity ≤ calibrated quantile giving marginal miscoverage ≤ α. 
   - Free parameter: α is miscoverage (fraction of blocks expected to be false positives per the conformal guarantee interpretation). 
   - Guarantee: Marginal (or weighted) finite-sample coverage 1−α under exchangeability (or weighted exchangeability under covariate shift if correct weights / likelihood ratios are used) [Lei etal. 2018; Tibshirani etal. 2020] [https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf] [https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf]. 
   - Why try: conformal provides a direct route to per-input variable-size selection with explicit coverage guarantees that can be adjusted to covariate shift when a good density-ratio estimator is available.

4. VisuShrink / χ-universal threshold on norms (analytic tail control)
   - Algorithm: Estimate noise σ (from inactive blocks or calibration samples) and set threshold λ = σ̂√(2 log m) where m≈256. Keep blocks with norm ≥ λ (or use soft thresholding). 
   - Free parameter: implicitly sets a tail-exceedance probability tied to log m; can be adapted to target α by adjusting the multiplicative constant. 
   - Guarantee: Near‑minimax MSE and high-probability tail control under Gaussian noise assumptions (thus, a tail‑probability‑style α interpretation under Gaussian null) [Donoho & Johnstone 1994] [Donoho & Johnstone 1994](https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf). 
   - Why try: simple closed-form rule directly matched to your χ null; effective when the Gaussian null is correct and for controlling Type I risk in a dimension-aware way.

5. Stability Selection applied to χ p-values (subsampling + frequency threshold)
   - Algorithm: Subsample inputs or latent activation vectors repeatedly; apply a per-subsample selection (e.g., BH or λ‑thresholding) and compute selection frequencies; select blocks with frequency ≥ πthr. The method can be calibrated to bound expected number of false positives (E[V]) and thus provides an α‑style interpretation. 
   - Free parameter: selection-frequency threshold πthr or expected number-of-false-positives bound; convert to an α-style control via Meinshausen & Bühlmann’s formulas. 
   - Guarantee: Bounds on expected false positives (and related FWER control) under stated assumptions; more robust to correlations than naive single-run selection [Meinshausen & Bühlmann 2010] [https://doi.org/10.1111/j.1467-9868.2010.00740.x]. 
   - Why try: gives a robustness layer to handle dependence and instability in per-input latent activations.

6. Higher‑Criticism / HC‑max threshold (for detection of sparse, weak signals)
   - Algorithm: Compute p-values against χ null, compute HC statistic across p-value range, choose threshold corresponding to peak HC, and select p-values below that threshold. 
   - Free parameter: α is implicit in the decision rule via HC thresholding or target significance; HC is typically used for detection/ordering then combined with other rules for selection. 
   - Guarantee: Asymptotically optimal for detection of rare/weak signals in high dimensions under specific sparsity regimes and null models [Donoho & Jin 2004] [https://arxiv.org/abs/math/0410072]. 
   - Why try: excellent ranking/detection performance in sparse-weak regimes — pairs well with an FDR-style calibration (e.g., use HC to rank and BH or knockoffs to choose absolute cut).

Notes on implementation choices: 1) Since you have a well-fitting χ-null for inactive blocks, the simplest, highest-confidence starting point is to compute per-block p-values from that χ-null and then apply an FDR procedure (BH or Storey) or knockoffs if Model‑X is available; 2) if covariate/test-shift is a concern, incorporate weighted conformal calibration or likelihood-ratio weighting into your calibration; 3) participation-ratio (Rényi‑2) is an excellent ranking statistic but requires null calibration (e.g., translate its value into a p-value under the χ-null or use subsampling/perm tests) before thresholding at an α-level.


# Prior‑art verdict and adversarial / disconfirmation findings

- Prior‑art verdict:
  - The general set of tools you need exists across multiple literatures (multiple testing and knockoffs for false‑discovery control; conformal prediction for finite-sample coverage under exchangeability; compressed sensing/stability selection for support recovery guarantees) [Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf) [Storey 2002](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf) [Meinshausen & Bühlmann 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x) [Lei etal. 2018](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf) [Tibshirani etal. 2020](https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf). 
  - Direct prior-art that applies a null-referenced, per-input α-controlled threshold specifically to SAE / latent dictionary codes is very limited: one explicit arXiv contribution applies Model‑X knockoffs to SAE features and reports finite-sample FDR control in that setting [Enkhbayar 2025](https://arxiv.org/html/2511.11711v1). Outside that, SAE/interpretability papers typically use fixed‑k or learned global thresholds rather than per-input α-style thresholds [BatchTopK; Huben etal.]. Thus your exact framing is largely novel for SAE/dictionary latents, with the notable exception above.

- Disconfirmations / known failure conditions (when null-referenced α-style methods break):
  1. Dependence / correlated tests can make BH anti‑conservative — BH’s guarantee is exact for independent tests and extends only to certain dependence classes; arbitrary dependence breaks nominal FDR control [Benjamini & Hochberg 1995; literature on BH under dependence]. For interpretability latents that are strongly correlated, naive BH on per-block p-values may under‑control FDR [Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf).
  2. Model‑X knockoffs require accurate knowledge (or simulation ability) of the covariate / latent distribution; mis-specifying Model‑X or constructing invalid knockoffs can invalidate the FDR guarantee [knockoffs literature; Enkhbayar 2025](https://arxiv.org/html/2511.11711v1). 
  3. Conformal prediction relies on exchangeability (or correct reweighting for covariate shift). Under non-exchangeability or when likelihood ratios are poorly estimated, coverage guarantees can fail [Lei etal. 2018; Tibshirani etal. 2020] [https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf] [https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf]. Recent work documents corrections and carefully bounded relaxations but not universal immunity to arbitrary shift [weighted conformal literature].
  4. Universal thresholding / VisuShrink assumes Gaussian noise; heavy-tailed or strongly non-Gaussian noise breaks the optimality/high-probability guarantees and requires robustified thresholds or moment conditions [Donoho & Johnstone 1994; robustness analyses of wavelet thresholding].
  5. Stability Selection’s error bounds assume a degree of stability in the base selector and do not guarantee robustness if the underlying selection procedure becomes unstable under changing noise/distribution — trimmed-stability and robust variants mitigate but do not eliminate failures [Meinshausen & Bühlmann 2010; trimmed-stability work].

  All of these failure modes have been documented in the literature and must be considered adversarially when selecting a method for your application [examples: BH dependence warnings; conformal under non-exchangeability; universal threshold sensitivity to non-Gaussian noise; stability selection caveats] [Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf) [https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf] [https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf] [https://doi.org/10.1111/j.1467-9868.2010.00740.x].


# How each evidence frontier maps onto your empirical findings and why they predict them

- Classical multiple-testing & shrinkage (BH, Storey, VisuShrink, SURE): explains and supports (1) χ-null for inactive items and (5) failure of knee-gap heuristics in continuous decay regimes; VisuShrink/SURE motivate using a χ-derived analytic threshold and continuous shrinkage rather than hard knee detection [Donoho & Johnstone 1994](https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf). BH/Storey explain how to convert χ p-values into an α‑calibrated variable-size selection [Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf) [Storey 2002](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf).

- Conformal prediction: maps onto the desideratum of a single α with finite-sample coverage transferable across inputs; weighted conformal directly addresses covariate shift (so it maps to the need for cross-sample transfer when distributions move) [Lei etal. 2018; Tibshirani etal. 2020] [https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf] [https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf]. However, conformal methods do not directly produce a subset of latent features unless you define an appropriate nonconformity and treat each block as a “label” candidate — doing so is conceptually possible but requires careful design and probably weighting.

- Compressed sensing / Higher Criticism: explains (3) participation-ratio / collision probability’s strong ranking power (Rényi‑2 approximates effective support) and identifies regimes where HC is asymptotically optimal for sparse detection, explaining why HC-style measures rank support well but may need calibration for absolute selection size [Donoho & Jin 2004] [https://arxiv.org/abs/math/0410072]. CS literature also clarifies that selection guarantees (and transfer across sparsities) require sensing/matrix conditions (RIP/incoherence) that may or may not hold for learned dictionary encoders [Candès etal. 2006](https://doi.org/10.1109/TIT.2005.862083).

- Neural-interpretability / SAE work: explains why max-referenced rules fail in practice (the largest latent can be an artifact) and demonstrates that most SAE methods use fixed-k, batch constraints or learned thresholds rather than per-input null α; the lone exception explicitly applying Model‑X knockoffs shows the direction to get finite-sample α control in SAEs [BatchTopK; Huben etal.; Enkhbayar 2025] [https://arxiv.org/html/2511.11711v1].


# Practical recommended recipe (operationalized, prioritized)

Given your empirical facts (χ-null fit extremely tight; participation-ratio ranks best but is scale‑miscalibrated; max is unreliable; knee fails) the highest-probability, theoretically principled pipeline is:

1. Use the χ (df = block_dim) null to compute per-block two-sided or one-sided p-values for inactivity per input (directly justified by your KS fit). Cite χ theory as basis [chi theory].
   - Why: uses your strongest empirical fact — a well-specified noise null is the central enabler of dimensionless α control [Donoho & Johnstone 1994] [https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf].

2. Primary selection option: apply Model‑X knockoff+ to SAE latents when you can sample/estimate the latent joint distribution well. Use knockoff+ selection with target FDR q (your α). This gives finite-sample FDR control per instance/experiment under Model‑X [Enkhbayar 2025](https://arxiv.org/html/2511.11711v1). 

3. If Model‑X is not plausible: use BH or Storey q-values on χ p-values per input (or per-batch if per-input m is small). Start with BH at chosen α, but be mindful of latent dependence; if latents show correlation, combine BH with stability selection or use the empirical null estimation (Storey) to improve calibration [Benjamini & Hochberg 1995] [Storey 2002] [Meinshausen & Bühlmann 2010].

4. If covariate shift / domain change across inputs is expected: adopt weighted conformal calibration or likelihood-ratio weighting when calibrating thresholds (requires estimation of test/training covariate density ratio) to retain a 1−α coverage guarantee under shift [Tibshirani etal. 2020] [https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf]. 

5. Use participation-ratio / Rényi‑2 as a ranking statistic (to order inputs by likely support) and then apply one of the α-calibrated rules above to convert ranking into absolute selection, or use HC to detect the best operating region then apply FDR on the corresponding candidates [Donoho & Jin 2004] [https://arxiv.org/abs/math/0410072].

6. For robustness to dependence and heavy-tailed noise: combine subsampling/stability selection with BH or use trimmed-stability variants; for heavy tails, robustify the χ-null estimate (e.g., use robust σ̂, Winsorization/trimmed statistics, or median‑based estimators) before computing p-values or thresholds [Meinshausen & Bühlmann 2010; robust wavelet thresholding literature].


# Concrete testing checklist (what to measure when you compare rules)

- Per-input calibration: false-positive fraction (FP / selected), measured over held-out inputs with known ground-truth support. For FDR methods measure realized FDP and its expectation.
- Rank correlation: Pearson / Spearman between rule score (or size) and true support count (as you already did for participation-ratio). 
- Robustness to shift: simulate covariate shift (e.g., change input distribution or introduce positional artifacts) and measure change in realized FDP and power for each α. 
- Dependence sensitivity: measure realized FDP when you introduce latent correlation structures (e.g., block correlation) to assess BH anti‑conservatism.
- Sensitivity to null mis-specification: inject heavy-tailed or non-Gaussian noise in inactive blocks and measure calibration loss for χ-based p-values and VisuShrink thresholds.


# Explicit cautions and limitations (evidence-based)

- If latent-block dependence is strong and unknown, BH may be anti‑conservative; prefer knockoffs or stability selection variants when dependence cannot be trusted [Benjamini & Hochberg 1995; Meinshausen & Bühlmann 2010].
- Model‑X knockoffs require accurate modeling/simulation of the latent joint; if you cannot fit a Model‑X surrogate, knockoffs’ finite-sample guarantee does not hold [knockoffs literature; Enkhbayar 2025].
- Conformal guarantees hinge on exchangeability or accurate likelihood-ratio weighting; poor estimation of weights or non-exchangeability breaks guarantees [Lei etal. 2018; Tibshirani etal. 2020].
- Universal thresholds (VisuShrink) assume Gaussian noise; heavy-tailed or skewed nulls require robustified noise estimation or alternative thresholds [Donoho & Johnstone 1994].


# Final prescriptive summary (single-paragraph operational answer you can act on now)

Given your strong empirical χ-null for inactive block norms, the single most principled and transferable approach is to (A) compute per-block p-values from that χ-null, (B) apply a multiple-testing FDR control method that admits a finite‑sample / robust dependence treatment — ideally Model‑X knockoffs if you can simulate the latent joint (gives finite‑sample FDR ≤ α) otherwise BH/Storey combined with stability selection (expected FDR control under weaker assumptions), and (C) use weighted conformal calibration when covariate/test distribution shift is present and you can estimate density ratios to maintain the same α interpretation across inputs [Enkhbayar 2025; Benjamini & Hochberg 1995; Storey 2002; Tibshirani etal. 2020] [Enkhbayar 2025](https://arxiv.org/html/2511.11711v1) [https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf] [https://genomics.princeton.edu/storeylab/papers/directfdr.pdf] [https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf].


# Sources cited inline (selected canonical sources used above)
- [Controlling the False Discovery Rate: Benjamini & Hochberg 1995](https://www.stat.purdue.edu/~doerge/BIOINFORM.D/FALL06/Benjamini%20and%20Y%20FDR.pdf)
- [A Direct Approach to False Discovery Rates: Storey 2002](https://genomics.princeton.edu/storeylab/papers/directfdr.pdf)
- [Higher Criticism for Detecting Sparse Heterogeneous Mixtures: Donoho & Jin 2004](https://arxiv.org/abs/math/0410072)
- [Adapting to Unknown Smoothness via Wavelet Shrinkage: Donoho & Johnstone 1994](https://web.stanford.edu/dept/statistics/cgi-bin/donoho/wp-content/uploads/2018/08/denoiserelease3.pdf)
- [Distribution-Free Predictive Inference for Regression (Split Conformal): Lei etal. 2018](https://www.stat.cmu.edu/~ryantibs/papers/conformal.pdf)
- [Conformal Prediction Under Covariate Shift (Weighted Conformal): Tibshirani etal. 2020](https://www.stat.cmu.edu/~ryantibs/papers/weightedcp.pdf)
- [Stability Selection: Meinshausen & Bühlmann 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x)
- [Observed Universality of Phase Transitions in High-Dimensional Geometry: Donoho & Tanner 2009](https://doi.org/10.1098/rsta.2009.0157)
- [Participation‑ratio / Rényi‑2 / effective support references and χ-null theory: classical χ/χ² theory explanatory note](https://pmc.ncbi.nlm.nih.gov/articles/PMC3900058/)
- [SAE application with Model-X knockoffs: Enkhbayar (2025) "Which Sparse Autoencoder Features Are Real? Model‑X Knockoffs for False Discovery Rate Control" (arXiv)](https://arxiv.org/html/2511.11711v1)
- [Nucleus (top‑p) sampling contrast: Holtzman etal. 2019 "The Curious Case of Neural Text Degeneration" (nucleus sampling)](https://arxiv.org/abs/1904.09751)

