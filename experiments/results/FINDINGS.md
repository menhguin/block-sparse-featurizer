# Findings — "default shape" of BSF block-norm distributions (2026-07-09)

Substrates (256 blocks, group_size=3, DINOv3-ViT-B/16 via timm, 300 rabbit imgs, 58,800 patches):

| substrate | R² | mean L0 | dead | inactive-norm mean | active-norm mean | chi₃ fit KS-D | rank8/rank9 ratio |
|---|---|---|---|---|---|---|---|
| vanilla_topk    | 0.813 | 8.0  | 0  | 2.21 | 10.88 | **0.0019** (excellent) | 1.048 |
| grass_topk      | 0.787 | 8.0  | 0  | 1.71 | 6.92  | 0.013 (very good)      | 1.048 |
| grouplasso_soft | 0.597 | 3.83 | 50 | 2.20 (bimodal) | 9.37 | 0.093 (poor) | 1.026 |

Plot: `block_norm_distributions.png` (rows=substrate; A=global log-y hist, B=active-vs-inactive, C=inactive vs chi₃ fit, D=sorted profile rank≤40).

## The three headline reads

**1. NOT dramatically bimodal at the global level (Minh's doubt: directionally right).**
Panel A for all three = one big spike near 0-3 + a smooth heavy right tail. No screaming valley. The small-vocab intuition held: 256 blocks don't produce the clean two-hump separation a 128k-vocab logit distribution does. min-z (max-anchored, *needs* a sharp knee/gap) has a **weak** case here.

**2. BUT the noise floor is textbook chi₃ — the chi-floor precondition is met beautifully.**
On both top-k substrates the *inactive* block norms fit a chi distribution with df=group_size=3 almost perfectly (**KS-D=0.0019** for vanilla — that is a near-perfect fit). This is the single most important result: it means "block active" can be posed as a real hypothesis test against a well-specified null, with a genuine false-positive-rate (α) interpretation. **The chi-floor variant has strong empirical support; min-z does not.** This is exactly the asymmetry the theory file predicted — BSF training *manufactures* the clean noise floor that the sampling paper couldn't lean on.

**3. k=8 is not a natural boundary — it cuts a smooth continuum (the quantitative kill-shot).**
The sorted-profile knee (Panel D) is *soft*, sitting around rank 5-10, and **rank-8 and rank-9 block norms differ by only ~5%** (ratio 1.048). There is no cliff at 8. Meanwhile the noise floor (inactive mean) is ~2.2 but the sorted profile is still ~3-4 out at rank 20+ — i.e. **there are real, above-noise-floor learned concepts (0 dead blocks → every block is a trained feature) that fixed-k=8 discards on rich patches.** Fixed k both truncates real signal on rich patches and (symmetrically) over-selects into the chi noise on simple ones.

## What this means for the method

- The experiment **cleanly discriminates our two variants**: chi-floor (noise-referenced) = principled + empirically supported; min-z (max-referenced) = weak, no sharp gap to snap to.
- **grouplasso_soft is the cautionary foil**: it's the only one that *manufactures* visible bimodality (Panel B gray shows a shrunk-to-~0.5 spike + a ~2.5 hump) — but at a large R² cost (0.60 vs 0.81) and its inactive pop no longer fits a single chi (KS-D=0.093). Soft-thresholding buys a gap by paying reconstruction; it is not free structure.

## Important caveat (honest)

This is the **precondition** for a win, not the win. It shows a real, characterizable boundary that fixed-k doesn't track — it does **not** yet show that exploiting it improves R² at matched mean-L0. That is the E1 experiment (apply top-k vs min-z vs chi-floor to frozen magnitudes, sweep each, overlay the R²-vs-mean-L0 Pareto). The chi₃-noise-floor result makes chi-floor the odds-on variant to test first.

Also: numbers are on the timm DINOv3 mirror (identical weights, non-gated) pending Meta's gated-repo approval; shape conclusions are robust to that, publication-grade numbers should be re-run on the facebook repo.
