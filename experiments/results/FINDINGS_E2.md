# E2 robustness results — the robustness win goes to top-k, not chi (2026-07-09)

**The reframe tested:** does chi-floor give operating-point robustness (one setting
covers the sparsity-budget range) the way top-nσ gives temperature robustness? Plot:
`robustness_curves.png`. Table: `train_eval_matrix.json`.

## Result: NO. And the most robust single model is top-k (k=8).

R² when each trained model is eval-swept to mean-L0 ∈ {4,8,16} (native envelope = best-in-column):

| trained model | L0=4 | L0=8 | L0=16 | max sag below envelope |
|---|---|---|---|---|
| **topk_k8** | 0.752 | **0.812** | **0.803** | **only −0.046** (at L0=4) |
| topk_k16 | 0.692 | 0.803 | 0.849 | −0.077 (at L0=4) |
| topk_k4 | 0.769 | 0.751 | 0.649 | −0.200 (at L0=16) |
| chi_q2.5 | 0.770 | 0.762 | 0.697 | −0.152 (at L0=16) |
| chi_q3.5 | 0.623 | 0.623 | 0.673 | −0.176 (collapsed, 229 dead) |
| _envelope_ | 0.769 | 0.812 | 0.849 | — |

**topk_k8 is the flattest, highest single curve** — it hits the envelope at its home
point and sags only ~0.05 when swept, INCLUDING sweeping UP to L0=16 (0.803 vs 0.849).
No chi model tracks the envelope as well. chi_q2.5 ties top-k at low L0 but sags hard
climbing to L0=16; chi_q3.5 partially collapsed in training (229/256 dead blocks).

## Why the top-nσ analogy breaks (the honest mechanism)

The temperature-invariance analogy had a hidden disanalogy I under-weighted:
- In sampling, temperature is an **eval-time** knob applied to a FIXED model. top-nσ's
  win is being invariant to that eval-time transform. Genuinely one model, many temps.
- Here, the "operating point" (sparsity budget) is baked into WHAT THE DICTIONARY LEARNS.
  A budget-8 dictionary and a budget-16 dictionary are *different learned objects*. There
  is no single fixed model that all budgets are eval-time transforms of — so there is no
  invariance for chi-floor to exploit. The analogy needed the operating condition to be
  eval-time; sparsity budget is train-time. That's the crux.

And empirically top-k turns out to ALREADY be robust to eval-time k-sweeping (topk_k8
holds 0.80+ from L0=8 to L0=16), because extra blocks are trained-and-ranked; it only
sags sweeping *below* train-k. So top-k had the robustness property we hoped chi would
provide. Nothing left for chi to win on this axis.

## Standing tally after two clean tests

1. E1 (frozen top-k substrate): all adaptive lose; perpatch_chi ties (−0.003), min_z
   dies (−0.03). Confirmed noise-ref ≫ max-ref, but biased toward top-k.
2. E2 (co-adapted training + robustness): top-k(8) is the most robust single model;
   chi gives no operating-point-robustness advantage. Reframe fails.

**Honest headline forming:** on THIS problem (DINOv3 vision patches, 256 blocks,
recon-R² objective), distribution-aware block selection is at best a reparameterization
of the sparsity knob and does not beat fixed top-k on either raw R² or operating-point
robustness. That is a clean, reportable negative.

## The important caveat — we've only tested the WRONG metric for the thesis

R² is a RECONSTRUCTION metric. The whole thesis is INTERPRETABILITY (monosemanticity,
attribution). The theory always said the R² win would be marginal and the *real* payoff
is mechanism 2 — chi-floor not forcing blocks to fire on the sparse tail => cleaner
monosemantic dictionaries. We have NOT measured that. Two negatives on R²/robustness do
NOT touch the monosemanticity claim. The live question is whether chi-floor produces more
monosemantic / less-polluted features at matched R², which needs a concept-purity metric
(e.g. per-block firing consistency, or the paper's manifold-quality measures), not R².

## Next-decision fork
- (a) pivot the metric to monosemanticity/concept-purity — the thesis-aligned test we
  haven't run; the only place a win is still theoretically expected; OR
- (b) call distribution-aware *selection* a negative on vision-BSF and move to idea #2
  (BSF-on-attention), carrying forward the clean "noise-ref>>max-ref + χ₃-floor-is-real"
  measurement as the salvageable insight.
