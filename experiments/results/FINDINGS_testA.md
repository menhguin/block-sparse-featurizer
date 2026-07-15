# Test A — selection-correctness vs per-patch true support (2026-07-09) — ADAPTIVE WINS

**The reframe that produced it:** E1/E2 tested in-distribution R²-*average*,
which averages over the complexity distribution and hides tail errors. top-nσ's
real win was matching per-input structure. Test A measures selection *correctness*
against a rule-agnostic per-patch "true support" k*.

## Critical setup fix (was a broken ruler first pass)
The BSF decoder is **overcomplete + non-orthogonal** (768×768 = 256 blocks × 3).
Per-patch R² **PEAKS (~k=10) then DECLINES** as low-norm blocks inject garbage
(patch0: R²=0.86@k8 → 0.22@k256). First-pass k* was defined vs full-256 R² (which
is *bad* recon) → k* collapsed to ~1. FIXED: k* = smallest count reaching 95% of
each patch's OWN ACHIEVABLE PEAK R². This non-orthogonality is itself the reason
flooding-in blocks (min_z low-p) tanked in E1 — now quantified.

## Ground truth
peak_r2 mean=0.814, peak_k mean=10.5. **k*: mean=5.4, median=5, p5=3, p95=8, CV=0.308.**
Real per-patch support variance (3→8 blocks across the 5th–95th pct).

## Result: at matched mean-selected (≈5.4), adaptive tracks k*, fixed-k cannot

| selector | corr(selected, k*) | mean over-select | mean truncate | total error |
|---|---|---|---|---|
| fixed_k=5 | **0** (flat by construction) | 0.42 | 0.84 | **1.26** |
| **chi_floor (q=2.71)** | **0.576** | 0.49 | 0.49 | **0.99** ✅ best |
| min_z (p=0.28) | 0.517 | 0.91 | 0.91 | 1.82 |
| global_floor | 0.331 | — | — | 1.30 |

**Plot (`testA_selection_vs_support.png`):** fixed_k is a dead-flat horizontal line
at y=5 — ignores k* entirely. chi_floor and min_z RISE with k* (track the diagonal).
min_z tracks the diagonal slope closest but overshoots noisily (worst total error);
chi_floor has the **lowest total selection error (0.99 vs fixed-k's 1.26, −21%)** —
the best balance of tracking + stability.

## The tercile story (this is the mechanism, made visible)

fixed_k=5's errors are ENTIRELY on the tails, zero in the middle — textbook
misspecification:
- **simple** tercile (k*=3.5): fixed-k over-selects by **+1.47 blocks** (forces 5
  when 3.5 needed → admits ~1.5 noise blocks). chi over-selects only +0.7.
- **medium** (k*=5.0): fixed-k perfect (0 error) — because k=5 was tuned to the mean.
- **complex** (k*=6.8): fixed-k truncates by **−1.84 blocks** (clips at 5, drops ~2
  real concepts). chi truncates only −0.9.

chi_floor spreads small errors evenly (0.7/0.53/0.34 over-select; 0.08/0.23/0.9
truncate) instead of concentrating them on the tails. **This is EXACTLY the top-nσ
mechanism: fixed-count is structurally wrong on both tails; noise-referenced
selection is complexity-aware by construction.**

## Reconciliation with the E1/E2/mono negatives (no contradiction)

All results are consistent and the synthesis is the actual finding:
- Fixed-k's tail errors **cancel in the aggregate R²-average** (over-selection on
  simple patches barely hurts R²; truncation on complex ones costs a little) → E1/E2
  saw ~parity. The mean hides the tails.
- The errors DON'T cancel in **selection correctness** (Test A) → adaptive clearly
  better at matching true support.
- So the honest headline: **distribution-aware selection doesn't improve the
  training-distribution R²-average, but it IS materially more complexity-aware
  (−21% selection error, tracks true support where fixed-k is flat). The value is a
  GENERALIZATION/robustness property, not an in-distribution reconstruction win —
  precisely as it was for top-nσ (temperature robustness, not higher greedy acc).**

## Caveats (honest)
- k* built from norm-ranked greedy addition → mildly sympathetic to norm-based
  selection generally, but NEUTRAL for fixed-vs-adaptive (both rank by norm). Stated.
- This is still a within-rabbit measurement. The strongest generalization claim
  needs Test B (cross-distribution transfer): does one chi setting hold across
  input domains where fixed-k needs re-tuning? Test A shows the PRECONDITION
  (adaptive tracks per-input complexity); B would show it TRANSFERS.
- corr NaN for fixed_k is expected (zero variance in a constant).

## Verdict update on idea #1
NOT a flat negative. Refined: **on vision-BSF, distribution-aware selection is a
selection-correctness / complexity-robustness improvement (chi-floor −21% total
selection error, tracks true support tercile-by-tercile) that does NOT show up in
in-distribution aggregate R². chi-floor > min_z (lowest error + stable). The
in-distribution-R² framing was measuring the wrong axis; the pool-variance /
robustness reframe found the right one.** Next: Test B (cross-distribution transfer).
