# Test B'' — within-rabbit complexity-split transfer: ADAPTIVE WINS (2026-07-11)

The clean isolation Test B lacked. SAME rabbit dictionary reconstructs both halves
equally well (zero domain shift), so any transfer gap is the SELECTION RULE's, not
the dictionary's. Split rabbit patches by k* median → LOW (mean k*=4.22) vs HIGH
(mean k*=6.84). Calibrate each rule's knob on one half, FREEZE, apply to the other.

## Result: adaptive transfers the complexity gap, fixed-k cannot

**CALIB_LOW → TEST_HIGH** (tune on simple patches, test on complex; the harder direction):
| rule (frozen from LOW) | selected on HIGH | HIGH k*=6.84 | corr | err |
|---|---|---|---|---|
| fixed_k=4 | 4.00 (stuck) | 6.84 | 0.00 | 2.84 |
| **BH α=0.005** | **5.54** (auto-widened) | 6.84 | 0.05 | **1.61** |
| chi q=3.01 | 5.42 (auto-widened) | 6.84 | 0.03 | 1.62 |

**CALIB_HIGH → TEST_LOW** (tune on complex, test on simple):
| rule (frozen from HIGH) | selected on LOW | LOW k*=4.22 | corr | err |
|---|---|---|---|---|
| fixed_k=7 | 7.00 (stuck, over-selects) | 4.22 | 0.00 | 2.78 |
| **BH α=0.050** | **4.86** (auto-narrowed) | 4.22 | **0.595** | **0.91** |
| chi q=2.55 | 5.01 (auto-narrowed) | 4.22 | 0.599 | 0.97 |

## The read

**Adaptive rules cut error ~40-67% vs fixed-k across the complexity gap, and they move
the RIGHT way with a FROZEN knob:**
- fixed_k is stuck at its calibration count (4 or 7) and is wrong by ~2.8 blocks on the
  other half — it CANNOT adapt, by construction.
- BH/chi, with knob frozen, **auto-adjust the count toward the test half's true support**:
  calibrated on LOW (simple) they widen 4→5.5 on HIGH; calibrated on HIGH they narrow
  7→4.9 on LOW. The α/q knob transfers; the count follows the data.
- This is exactly the fixed-k kill-zone the thesis predicted, now demonstrated cleanly
  with no dictionary confound.

## Honest caveats
1. **corr is weak in the LOW→HIGH direction** (0.05) though error is much better. Reason:
  within the HIGH half alone, remaining k* variance is compressed (we already conditioned
  on high complexity), so rank-correlation has little range to work with — the WIN there
  is the error/mean-count adaptation, not ranking. In HIGH→LOW, corr is strong (0.60)
  AND error low — the fuller win.
2. **BH ≈ chi-floor again** — at matched calibration they track together. FDR's advantage
  over chi remains the PRINCIPLED KNOB (α=target FDR, dimensionless) not raw performance;
  both beat fixed-k on transfer.
3. **realized-FDR proxy reads 0.0** — the proxy (fraction of selected blocks with p>0.5)
  is too crude; on these tight-contrast rabbit patches selected blocks all have tiny p, so
  the proxy saturates at 0. Need a better realized-FDR estimator (e.g. inject known-null
  synthetic blocks and count how many get selected) to actually verify α's calibration —
  that's a real TODO, the current number is uninformative not meaningful.

## Verdict
Test B'' gives the FIRST clean evidence that distribution-aware selection TRANSFERS across
a complexity gap where fixed-k structurally can't (error −40-67%, knob frozen, count
auto-adapts correctly in both directions). Combined with FDR-v1's principled α-knob, the
thesis now has: (1) in-distribution competitiveness, (2) a non-arbitrary transferable knob,
(3) clean cross-complexity transfer within a competent dictionary. What's STILL unproven:
transfer across a genuine DOMAIN shift with a competent dictionary (needs B''' = broad-
trained dictionary), and actual α-calibration verification (needs a real realized-FDR test,
not the saturated proxy).
