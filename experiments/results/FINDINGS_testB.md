# Test B — cross-distribution transfer: FAILED (and confounded) (2026-07-11)

Froze rabbit-calibrated knobs (fixed_k=5, BH α=0.10, chi-floor q=2.71), applied the
rabbit-trained dictionary to Imagenette (10 diverse ImageNet classes, 300 imgs, same
DINOv3 pipeline). Hypothesis: α holds its meaning while fixed-k's error swings.

## Result: hypothesis NOT supported — everything failed, adaptive rules WORST

| rule (frozen rabbit knob) | mean selected | Imagenette k*=10.12 | corr w/ k* | total err |
|---|---|---|---|---|
| fixed_k=5 | 5.00 | (needs 10) | 0.00 | 8.21 |
| BH α=0.10 | **2.14** | (needs 10) | 0.14 | 8.54 |
| chi-floor q=2.71 | **1.48** | (needs 10) | 0.14 | 8.83 |

- Imagenette k*: **mean 10.12, CV 1.41** (rabbit was 5.41 / 0.31) — nearly 2× the true
  support and 4.5× the complexity variance. Diverse classes = far more concepts/patch.
- **The adaptive rules moved the WRONG WAY:** true support went UP (5.4→10.1) but BH/chi
  selected FEWER (5.4→2.1, 1.5). α=0.10 produced 2 blocks when 10 were needed. corr
  collapsed 0.62→0.14. This is the opposite of the transfer claim.

## Why it failed — the design is CONFOUNDED (my error, flagging it)

Test B as run conflates TWO different transfers:
1. **Does the SELECTION RULE transfer?** (what we wanted to test)
2. **Does the DICTIONARY transfer?** (a rabbit-TRAINED dictionary applied to Imagenette)

The dictionary was trained only on rabbits. Applied to Imagenette it reconstructs
poorly and its block norms don't carry the same meaning — so the χ₃ noise null MEASURED
on rabbits no longer describes the inactive-block distribution on Imagenette. The
per-patch σ̂ (40th-pctile / χ₃ inversion) is then mis-estimated, the p-values are
computed against a wrong null, and almost nothing clears threshold → severe
under-selection. The CV=1.41 blowout is the tell: the rabbit dictionary fits some
Imagenette patches (rabbit-like texture) and badly fails others, producing wildly
variable reconstruction — dictionary misfit, not selection behavior.

**So "everything fails" here is dominated by DICTIONARY domain-shift swamping any signal
about whether the α-knob transfers.** The test doesn't actually answer the question it
was built for. Honest null + confound, not a clean disconfirmation of the FDR thesis.

## The instructive part
The confound IS the lesson: to test selection-rule transfer you must hold dictionary
competence roughly constant across both distributions. The rabbit-only dictionary can't
do that. Two clean redesigns:
- **B′ (isolate rule transfer):** train the dictionary on a BROAD set (e.g. Imagenette
  itself, or rabbits+Imagenette), calibrate α on one held-out subset, test on another
  subset of similar dictionary-competence. Now dictionary fit is ~constant and only the
  rule's knob is under test.
- **B″ (in-distribution split, cheapest honest version):** even within rabbits, split
  patches into low- vs high-complexity halves, calibrate α on one half, test realized
  FDR on the other. Pure rule-transfer, zero dictionary shift.

## Verdict
FDR-v1's in-distribution result (untuned α=0.10 → corr 0.62, principled knob) STANDS —
Test B doesn't touch it. But the cross-distribution transfer claim is UNPROVEN: this
attempt was confounded by dictionary domain-shift and must be redone with a
distribution-competent dictionary (B′) or an in-distribution complexity split (B″)
before any "α transfers across shift" claim can be made. Do NOT claim transfer yet.
