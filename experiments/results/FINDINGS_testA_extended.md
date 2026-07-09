# Test A-extended — p-less / DiffSampling / Top-H vs k* (2026-07-10)

Same frozen topk_k8 model + peak-based k* (mean 5.4, CV 0.308) as Test A. Which
shape-estimator best predicts per-patch TRUE support? corr(selected_count, k*):

| selector | mean sel | std sel | **corr w/ k*** | total err | note |
|---|---|---|---|---|---|
| **p_less_L1** (norm shares) | 14.0 | 9.13 | **+0.695** ✅ BEST | 8.58 | highest corr; but over-selects 2.6× |
| perpatch_chi (prior) | ~8 | — | +0.576 | 0.99 | best error/corr balance |
| min_z (prior) | — | — | +0.517 | — | |
| p_less_E (energy shares) | 2.24 | 1.26 | +0.400 | 3.17 | under-selects |
| global_floor (prior) | — | — | +0.331 | — | |
| fixed_k=5 | 5.0 | 0 | 0.000 | 1.26 | flat by construction |
| **diffsample** | 1.66 | 1.01 | **+0.044** ❌ | 3.75 | ~random — cliff assumption fails |
| **top_h** (α=0.27) | 5.41 | 0.65 | **−0.528** ❌❌ | 1.54 | ANTI-correlated! |

## Reads

**p_less_L1 has the HIGHEST correlation with true support of anything tested (0.695 > chi's 0.576).**
The participation-ratio intuition was right: `1/Σ(share²)` = effective # of active
blocks, and it tracks k* better than any other estimator. BUT it over-selects
badly (mean 14 vs k*=5.4) → huge total error (8.58). So the *ranking* it induces
over patches is excellent (best corr), but its *absolute threshold* is mis-scaled
(too permissive). This is the classic "great signal, wrong offset" — fixable by
composing p-less's SHAPE with a scale correction (e.g. use p-less to set the
relative count, calibrate the absolute level separately). **Most promising new
signal in the whole study**, but not usable as-is.

**p_less_E (energy shares) is the opposite** — under-selects (2.24), lower corr
(0.40). Squaring the shares over-concentrates on the top blocks, so the effective
count collapses toward the dominant block — the dominant-block leakage concern from
the theory, realized. L1 shares are the right version, not energy shares.

**DiffSampling: corr 0.044 ≈ RANDOM. Our data killed it exactly as predicted.**
Test A already showed the sorted profile is a smooth continuum (rank8/rank9 ratio
1.048, no knee). With no cliff, the largest *absolute* gap sits at the very top
(steepest near rank 1-2), so it cuts to ~1.7 blocks regardless of patch complexity
→ zero correlation with k*. This also ANSWERS the open per-patch-cliff question:
individual patches don't have sharp knees at varying ranks either (if they did,
diffsample would correlate). The smoothness is real at the per-patch level, not a
median-smearing artifact. **DiffSampling is dead for this problem — and now we know
WHY (no cliffs exist, per-patch or aggregate).**

**Top-H: corr −0.528, ANTI-correlated (the biggest surprise).**
Entropy-saturation selects FEWER blocks on high-k* patches — backwards. Mechanism:
complex patches (high k*) spread energy across many blocks → the renormalized
top-set reaches a given entropy fraction FAST (each added block is similar
magnitude → entropy climbs quickly) → α·H threshold trips early → few blocks kept.
Simple patches (low k*) have one dominant block → entropy climbs slowly as you add
→ keeps more. So entropy-saturation is inversely related to support here. Entropy
of a norm distribution is NOT a proxy for concept count in BSF the way it is for
token distributions — a genuine disanalogy worth writing down.

## Verdict / next
- **p_less_L1 is the standout**: best support-tracking signal found (corr 0.695).
  Actionable next step = decouple its shape from its scale (use the participation-
  ratio ranking, fix the absolute threshold via a calibration constant or by
  matching mean-L0). If a scale-corrected p-less holds corr ~0.7 at matched mean,
  it beats chi-floor as the support estimator.
- chi_floor remains best error/corr BALANCE as-is (0.576 corr, 0.99 err).
- DiffSampling & Top-H: both empirically dead here, each for a clean, documented
  reason (no cliffs; entropy anti-tracks support). Negative results worth keeping —
  they rule out two whole reference-point families for BSF.
