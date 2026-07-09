# Fork (a) — monosemanticity: chi-floor vs top-k at matched L0 (2026-07-09)

Tests MECHANISM 2 (the thesis-aligned claim R² can't see): does chi-floor produce
more monosemantic / less-polluted features than fixed top-k? Matched at L0≈12
(chi_q2.0 native L0=11.9; trained topk_k12 to match — see budget-instability note).

## Matched result (both L0≈12, both 256/256 alive)

| metric | topk_k12 | chi_q2.0 | who wins | margin |
|---|---|---|---|---|
| R² | 0.834 | 0.836 | chi | +0.003 (noise) |
| M1 passenger-firing rate ↓ | 0.361 | 0.340 | chi | −0.021 |
| M2 median fire-freq ↓ (selectivity) | 0.0227 | 0.0205 | chi | −0.002 |
| M3 top-fire input coherence ↑ (money metric) | 0.3797 | 0.3839 | chi | +0.004 |
| random-coherence floor | 0.0005 | 0.0005 | — | (both crush it) |

**chi wins all four — but every margin is within noise.** M3 coherence edge is
+0.004 on a 0.38 base (~1%); passenger rate is the largest signal (chi ~2pp fewer
passenger firings, weakly in mechanism-2's predicted direction). This is a
whisper, not a result. It does NOT support a "distribution-aware selection makes
materially more monosemantic features" claim on vision-BSF.

## Unmatched run (chi L0=11.9 vs topk_k8, L0=8) — for the record
chi coherence 0.384 vs 0.379, passenger 0.34 vs 0.24 (top-k better — but top-k
had fewer active blocks so fewer passengers; unmatched, discard the passenger
comparison). Superseded by the matched run above.

## Extra finding: chi-floor CANNOT reliably target a sparsity budget
q→L0 map is chaotic under co-adapted training: q2.0→11.9, q2.3→4.4, q2.6→4.6,
q2.9→18.9; dead-block counts swing 2→46→145. The hard per-patch threshold and the
learned magnitudes co-evolve in an unstable feedback loop. This is a real
PRACTICAL DOWNSIDE vs top-k (exact-k by construction): tuning chi to a target
sparsity is fiddly and non-monotone. Worth stating plainly — it's the kind of
thing that sinks a method in practice independent of the metric outcome.

## VERDICT on idea #1 (distribution-aware block selection for vision-BSF)

Three independent tests, one consistent story:
1. E1 (frozen R²): adaptive loses on top-k's turf; noise-ref(chi) ≫ max-ref(min_z).
2. E2 (co-adapted R² + robustness): top-k(8) is the most robust single model;
   chi gives no operating-point-robustness advantage. temp-invariance analogy
   breaks (sparsity budget is train-time, not eval-time).
3. Fork-a (monosemanticity, matched L0): chi wins all 4 metrics but every margin
   is within noise; plus chi has an unstable budget-targeting downside.

**Bottom line: on vision-BSF (DINOv3 patches, 256 blocks), distribution-aware
block selection is a reparameterization of the sparsity knob — not a win on
reconstruction, robustness, OR monosemanticity.** Clean, honest negative.

## Salvageable insights (hold regardless of the negative)
- **noise-referenced ≫ max-referenced**, confirmed 10× on E1. If distribution-aware
  selection is ever revisited (different substrate/modality), the chi-floor is the
  right form; min-z is dead.
- **inactive block norms fit χ₃ near-perfectly (KS-D=0.0019)** — the "concept
  membership is a well-posed hypothesis test" framing is empirically real; it just
  doesn't convert to a downstream win here.
- **k is genuinely unjustified** (no repo rationale, README=16 vs notebooks=8) and
  cuts a smooth continuum (rank8/rank9 ratio 1.048) — but fixed-k being arbitrary
  ≠ adaptive being better; top-k's arbitrariness turns out to be harmless here.

## Why this might still differ elsewhere (for the Goodfire note / idea #2)
Everything here is VISION (DINOv3), where per-patch concept count may just not vary
enough for adaptivity to matter — 256 blocks, dense visual texture, most patches
genuinely ~similar complexity. The min-p/top-nσ win was on LANGUAGE logits (huge
vocab, wildly varying per-token entropy). The honest open question: does
distribution-aware selection matter more where per-input complexity varies MORE —
i.e. language, or BSF-on-attention (idea #2)? This negative is scoped to vision-BSF,
not to the whole idea family.
