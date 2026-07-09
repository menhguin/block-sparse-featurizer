# E1 Pareto results — the "frozen top-k substrate" is rigged for top-k (2026-07-09)

**Setup:** apply 4 selectors to the FROZEN pre-gate magnitudes of the top-k-trained
models, sweep each, trace R²-vs-mean-L0. Plot: `pareto_E1.png`.

## Result: top-k dominates at every matched L0

Δ R² vs top-k at matched mean-L0≈8:

| selector | vanilla | grass |
|---|---|---|
| top-k (baseline) | 0 | 0 |
| **perpatch_chi** (noise-ref) | **−0.003** | **−0.004** |
| global_floor (water-fill) | −0.005 | −0.019 |
| min_z (max-ref) | −0.029 | −0.036 |

Curve shapes (Panel): top-k is the upper envelope everywhere. perpatch_chi hugs it
(nearly coincident, L0 5-15). global_floor close. **min_z is a disaster** — flat
~0.56-0.59 and *declining* as L0 rises.

## Why top-k wins here — the confound I under-weighted

**This test is structurally biased toward top-k, and I should have flagged it louder
before running it.** The magnitudes were TRAINED with top-k selection in the loop.
The encoder/decoder co-adapted to "exactly the top-8 blocks get decoded together" —
so block norm on a frozen top-k model is not a clean, selector-agnostic measure of
reconstruction contribution. Evaluating a *different* selection rule on magnitudes
optimised for top-k is playing an away game. The right read of E1-on-top-k-substrate
is **"how much does each alternative lose when forced onto top-k's home turf,"** NOT
"which selector is best." I earlier called E1 "the clean test" — that was wrong; the
clean test is E2 (train each selector in the loop), where magnitudes co-adapt to the
rule.

**Correction to a claim in the script header:** I asserted global_floor is R²-optimal
by a water-filling argument. That holds only for a *pure orthonormal projection*
decoder. BSF decoder atoms are orthonormal *within* a concept but NOT across concepts,
and the codes are signed + co-adapted — so adding a block changes reconstruction by its
full signed contribution (can overshoot and HURT). That non-orthogonality is exactly
why min_z's curve *declines*: lowering p floods in low-norm blocks whose untrained
signed codes actively damage the reconstruction. Water-filling optimality was an
overclaim; retracted.

## The actual signal (why this is encouraging, not discouraging)

1. **Noise-referenced ≫ max-referenced, confirmed empirically.** perpatch_chi loses by
   ~0.003; min_z loses by ~0.03 — an order of magnitude worse. Every earlier theoretical
   argument (murky max, clean χ₃ floor) predicted this ordering, and the data delivers it
   hard. min_z is dead; the chi variant is the one that "makes sense."
2. **perpatch_chi essentially TIES top-k despite playing top-k's home game.** Matching
   the co-adapted baseline to within 0.3% *when the deck is stacked* is the opposite of a
   kill-shot — it says the noise-referenced selector is a viable free variable, and E2
   (its home turf) is where it can actually pull ahead.
3. The honest expectation for E2: adaptive wins *if* the co-adaptation flips in its favour.
   If perpatch_chi still only ties top-k under E2, the honest conclusion is "distribution-
   aware selection is a reparameterisation, not a reconstruction win" — which is itself a
   clean, publishable negative that we'd report straight.

## Next

E2: retrain vanilla BSF with (a) top-k, (b) perpatch_chi in the selection loop; compare
Pareto. That's the unbiased test. min_z dropped from contention.
