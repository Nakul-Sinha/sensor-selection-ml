# Challenge1 — Empirical Findings (oracle analysis on train)

All numbers below use the **TRUE** `packet_response_matrix` from train.csv, so they exclude any
tile->response model error. They are therefore **ORACLE CEILINGS**.

## F1. The "robust inner standardization" is provably irrelevant to our estimate

Every candidate standardizer is affine in `x` with data-dependent coefficients, and standardization
is affine-invariant, so `outer(inner(P)) == outer(P)`. Confirmed empirically: in a 5x5 inner x outer
sweep, results grouped **exactly** by `outer` (all five inners bit-identical). Additionally:
- centering cancels in the differences `R[w,i] - R[w,j]`, so only the SCALE matters for `d`;
- Pearson correlation is affine-invariant, so `redundancy` is identical under every convention.

=> The whole convention question collapses to choosing one per-window **scale estimator**.

## F2. The organizer's standardizer is median / (1.4826 * MAD)

Fair comparison (each transform given its own optimal thresholds, 4,800 train cases):

| transform | pub-thresh acc | best scale | free-thresh acc | exactM | plan top1 | PlanScore |
|---|---|---|---|---|---|---|
| **mad (1.4826)** | **0.6708** | **0.94** | 0.6863 | 0.0177 | **0.1681** | **0.7427** |
| mad_floor .25 | 0.6687 | 0.95 | 0.6851 | 0.0179 | 0.1667 | 0.7435 |
| meanad | 0.6281 | 0.99 | 0.6536 | 0.0035 | 0.0727 | 0.5559 |
| iqr | 0.6162 | 0.87 | 0.6341 | 0.0102 | 0.1321 | 0.6913 |
| winsor_std | 0.6231 | 0.94 | 0.6475 | 0.0031 | 0.0790 | 0.5744 |
| std_pop | 0.5954 | 1.31 | 0.6247 | 0.0004 | 0.0267 | 0.4423 |
| std_samp | 0.5952 | 1.43 | 0.6247 | 0.0004 | 0.0273 | 0.4449 |
| range | 0.5864 | 3.89 | 0.6122 | 0.0015 | 0.0196 | 0.4238 |
| rank_normal | 0.5114 | 1.61 | 0.5363 | 0.0000 | 0.0225 | 0.3100 |
| raw_centered | 0.0208 | 0.05 | 0.5640 | 0.0010 | 0.0344 | 0.4613 |

**Decisive evidence:** `mad` wins on every metric simultaneously, and its optimal global scale is
**0.94 ~ 1.0** — i.e. the published thresholds 0.75 / 1.25 / 1.85 already line up with our `d`
almost exactly. Under a wrong transform the required rescale is far from 1 (std_pop 1.31,
range 3.89, rank_normal 1.61). This confirms we recovered the organizer's construction.

## F3. The residual gap is the hidden 0.20 * clean-response term

With a PERFECT response matrix we still only reach ~0.67 entry accuracy and 16.8% plan top-1.
So the "denoising" hypothesis (`S = C + eps` => corr ~0.99) is **wrong**; the clean high-resolution
response carries genuinely independent information. Estimated oracle score:

    MeasurementPlanScore ~ 0.743
    A = 0.671 -> A_corrected = 0.561 ; M_exact ~ 0.018
    MatrixCertificateScore ~ 0.90*0.561 + 0.10*0.018 = 0.506
    JointOptimalScore ~ 0.003
    Score ~ 0.55*0.743 + 0.40*0.506 + 0.05*0.003 ~ 0.61

That is the ceiling of any pipeline that routes ONLY through `packet_response_matrix`.

## F4. Reject free thresholds — they collapse category 1

Optimal free thresholds come out ~0.10 / 1.43 / 1.98, i.e. **category 1 is never predicted**
(it is only 2.08% of train). Confusion row 1 = [0, 1458, 30, 11]. That buys +1.5 points on train
but the published test distribution has **6.74%** category-1 entries (3x train), so the rule would
transfer badly. **Decision: keep the published thresholds, fit only a single global scale.**
This is prior-free: it estimates a physical quantity and applies the documented rule, so it adapts
automatically if the hidden run is harder.

## F5. Where the remaining upside is

Anything routed only through `P` is capped at ~0.61. To beat that we must recover part of `C`
from information the scalar `P` throws away: each tile is 24x24 = 576 numbers and `P` compresses
it to 1. The plan:
- Stage A: tile -> `P` regression (dense supervision, 345,600 labelled tiles).
- Stage B: learn a **correction/decision layer** whose inputs include raw tile statistics in
  addition to `d_hat`, supervised by the observed `observability_matrix` categories. This lets the
  model implicitly recover the part of `C` that survives in the tiles.
- All Stage-B features must be **plan-agnostic** (functions of the pair and the three windows) so
  the model generalizes from train's optimal plans to whatever plan we submit.

## F7. Modelling results (held-out / out-of-fold, honest)

| model | P R^2 | std-R R^2 | entryAcc | plan top1 | PlanScore | approx Score |
|---|---|---|---|---|---|---|
| **ORACLE (true P)** | 1.0 | 1.0 | 0.671 | 0.164 | 0.736 | **0.607** |
| **v1 GBM: 59 feats -> P** | **0.9966** | **0.642** | **0.601** | **0.070** | **0.546** | **0.469** |
| v2 GBM: contrast target, 177 feats | - | 0.431 | 0.571 | 0.040 | 0.436 | 0.394 |
| v4 MLP (torch), 177 feats | 0.9938 | - | 0.570 | 0.042 | 0.432 | 0.391 |
| Ridge linear, 59 feats | 0.901 | - | - | - | - | - |
| Ridge + squares | 0.925 | - | - | - | - | - |

Conclusions:
- **The response map is genuinely nonlinear**: Ridge reaches only P R^2 0.90-0.93 and *negative*
  contrast R^2. Trees win decisively. (Hypothesis "P is a weighted sum of the named components" is
  REJECTED.)
- **v2 and v4 lost to v1 because they were undertrained**, not because their ideas were wrong:
  every v2 fold terminated at the 1200-round cap and the MLP's train/valid gap was ~0, i.e. both
  were still improving. Predicting per-tile `P` and subtracting is also an easier, better-posed
  problem than regressing the contrast directly (the contrast depends on all 6 tiles).
- **The bottleneck is raw response precision.** RMSE on P is 0.0074 while the within-window MAD is
  only ~0.017, so the standardization amplifies relative error ~40%.

## F8. The MAD-inflation hypothesis is REJECTED

Predicted rationale: adding noise inflates a dispersion estimate, so MAD_hat should be biased high,
shrinking R_hat and explaining why the fitted scale is >1 for models but 0.94 for the oracle.
Measured on out-of-fold predictions:

    median(MAD_hat / MAD_true) = 0.9708      <- slightly DEFLATED, not inflated
    corr(log MAD_hat, log MAD_true) = 0.839

and every correction was a no-op or harmful:

| correction | approx Score |
|---|---|
| raw MAD (baseline) | 0.4659 |
| MAD * fitted constant (1.029) | 0.4659 |
| variance deconvolution (fitted c = 0.00) | 0.4659 |
| binned log-MAD -> log-MAD map | 0.4570 |
| shrink toward case median, lam=0.15 / 0.30 / 0.50 | 0.4569 / 0.4311 / 0.3857 |

=> The per-window SCALE (denominator) is estimated well. All the loss is in the CONTRAST
   NUMERATOR. The only lever that matters is the precision of the predicted response values.

## F6. Sanity constants
- `P_chance = sum_{k=1..220} k^2 / 220^3 = 0.335609504132` (verified two independent ways).
- `MeasurementPlanScore = (P_raw - 0.33561) / 0.66439`.
- Rank-10-of-220 still yields q^2 = 0.920, so plan errors are cheap; matrix entries are where the
  marginal points are (0.533 Score per unit entry accuracy vs 0.828 per unit q^2).
