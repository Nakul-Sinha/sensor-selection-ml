# Challenge1 — Solution Strategy

## The causal chain (from the spec)

```
tile X[e,w] (24x24)
  --f-->  P[e,w]              packet_response_matrix        (TRAIN-VISIBLE label)
  --standardize over e within each w-->  S[:,w]
  --0.80*S + 0.20*C-->  G      (C = hidden clean high-res response, standardized)
  --standardize over e within each w-->  R[w,e]  shape (12,6)   (GRADER'S CATALOG)
  --> d(i,j,P) = sqrt(mean_{w in P} (R[w,i]-R[w,j])^2)
  --> utility(P) = min(D) + 0.18*mean(D) - 0.035*redundancy(P)
  --> plan = argmax utility ; matrix[i][j] = bucket(d) @ 0.75 / 1.25 / 1.85
```

**The whole problem reduces to estimating `R`.** Everything downstream is deterministic arithmetic.

## Key observations

1. **`P[e,w]` is a deterministic per-tile scalar.** The spec even names its ingredients: frequency
   centroid, low-to-mid vs high band balance, temporal centroid, spectral roughness, pulse contrast.
   Train gives 4,800 x 72 = **345,600 labeled tiles**. This is a dense, well-posed regression.

2. **We can never see `C`** (the 0.20 clean component). If `C` were independent of `S`:
   `R = (0.8S + 0.2C)/sqrt(0.68) = 0.970*S + 0.243*C` -> corr(R, S) ~ 0.97.
   So `R_hat = S` is a strong estimator; the 0.2 term sets our accuracy ceiling.
   In reality C and S measure the same physical response, so correlation is likely higher.

3. **The grader re-derives the matrix for OUR submitted plan.** So plan and matrix must be
   *self-consistent*. We never submit a train-copied matrix. Alternate good plans score well.

4. **`plan_raw = q^2`, chance `P_chance ~ 0.336`** (mean of (k/220)^2 over k=1..220 = 0.3356).
   Being in the top handful of 220 plans is nearly as good as optimal, so small ranking errors
   are cheap. Matrix entries are where the marginal points are.

## Pipeline

### Stage 1 — tile -> response regression (the real ML)
- Vectorized feature extraction per tile: spectral/temporal centroids & spreads, band-energy
  ratios (low/mid/high), roughness (successive-diff energy), pulse contrast (peak/median, kurtosis),
  total energy, percentiles, marginal-profile moments, entropy.
- Fit gradient-boosted regressor (LightGBM) on train tiles -> `P`. Validate by held-out CASE groups
  (never split within a case).
- Target: R^2 > 0.99. If the mapping is essentially linear in the named features, this is achievable.

### Stage 2 — build R_hat
- Predict `P_hat` for every test tile, reshape to (6,12).
- Robustly standardize the 6 episode values within each window. **Determine the exact estimator
  (mean/std vs median/MAD) empirically** by which one best reproduces train targets.
- `R_hat = S^T` -> shape (12,6).

### Stage 3 — plan selection
- Evaluate all 220 plans with the published utility formula, take argmax.
- Fully vectorizable: precompute per-window pair-diff squares (12 x 15) and window-row correlations
  (12 x 12), then combine per plan.

### Stage 4 — matrix as a *learned* calibration (not naive thresholding)
Because `R_hat != R`, naive bucketing of `d_hat` at 0.75/1.25/1.85 is biased. Instead train a
classifier on train data:
- For each train case, using its known `window_schedule`, compute `d_hat(i,j)` from our `R_hat`.
- Label = the true margin category from `observability_matrix`.
- Features: `d_hat`, the three per-window |differences| that compose it, case-level distance stats
  (mean/std/min/max/rank of this pair within the case), redundancy of the plan.
- Learn `P(margin | features)` -> Bayes-optimal category. This recovers the systematic distortion
  from the missing 0.20 C component and should beat hard thresholds meaningfully.

### Stage 5 — assemble submission
Exactly 3 columns in order, one row per test id, ascending window tokens, symmetric matrix,
zero diagonal, values 1..4, JSON compact (no spaces) to respect the 220-char matrix limit and
15-char schedule limit.

## Validation protocol (guards against over/under-fitting)
- **Group split by case_id** for the tile regressor.
- **Oracle ceiling run:** feed the TRUE train `P` through Stages 2-4 and score against train
  targets. This isolates "how much the hidden C costs us" from "how much our regression costs us".
- **Full local scorer:** implement the exact Observability Code Score (plan + matrix + joint,
  with chance correction) and run it on a held-out slice of train. This is the number we optimize.
- Never touch test labels; no leaderboard probing (challenge explicitly forbids it).

## Compliance notes (guidebook)
- Real training happens **inside** `solution.py` (Stage 1 + Stage 4 both fit models at runtime).
- End-to-end from raw data every run; no cached artifacts, no external data, no pretrained weights
  needed at all.
- No use of `case_id`, row order, path, or hashes as predictors (explicitly forbidden).
- Deterministic arithmetic downstream of the model is fine: it is the *published scoring formula*,
  not a hardcoded dataset insight. The learned parts (P regressor, margin calibrator) carry the
  insight.
- CPU-only, target well under the 1.5h budget; include a wall-clock guard.

## Score expectation
With corr(R_hat, R) ~ 0.97 and accurate P regression:
- MeasurementPlanScore ~ 0.85 (P_raw ~ 0.90 vs chance 0.336)
- MatrixCertificateScore ~ 0.54-0.70 depending on calibration quality
- Score ~ 0.65-0.75, versus the 0.32 bar.

---

# Refinements (decided before data landed)

## R1. Why the hidden 0.20 may cost far less than the worst case
If the clean response `C` and the visible response `S` measure the *same* physical quantity, then
plausibly `S = C + eps`. In that case
`G = 0.8*S + 0.2*C = C + 0.8*eps` — the hidden term **denoises** rather than adding new information.
With Var(C)=1, Var(eps)=s^2:  `corr(C+eps, C+0.8s^2...) = (1+0.8s^2)/sqrt((1+s^2)(1+0.64s^2))`
  s^2=0.25 -> 0.9966 ; s^2=1.0 -> 0.9939.
Worst case (C independent of S) gives corr 0.970. The oracle run measures which world we are in.

## R2. Calibrate on PREDICTED responses, not true ones  (avoids a silent train/test mismatch)
The margin calibrator must see the same noise distribution it will face at test time. Calibrating on
`d` computed from the TRUE train `P` would be optimistic and mis-set the decision boundaries.
Protocol:
- Split train cases: **fit-set (~70%)** trains the tile->response regressor.
- **calib-set (~30%)**: regressor predicts `P_hat` out-of-sample -> `R_hat` -> `d_hat`. The margin
  calibration and the convention selection are fit here, and honest validation is reported here.
- Optionally refit the tile regressor on 100% of train for the final test prediction, keeping the
  calibration fixed.

## R3. Prior-free thresholds vs learned classifier  (guard against distribution shift)
The published train/test margin distributions **differ materially**:
  train  1:2.08%  2:23.1%  3:25.1%  4:49.8%
  test   1:6.74%  2:22.7%  3:20.1%  4:50.5%
Test has ~3x more "weak separation" entries -> the third field run is genuinely harder/noisier.
A Bayes classifier trained on train imports the *train prior* and would systematically over-predict
high categories on test. Therefore:
- **Primary rule: make `d_hat` well-calibrated (an unbiased estimate of true `d`), then apply the
  PUBLISHED thresholds 0.75/1.25/1.85.** This is prior-free and shift-robust.
- Fit only a *monotone scalar correction* `d_hat -> d` (isotonic / simple scale+offset) on the
  calib-set. Monotone correction cannot import class priors.
- A full classifier is implemented as a challenger and compared on held-out data, but is only
  adopted if it wins for a reason that survives the shift argument.
- Do NOT use the published test count table to re-weight predictions: those counts are derived from
  withheld test responses, and the challenge forbids calibrating on them. (Also they describe the
  *canonical* plan's matrices, not our submitted plan's.)

## R4. Marginal-value arithmetic for joint plan/matrix optimization
  d(Score)/d(entry accuracy A) = 0.40 / 0.75 = **0.533 per unit A**
  d(Score)/d(q^2)              = 0.55 / 0.6644 = **0.828 per unit q^2**
Dropping from rank 1 to rank 2 of 220 costs q^2: 1 -> 0.9909, i.e. 0.0075 of Score.
So deviating one rank is only worth it if it buys **> ~1.4 percentage points** of entry accuracy.
=> Implement a *tiebreak only* among near-tied plans: prefer the plan whose 15 distances sit
   farthest from the 0.75/1.25/1.85 boundaries (most confidently bucketable). Validate; drop if flat.

## R5. Permutation invariance is free
Episodes and windows are shuffled independently per case. The pipeline is naturally invariant:
per-tile regression is per-tile; within-window standardization is a set operation over episodes;
plan enumeration is over unordered triples. Nothing needs de-shuffling, and nothing may key off
window/episode index identity.

## R6. Runtime budget shape
Dominant costs are (a) reading ~5,800 .npy files and (b) fitting the regressor. Feature extraction is
vectorized over all 72 tiles at once. Expected total well under 15 min on 8-10 CPU cores vs the 1.5 h
allowance. Include an explicit wall-clock guard that degrades gracefully (subsample training cases)
rather than ever overrunning.
