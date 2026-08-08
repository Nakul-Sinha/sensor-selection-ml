# Challenge1 — Bridge Sensor Window Selection and Observability Matrix Prediction

Source: https://shipd.ai/quests/eris/solutions/k979aq0a77rcmqd59d602jn7e58c3rxr

## Deliverables

| File | What it is |
|---|---|
| `solution.py` | **DELIVERABLE 1.** Self-contained end-to-end script. `python3 solution.py <public_dir> <submission_out>` |
| `submission.csv` | **DELIVERABLE 2.** Produced by running `solution.py` on the public dataset. |

## Documentation

| File | What it is |
|---|---|
| `PROBLEM.md` | Full challenge statement, captured verbatim. |
| `STRATEGY.md` | The approach and the design decisions taken before/while modelling. |
| `FINDINGS.md` | Empirical results: what was proven, measured, and what was tried and rejected. |

## Data

`dataset/public/` — extracted from the challenge zip (417 MB):
- `train.csv` (4,800 cases, 6 columns incl. targets)
- `test.csv` (1,000 cases, 3 columns)
- `sample_submission.csv`
- `response_packets/*.npy` — 5,800 float16 packets of shape (6, 12, 24, 24)

## Development code (`code/`)

NOT part of the submission — these are the experiment harnesses.

| File | Purpose |
|---|---|
| `features.py` | Vectorised 59-statistic tile feature extractor (+ tests, benchmark). |
| `scorer.py` | Independent implementation of the published Observability Code Score (+ 10 tests). |
| `explore_targets.py` | 25-way inner x outer standardization sweep; established the oracle ceiling. |
| `explore2.py` | Focused median/MAD diagnostics, free-threshold search. |
| `explore3.py` | Fair comparison of 11 per-window scale estimators, each with its own thresholds. |
| `build_features.py` | Caches extracted features for fast local iteration (dev only). |
| `model_response.py` | v1: GBM, 59 features -> packet_response. **Best configuration.** |
| `model_response2.py` | v2: contrast target + window-relative features. |
| `model_response3.py` | v3: direct standardized-R target. |
| `model_linear.py` | Ridge / polynomial hypothesis test (rejected). |
| `model_nn.py` | Torch MLP (loses to GBM). |
| `model_deep.py` | Plateau search with a large round budget. |
| `test_madfix.py` | MAD-correction experiments (hypothesis rejected). |

## The core idea in one paragraph

The grader keeps a hidden catalog `R` (12 windows x 6 episodes) and derives BOTH the plan utility
and the observability matrix from it with published formulas. So the entire task reduces to
estimating `R`. The challenge statement itself sanctions the route: regress each tile's
`packet_response_matrix` value from the tile, predict it for test tiles, and standardize the six
episode values within each window. Training supplies 4,800 x 72 = 345,600 labelled tiles for that
regression. Everything downstream is the published arithmetic.

## Reproduce

```bash
python3 solution.py ./dataset/public ./working/submission.csv
```

Runtime is dominated by feature extraction (~4 s for all 5,800 packets on 10 cores) and the
LightGBM fit. The script carries a wall-clock guard and degrades by subsampling training cases
rather than overrunning the 1.5 h budget.

## Compliance summary (Eris guidebook)

- Real supervised training happens inside `solution.py`; remove the model and nothing can be
  produced (test.csv carries only an id, a packet path, and a constant contract string).
- Nothing discovered offline is hardcoded: the per-window scale estimator is re-selected at runtime
  from a menu of six standard dispersion estimators, and the global distance scale is re-fitted at
  runtime on out-of-sample training predictions.
- The only hardcoded constants are quoted from the published statement (thresholds 0.75/1.25/1.85,
  utility weights 0.18/0.035, the 220-plan enumeration).
- End-to-end from raw data every run; no cached features, no pickled models.
- No external data, no pretrained weights, no synthetic training data, no network use.
- Inference uses only each case's own packet; no statistic is pooled across test cases.
- The published *test* margin-count table is deliberately NOT used to calibrate anything — that
  would be calibrating on withheld test responses.
