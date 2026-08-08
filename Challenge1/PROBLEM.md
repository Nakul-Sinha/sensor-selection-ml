# Bridge Sensor Window Selection and Observability Matrix Prediction

**Source:** https://shipd.ai/quests/eris/solutions/k979aq0a77rcmqd59d602jn7e58c3rxr
**Category:** Other · **Difficulty:** Medium · **Status:** Accepted
**Score range:** 0 to 1 — **higher is better**
**Entrypoint:** `python3 solution.py <public_dir> <submission_out>` (read both paths from `sys.argv`)
**Submission credits:** 6/6 at time of capture

---

## Overview

Choose three of twelve bridge-sensor windows, then describe how well those three windows distinguish six recorded bridge conditions.

Each case contains vibration evidence from six physical episodes, labeled e0 through e5, at twelve candidate measurement windows, labeled w00 through w11. The first submitted field names the three windows to retain. The second is a 6 by 6 matrix: entry `[i][j]` states how clearly the selected windows separate episode ei from episode ej, using categories 1 through 4.

This models a field-monitoring budget decision. A bridge team may have twelve usable measurements but enough telemetry, power, or processing capacity to preserve only three. A useful selection must distinguish all episode pairs, including the hardest pair, while avoiding three windows that carry nearly identical information.

The time-frequency tiles come from repeated bridge-vibration measurements. Training uses two field runs and the hidden set uses a third run. Episode and window order are shuffled independently in every case. The task is measurement planning from vibration evidence, not damage-label classification.

---

## Dataset

### Files

| Path | Description |
|---|---|
| `train.csv` | 4,800 cases with inputs, scored targets, and a train-only response supervision matrix. |
| `test.csv` | 1,000 cases built from a held-out repeated field run. |
| `sample_submission.csv` | A schema-valid fixed plan and fixed matrix for every test case. |
| `response_packets/*.npy` | Float16 time-frequency packets referenced by the CSV files. |

### CSV Columns

| Column | Data type | Availability | Description |
|---|---|---|---|
| `case_id` | string | train, test | Opaque identifier used only to align submissions. |
| `response_packet_path` | relative path string | train, test | Path relative to the public dataset root, such as `response_packets/bw_45a1....npy`. |
| `packet_contract` | fixed-format string | train, test | Declares packet shape, episode labels, window labels, and the three-window budget. |
| `window_schedule` | canonical token set | train only | Three selected window IDs. |
| `observability_matrix` | JSON integer matrix | train only | Symmetric 6 by 6 separation certificate for the selected windows. |
| `packet_response_matrix` | JSON float matrix | train only | Shape `(6, 12)`. Each row is an episode and each column is a public window. Values summarize visible spectral structure and provide auxiliary supervision. |

`train.csv` contains all six columns. `test.csv` contains only `case_id`, `response_packet_path`, and `packet_contract`.

### Packet Contract Format

`packet_contract` is the same fixed semicolon-separated string in every row:

```
shape=6x12x24x24;episodes=e0_e5;windows=w00_w11;window_budget=3
```

It is descriptive metadata rather than a hidden instruction. Split the string at semicolons, then split each segment at the first `=` character:

| Key | Value | Meaning |
|---|---|---|
| `shape` | `6x12x24x24` | Packet axes are six episodes, twelve windows, 24 time bins, and 24 frequency bins. |
| `episodes` | `e0_e5` | Episode labels run from `e0` through `e5`. |
| `windows` | `w00_w11` | Candidate window labels run from `w00` through `w11`. |
| `window_budget` | `3` | Every submitted schedule must contain exactly three distinct windows. |

### Response Packet

Every `.npy` file has shape `(6, 12, 24, 24)` and dtype float16:

- axis 0 contains episodes e0 through e5;
- axis 1 contains candidate windows w00 through w11;
- axes 2 and 3 form a 24 by 24 time-frequency tile.

One episode comes from a baseline condition and five come from physically changed conditions. Their order is randomized, and the omitted physical state varies by case. A candidate window represents one longitudinal station and one acceleration axis, aggregated across five stringers. Window identities are randomized independently in every case. Tiles summarize approximately 32 seconds of response between 0.5 and 40 Hz. Contrast changes, time rolls, additive noise, and occasional local dropouts prevent exact source matching.

### Packet Response Supervision

`packet_response_matrix` contains 72 finite floating-point values in episode-major order. Entry `[e][w]` summarizes the visible frequency centroid, low-to-middle and high-band balance, temporal centroid, spectral roughness, and pulse contrast of the tile at episode e and window w. It is an auxiliary training label and is not a submission column.

A solver can regress these values from training tiles, predict them for test tiles, and robustly standardize the six episode values within each window. The hidden response catalog used for target construction combines **0.80 times this standardized packet response with 0.20 times a standardized clean high-resolution response**. The six combined episode values within every window are standardized once more before plans and matrix margins are computed. This keeps a real clean-response contribution while making most of the target evidence observable in the released packet.

### Window Schedule

`window_schedule` contains exactly three distinct IDs from w00 through w11, joined by `|`:

```
w01|w06|w10
```

Token order does not change the score. Training targets use ascending order for a canonical representation. All 220 possible three-window schedules occur in training. The largest training schedule share is 0.812 percent; the largest test share is 1.30 percent.

### Observability Matrix

`observability_matrix` is a symmetric 6 by 6 JSON matrix. Rows and columns follow episode order e0 through e5. Diagonal entries are 0. Every off-diagonal entry is a categorical separation margin from 1 through 4:

| Value | Meaning |
|---:|---|
| 1 | Weak separation, normalized margin below 0.75. |
| 2 | Limited separation, margin from 0.75 up to but not including 1.25. |
| 3 | Strong separation, margin from 1.25 up to but not including 1.85. |
| 4 | Very strong separation, margin at least 1.85. |

Example:

```
[[0,3,4,2,3,4],[3,0,2,1,4,3],[4,2,0,3,2,4],[2,1,3,0,3,2],[3,4,2,3,0,4],[4,3,4,2,4,0]]
```

The matrix always refers to the submitted three-window plan. Submitting the hidden target matrix with a different plan is not coherent and is scored against the different plan actually submitted.

### Matrix Distribution

Counts below use the 15 unique off-diagonal entries per matrix.

| Margin value | Train entries | Test entries |
|---:|---:|---:|
| 1 | 1,499 | 1,011 |
| 2 | 16,636 | 3,401 |
| 3 | 18,038 | 3,007 |
| 4 | 35,827 | 7,581 |

---

## Submission Format

Write the final CSV to `./working/submission.csv` (in practice: the exact path given as `sys.argv[2]`).

The file must contain exactly these columns in this order:

| Column | Data type | Required content |
|---|---|---|
| `case_id` | string | One exact test identifier. |
| `window_schedule` | canonical token string | Exactly three distinct valid window IDs separated by `\|`. |
| `observability_matrix` | JSON integer matrix | Symmetric shape `(6, 6)`, zero diagonal, and off-diagonal values from 1 through 4. |

Example:

| case_id | window_schedule | observability_matrix |
|---|---|---|
| `bw_47f74cfc329d84facfb0` | `w01\|w06\|w10` | `[[0,3,4,2,3,4],[3,0,2,1,4,3],[4,2,0,3,2,4],[2,1,3,0,3,2],[3,4,2,3,0,4],[4,3,4,2,4,0]]` |

Include exactly one row for every test ID. **Extra columns, reordered columns, duplicate IDs, missing IDs, unknown IDs, blank IDs, and row-count mismatches reject the submission.** A single backend-managed visibility column is accepted and removed before schema validation. **A schedule longer than 15 characters or a matrix longer than 220 characters is malformed.** Malformed row values receive zero for every row-level component.

---

## Evaluation — Observability Code Score

```
Score = 0.55 * MeasurementPlanScore + 0.40 * MatrixCertificateScore + 0.05 * JointOptimalScore
```

Scoring follows three steps:

1. The grader measures the usefulness of the three submitted windows.
2. It constructs the correct separation matrix for those same three windows and compares it with the submitted matrix.
3. It awards a small joint bonus when both the plan and matrix are exactly optimal.

The grader retains a response catalog **R with shape (12,6)**. Row w represents candidate window w; column e represents episode e. As documented under Packet Response Supervision, each value is built from 0.80 times standardized visible-packet evidence and 0.20 times standardized clean high-resolution response evidence. This catalog is the common reference used for both plan scoring and matrix construction.

The important dependency is:

```
submitted window_schedule
    -> select three rows of R
    -> compute plan utility
    -> construct the expected observability_matrix
    -> compare that expected matrix with the submitted matrix
```

Therefore, the matrix is **not** compared only with the matrix printed in the hidden answer row. It is compared with the matrix implied by the participant's submitted schedule. A useful alternate schedule can score well when its accompanying matrix correctly certifies that alternate schedule.

### MeasurementPlanScore

For a submitted three-window plan P and episode pair (i,j), define:

```
d(i,j,P) = sqrt((1/3) * sum((R[w,i] - R[w,j])^2 for w in P))
```

Let `D(P)` be the 15 distances for all episode pairs. Let `redundancy(P)` be the mean absolute Pearson correlation among the three selected six-entry response rows. A nonfinite correlation is assigned redundancy 1.

```
utility(P) = min(D(P)) + 0.18 * mean(D(P)) - 0.035 * redundancy(P)
```

All 12 choose 3 = 220 valid plans are evaluated. Let `q(P)` be the fraction of those plans whose utility is less than or equal to the submitted plan's utility.

```
plan_raw_row = q(P)^2
```

For each hidden case i, define its random-plan baseline as:

```
c_i = (1/220) * sum(q_i(Q)^2 for every legal plan Q)
```

Here, `q_i(Q)` is the percentile rank of plan Q among the 220 plans for case i. Therefore, `c_i` is a per-case normalization constant: the expected squared-rank score when one of that case's 220 legal plans is chosen uniformly at random.

Let `P_raw` be the mean `plan_raw_row` over all hidden cases, and let `P_chance` be the mean of `c_i` over those same cases.

```
MeasurementPlanScore = clip((P_raw - P_chance) / (1 - P_chance), 0, 1)
```

The correction is applied only after both quantities have been averaged over the hidden set. An optimal plan scores 1, while a uniformly random legal plan has expected score 0 after correction.

### MatrixCertificateScore

The grader constructs the correct matrix for each submitted plan using the margin thresholds in the Dataset section. For hidden case i, let `a_i` be the fraction of the 15 unique upper-triangle entries that exactly match the correct matrix for that submitted plan:

```
a_i = matching_upper_triangle_entries_i / 15
```

Let A be the mean of `a_i` over all hidden cases. There are four valid off-diagonal categories, so the metric removes the 0.25 agreement expected from uniform category guessing:

```
A_corrected = clip((A - 0.25) / 0.75, 0, 1)
```

Let `M_exact` be the mean rate of complete 6 by 6 matrix exact matches for the submitted plans.

```
MatrixCertificateScore = 0.90 * A_corrected + 0.10 * M_exact
```

Entry chance correction is applied after averaging all test cases.

### JointOptimalScore

`joint_row` is 1 only when the submitted plan has maximum utility among all 220 plans **and** the submitted matrix is exactly correct for that plan. `JointOptimalScore` is its mean over test cases.

Minimum score: 0.0. Maximum score: 1.0. An exact optimal plan with its exact matrix scores 1.0.

---

## What Makes This Interesting

This is not damage-state classification and it is not ordinary sensor ranking. A selected window is valuable only through the three-window code it forms with the other selections. The weakest episode pair controls much of the utility, redundant sensors are penalized, and the matrix must explain the submitted plan rather than repeat one fixed label.

The setup mirrors a real deployment decision: choose a small measurement budget that preserves discrimination before high-bandwidth acquisition is available everywhere. The hidden-catalog grader also recognizes valid alternate plans, which avoids treating one arbitrary canonical schedule as the only acceptable engineering answer.

---

## What Not To Use (challenge-specific prohibitions)

- Do not use `case_id`, row order, packet order, hashes, path lengths, or split artifacts as predictors.
- Do not match public packets against external copies of the field experiment or build source-state, sensor-name, filename, or waveform lookup tables.
- Do not infer answers from unpublished source metadata, raw state names, sensor coordinates, damage descriptions, or hidden response catalogs.
- Do not exploit duplicate rows, malformed CSV structure, parser behavior, grader feedback, or submission ordering.
- Do not tune or calibrate on withheld test responses through repeated leaderboard probing.
- Hosted or closed-model API calls are not allowed at inference time. Local open-weight models are allowed.

---

## Expected Output

Your script receives the public dataset directory and exact submission CSV path as two positional arguments.

```
python3 solution.py <public_dir> <submission_out>
```
