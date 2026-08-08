# Project Eris challenges

This repo is where I track my work on the Project Eris challenges from shipd.ai.

## What I am doing

I am solving **Challenge1: Bridge Sensor Window Selection and Observability Matrix Prediction**.

Each case gives me a packet of vibration data: 6 episodes x 12 candidate sensor windows, and each
of those 72 combinations is a 24x24 time by frequency tile. I have to pick the best 3 of the 12
windows, then hand in a 6x6 matrix saying how well those 3 windows separate every pair of episodes,
graded on a scale of 1 to 4.

The thing that makes this tractable is that the grader is not mysterious. It keeps a hidden catalog
`R` (12 windows x 6 episodes) and derives both the plan score and the matrix from it using formulas
that are published in the problem statement. So the whole challenge collapses to one question: how
accurately can I estimate `R`?

## My approach

I estimate `R` in four steps.

1. I extract 59 spectral and temporal statistics from every 24x24 tile.
2. I train a LightGBM model to predict each tile's response value. Training gives me 4,800 cases
   x 72 tiles, so 345,600 labelled examples, which is plenty.
3. I standardize the 6 episode values within each window, which is exactly how the organizer builds
   the catalog.
4. I run the published utility formula over all 220 possible 3-window plans, take the best one, and
   bucket its 15 pair distances into the 1 to 4 categories.

Everything downstream of step 2 is just the published arithmetic. The model does the actual
learning, which is what the rulebook asks for.

## What I figured out along the way

I spent a while reverse engineering how the organizer builds the catalog, and confirmed it is a
median over MAD standardization. Two pieces of evidence convinced me. First, it wins on every
metric at once. Second, and more tellingly, the rescaling factor it needs is 0.94, basically 1.0,
meaning the published thresholds already line up with my distances. Every wrong convention needs a
rescale far from 1.

I also proved that the choice of the inner "robust" standardization does not matter at all, because
standardization is affine invariant. That collapsed a 25 way search down to a single scale constant.

A few things I tried and threw away, so I do not repeat them:

- The response is not a linear combination of the named statistics. Ridge regression gets a
  negative R squared on the part that matters.
- A neural net loses to LightGBM here.
- My theory that noise inflates the MAD estimate was wrong. I measured it at 0.971, slightly
  deflated, and every correction I tried made things worse.

That last round of testing was useful because it isolated the real bottleneck, which is simply the
precision of the predicted response values.

## Layout

| Path | What it is |
|---|---|
| `Challenge1/solution.py` | The submission script. Run it as `python3 solution.py <public_dir> <submission_out>` |
| `Challenge1/PROBLEM.md` | The challenge statement, saved verbatim |
| `Challenge1/STRATEGY.md` | My plan and the design decisions behind it |
| `Challenge1/FINDINGS.md` | Measurements, including the things I ruled out |
| `Challenge1/code/` | Experiment scripts. Not part of the submission |
| `Exemplary submission/` | Reference challenges I archived earlier, for style |
| `eris-guidebook/` | The solver rulebook |
| `tools/ship.ps1` | Helper that turns one change into one branch, one PR, and a squash merge |

Datasets and credentials are gitignored. The challenge data is about 500 MB, so it does not belong
here.

## Rules I am working under

The rulebook is strict about a few things, so I designed around them from the start.

- Real training has to happen inside the submission script. Mine does. If you delete the model,
  nothing can be produced at all, because the test file only carries an id, a file path, and a
  constant string.
- I am not allowed to hardcode things I discovered offline. So even though I know the standardizer
  is median over MAD, the script re-selects it at runtime from a menu of six standard estimators,
  and re-fits the scale factor at runtime too.
- The script has to run end to end from raw data every time, so no cached features and no saved
  models.
- I deliberately do not use the published test set margin counts to calibrate anything. That table
  is derived from withheld test responses, and tuning against it would break the rules.
