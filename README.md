# ML challenge work

I am working on a machine learning challenge in signal processing and sensor selection.

The task is a combination of regression and combinatorial optimisation. I start from stacked
time by frequency tiles, learn a scalar response value for each tile, and use those predictions to
choose an optimal small subset of sensors and produce a categorical separability matrix.

## Stack

- Python, NumPy, pandas
- LightGBM for the main regression model
- PyTorch and scikit-learn for baselines I compared against
- joblib for parallel feature extraction

## Layout

| Path | What it is |
|---|---|
| `solution.py` | The end to end script |
| `code/` | Experiment scripts, feature extraction, and a local scorer |
| `tools/ship.ps1` | Helper that turns one change into one branch, one PR, and a squash merge |

Data files and my working notes are gitignored.
