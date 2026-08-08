import pandas as pd
import numpy as np


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    if not isinstance(submission, pd.DataFrame):
        raise ValueError("Submission must be a pandas DataFrame")

    if "liquidity_target" not in submission.columns:
        raise ValueError("Submission must contain 'liquidity_target' column")

    if len(submission) != len(answers):
        raise ValueError(
            f"Submission length ({len(submission)}) doesn't match "
            f"answers ({len(answers)})"
        )

    # Handle NaN values
    nan_percent = (submission["liquidity_target"].isna().sum() / len(submission)) * 100
    if nan_percent > 50:
        raise ValueError(f"Submission has {nan_percent:.1f}% NaN values (max: 50%)")

    # Align indices
    common_idx = submission.index.intersection(answers.index)
    if len(common_idx) < len(answers) * 0.8:
        raise ValueError(f"Only {len(common_idx)/len(answers)*100:.1f}% indices match")

    y_pred = submission.loc[common_idx, "liquidity_target"]
    y_true = answers.loc[common_idx, "liquidity_target"]

    # Fill remaining NaNs with forward-fill
    if y_pred.isna().any():
        y_pred = y_pred.ffill().bfill()

    valid_mask = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_mask].values.astype(np.float64)
    y_pred = y_pred[valid_mask].values.astype(np.float64)

    if len(y_true) < 10:
        raise ValueError(f"Too few valid data points: {len(y_true)} (min: 10)")

    # SMAPE calculation
    numerator = 2 * np.abs(y_true - y_pred)
    denominator = np.abs(y_true) + np.abs(y_pred)

    zero_mask = denominator == 0
    smape_values = np.zeros_like(y_true)
    smape_values[~zero_mask] = numerator[~zero_mask] / denominator[~zero_mask]

    smape = 100 * np.mean(smape_values)
    return max(0.0, float(smape))