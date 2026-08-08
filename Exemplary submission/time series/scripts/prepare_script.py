"""Temporal train/test split for time-series liquidity data."""
import pandas as pd
import numpy as np
from pathlib import Path


def prepare(raw: Path, public: Path, private: Path) -> None:
    np.random.seed(42)

    features = pd.read_csv(
        list(raw.glob("*features.csv"))[0], index_col=0, parse_dates=True
    )
    target = pd.read_csv(
        list(raw.glob("*target.csv"))[0], index_col=0, parse_dates=True
    )

    # Sort chronologically (critical for time-series)
    features = features.sort_index()
    target = target.sort_index()

    # Temporal split: 80% train, 20% test (strictly chronological)
    split_idx = int(len(features) * 0.8)

    X_train = features.iloc[:split_idx]
    X_test = features.iloc[split_idx:]
    y_train = target.iloc[:split_idx]
    y_test = target.iloc[split_idx:]

    # Verify no temporal overlap
    if X_train.index.max() >= X_test.index.min():
        raise ValueError("Temporal split failed: train and test data overlap")

    # Public: train with labels, test without
    train_df = X_train.copy()
    train_df["liquidity_target"] = y_train["liquidity_target"]
    train_df.to_csv(public / "train.csv")
    X_test.to_csv(public / "test.csv")

    # Sample submission with NaN placeholders
    pd.DataFrame(
        index=X_test.index, data={"liquidity_target": np.nan}
    ).to_csv(public / "sample_submission.csv")

    # Private answers
    y_test.to_csv(private / "answers.csv")