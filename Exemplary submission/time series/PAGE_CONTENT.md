# Intraday Liquidity Forecasting - Exemplary Submission

Source (copied verbatim from Shipd docs):
`https://shipd.ai/quests/eris/docs/examples/liquidity-forecasting`

Tags: EXEMPLARY | regression | time-series | finance | forecasting

## How to use this folder

1. Read `01_dataset.md`, `02_problem.md`, `03_solution.md` in order.
2. Open scripts in `scripts/` (prepare, grading, config, solution).
3. Dataset files: see `dataset/` - the page only lists file names/sizes on UI cards (not direct public download links), so binaries are documented, not included.
4. `liquidity-forecasting.md` is a formatted single-file version of everything (prose + code).

## Dataset

See `01_dataset.md` (verbatim page text).

## Problem

See `02_problem.md` (verbatim page text).

## Solution

See `03_solution.md` (verbatim page text).

## Scripts (verbatim from page code editors)

### `prepare_script.py`

```python
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
```

### `grading_script.py`

```python
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
```

### `config.yaml`

```yaml
name: Intraday Liquidity Forecasting
difficulty: medium
domain: tabular

grade:
  direction: minimize
  minimum: 0.0
  maximum: 200.0
```

### `solution_code.py`

```python
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

# Load data
train = pd.read_csv(public_dir / 'train.csv', index_col=0, parse_dates=True)
test = pd.read_csv(public_dir / 'test.csv', index_col=0, parse_dates=True)

y_train = train['liquidity_target'].copy()
X_train = train.drop(columns=['liquidity_target'])

# Handle missing values with forward-fill (respects temporal causality)
for df in [X_train, test]:
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].ffill().bfill().fillna(df[col].median())


def engineer_features(df):
    """Create time-series features using only past data (no leakage)."""
    out = df.copy()

    # Cyclical time encoding
    out['hour_sin'] = np.sin(2 * np.pi * out.index.hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out.index.hour / 24)
    out['day_sin'] = np.sin(2 * np.pi * out.index.dayofweek / 7)
    out['day_cos'] = np.cos(2 * np.pi * out.index.dayofweek / 7)

    # Trading session indicators
    hour = out.index.hour
    out['asia_session'] = ((hour >= 0) & (hour < 8)).astype(int)
    out['europe_session'] = ((hour >= 7) & (hour < 16)).astype(int)
    out['us_session'] = ((hour >= 13) & (hour < 21)).astype(int)
    out['is_weekend'] = (out.index.dayofweek >= 5).astype(int)

    # Lag features (strictly backward-looking to prevent leakage)
    lag_cols = ['volatility', 'trading_volume', 'spread', 'order_flow', 'market_stress']
    existing = [c for c in lag_cols if c in df.columns]
    for col in existing:
        for lag in [1, 2, 3, 5, 10, 20, 30, 60]:
            out[f'{col}_lag_{lag}'] = df[col].shift(lag)

    # Rolling statistics (backward-looking windows)
    for col in existing[:5]:
        for window in [5, 10, 30, 60, 120]:
            out[f'{col}_rolling_mean_{window}'] = df[col].rolling(window, min_periods=1).mean()
            out[f'{col}_rolling_std_{window}'] = df[col].rolling(window, min_periods=1).std()
            out[f'{col}_rolling_min_{window}'] = df[col].rolling(window, min_periods=1).min()
            out[f'{col}_rolling_max_{window}'] = df[col].rolling(window, min_periods=1).max()

    # Cross-feature interactions
    if 'volatility' in df.columns and 'spread' in df.columns:
        out['vol_spread_interaction'] = df['volatility'] * df['spread']
    if 'market_stress' in df.columns and 'lcr_ratio' in df.columns:
        out['stress_lcr_interaction'] = df['market_stress'] * (1 - df['lcr_ratio'])

    return out.bfill().ffill()


X_train_eng = engineer_features(X_train)
test_eng = engineer_features(test)

# Align columns between train and test
common_cols = X_train_eng.columns.intersection(test_eng.columns)
X_train_eng, test_eng = X_train_eng[common_cols], test_eng[common_cols]

# Scale with RobustScaler (handles outliers better than StandardScaler)
numeric_cols = X_train_eng.select_dtypes(include=[np.number]).columns.tolist()
scaler = RobustScaler(quantile_range=(10, 90))
X_train_scaled = X_train_eng[numeric_cols].copy()
X_train_scaled[:] = scaler.fit_transform(X_train_eng[numeric_cols])
test_scaled = test_eng[numeric_cols].copy()
test_scaled[:] = scaler.transform(test_eng[numeric_cols])

# Time-based validation split (critical - no random shuffling for time-series)
split_idx = int(len(X_train_scaled) * 0.8)
X_tr, X_val = X_train_scaled.iloc[:split_idx], X_train_scaled.iloc[split_idx:]
y_tr, y_val = y_train.iloc[:split_idx], y_train.iloc[split_idx:]

# Train LightGBM
lgb_params = {
    'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'mape',
    'num_leaves': 128, 'learning_rate': 0.05, 'feature_fraction': 0.8,
    'bagging_fraction': 0.8, 'bagging_freq': 5, 'verbose': -1,
    'min_child_samples': 20, 'reg_alpha': 0.1, 'reg_lambda': 0.1,
}

train_data = lgb.Dataset(X_tr, label=y_tr)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

model = lgb.train(
    lgb_params, train_data, valid_sets=[val_data],
    num_boost_round=2000,
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(100)],
)

# Validate with SMAPE
val_pred = model.predict(X_val)
smape = 100 * np.mean(
    2 * np.abs(y_val.values - val_pred) / (np.abs(y_val.values) + np.abs(val_pred))
)
print(f"Validation SMAPE: {smape:.2f}%")

# Retrain on full dataset
full_model = lgb.train(
    lgb_params, lgb.Dataset(X_train_scaled, label=y_train), num_boost_round=1000
)

# Predict and post-process
predictions = full_model.predict(test_scaled)
predictions = np.maximum(predictions, 10000)  # Floor at reasonable minimum
predictions = pd.Series(predictions).rolling(5, center=True, min_periods=1).mean().values
predictions = np.clip(predictions, y_train.quantile(0.005), y_train.quantile(0.995))

# Save submission
submission = pd.DataFrame(index=test.index, data={'liquidity_target': predictions})
submission_out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_out)
print(f"Submission saved: {len(submission)} predictions")
```
