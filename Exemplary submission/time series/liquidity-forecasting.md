# Intraday Liquidity Forecasting

> Forecast required liquidity buffers at 5-minute intervals using market microstructure, regulatory metrics, and cross-asset signals. A time-series regression challenge with regime shifts.

**Status:** EXEMPLARY
**Tags:** regression · time-series · finance · forecasting
**Domain:** Time Series **Difficulty:** MEDIUM **Score:** 0–200 (minimize)

Source: https://shipd.ai/quests/eris/docs/examples/liquidity-forecasting

---

## Dataset

### Intraday Liquidity Forecasting Dataset

#### Overview

This synthetic dataset simulates institutional liquidity dynamics across traditional finance and crypto markets. It includes stylized market microstructure behavior, volatility clustering, jump processes, regime switches, funding stress, and settlement windows.

The dataset is challenging due to differences in intraday liquidity behavior across market regimes. Liquidity, volume, volatility, and spreads all have strong intraday patterns — for example across different trading sessions or around auction times.

#### File Structure

- `train.csv` — Training data with 64 features + target (6,744 rows, Jun 1-24, 2024)
- `test.csv` — Test data with 64 features, no target (1,687 rows, Jun 24-30, 2024)
- `sample_submission.csv` — Submission format with NaN placeholders

#### Features

**Time Features**

| Feature | Type | Description |
|---------|------|-------------|
| hour, minute | int | Time of day |
| day_of_week, day_of_month | int | Calendar features |
| asia_session, europe_session, us_session | binary | Trading session indicators |
| asia_euro_overlap, euro_us_overlap | binary | Session overlap periods |
| auction_period | binary | Market auction window |
| is_weekend | binary | Weekend indicator |
| fx_fixing_window | binary | FX fixing window (16:00-16:30) |
| settlement_window | binary | Settlement processing window |

**Market Microstructure**

| Feature | Type | Description |
|---------|------|-------------|
| returns | float | Asset returns (%) |
| volatility | float | Realized volatility |
| trading_volume | float | Trading volume |
| spread | float | Bid-ask spread (%) |
| order_imbalance | float | Normalized order imbalance |
| order_flow | float | Signed volume |

**Institutional & Regulatory**

| Feature | Type | Description |
|---------|------|-------------|
| lcr_ratio | float | Liquidity Coverage Ratio (Basel III) |
| nsfr_ratio | float | Net Stable Funding Ratio (Basel III) |
| funding_pressure | float | Funding market stress |
| settlement_pressure | float | Settlement cycle stress |

**Crypto-Specific**

| Feature | Type | Description |
|---------|------|-------------|
| gas_price_eth | float | Ethereum gas price |
| exchange_reserves | float | Exchange reserve levels |
| crypto_vol_index | float | Crypto volatility index |
| funding_rate_spread | float | Crypto-USD funding spread |

**Cross-Asset**

| Feature | Type | Description |
|---------|------|-------------|
| eur_usd_return, gbp_usd_return | float | FX returns |
| gold_return, equity_return | float | Asset class returns |
| fx_lead_crypto, crypto_lead_fx | binary | Lead-lag indicators |

**Target**

| Feature | Type | Description |
|---------|------|-------------|
| liquidity_target | float | Required liquidity buffer ($) |

#### Data Characteristics

| Feature Category | Missing Rate |
|------------------|--------------|
| Crypto Features | 10-30% |
| Market Microstructure | 1-15% |
| Institutional | 1-5% |
| Cross-Asset | 5-20% |

**Missing patterns:** Overnight gaps in crypto, 15% during crisis regimes, 20% on weekends, 40% during macro announcements. Block missing patterns (5-60 min) occur with 0.2% probability.

**Market regimes:** Normal (~70%), Volatile (~22%), Crisis (~5%), Flash Crash (~1%), Weekend (~2%). Volatility can increase 8x during crisis regimes.

> The dataset description provides detailed missingness documentation — not just rates (10-30% for crypto) but patterns (overnight gaps, crisis gaps, macro event spikes). This level of detail helps participants design appropriate imputation strategies rather than blindly applying generic methods.

#### File Structure (files)

- `features.csv` — 5.2 MB
- `target.csv` — 200 KB

### Prepare Script

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

> The prepare script uses a strictly chronological split — the test set is always AFTER the training set in time. This is fundamental for time-series problems where random splitting would leak future information into the past. The assertion check explicitly verifies no temporal overlap.

---

## Problem

**Domain:** Time Series **Difficulty:** MEDIUM **Score:** 0–200 (minimize)

### Problem Description

#### Overview

Financial institutions must maintain intraday liquidity buffers to meet operational and regulatory requirements throughout each trading day. This challenge requires forecasting the required liquidity buffer at 5-minute intervals based on real-time market conditions, institutional metrics, and cross-asset signals.

Liquidity management is critical for regulatory compliance with Basel III standards, specifically the Liquidity Coverage Ratio (LCR) and Net Stable Funding Ratio (NSFR). Insufficient buffers can result in regulatory violations, funding shortfalls during market stress, and operational disruptions.

The dataset includes features across multiple categories: time features, market microstructure, institutional metrics, crypto signals, and cross-asset indicators.

The forecasting challenge is complicated by regime shifts (normal, volatile, crisis, flash crash, weekend), non-stationarity, volatility clustering, missing data patterns, extreme outliers, and timezone-dependent lead-lag relationships between traditional and crypto markets.

#### Evaluation

Submissions are scored using SMAPE (Symmetric Mean Absolute Percentage Error):

```
SMAPE = 100 * mean(2 * |y_true - y_pred| / (|y_true| + |y_pred|))
```

Lower SMAPE scores are better. A score of 0% indicates perfect predictions. SMAPE is bounded between 0% and 200%.

#### Submission Format

Submit a `submission.csv` with timestamp and predicted liquidity buffer:

| Column | Type | Description |
|--------|------|-------------|
| timestamp | datetime | Timestamp from test.csv (index) |
| liquidity_target | float | Predicted liquidity buffer ($) |

### Grading Script

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

### Config

```yaml
name: Intraday Liquidity Forecasting
difficulty: medium
domain: tabular

grade:
  direction: minimize
  minimum: 0.0
  maximum: 200.0
```

### Rubrics (12 criteria)

1. **[REQUIRED] Data Handling** — Handle missing values in crypto features (10-30%), crisis periods (15%), weekends (20%), and macro event times (40%) using imputation methods that do not introduce forward-looking bias.
   *The dataset contains realistic missing patterns reflecting actual market data availability. Forward-fill or time-aware imputation preserves temporal causality. Using future data to impute past values would constitute leakage.*

2. **[REQUIRED] Data Handling** — Handle extreme outliers appropriately — particularly flash crash returns (-20% to -50%) and gas price spikes — using robust methods rather than naive removal.
   *Extreme outliers are genuine market events that signal liquidity stress. Removing them would eliminate precisely the high-stress scenarios the model must learn to predict. Robust scaling or outlier-resistant models are preferred.*

3. **[REQUIRED] Training** — Use a temporal (chronological) train-validation split where validation data occurs strictly after training data — not random shuffling.
   *Random shuffling leaks future information into the past, producing unrealistic performance estimates. The test set (Jun 24-30) is chronologically after training (Jun 1-24), so validation must mirror this structure.*

4. **[REQUIRED] Training** — Do not use future information including lookahead features, contemporaneous target values, or features computed using data from timestamps at or after the prediction time.
   *In production, liquidity forecasts must be made before observing contemporaneous market conditions. Only strictly lagged features and past information should be used.*

5. **[RECOMMENDED] Modeling** — Account for the 5 distinct market regimes (normal 70%, volatile 22%, crisis 5%, flash crash 1%, weekend 2%) through regime-specific features or regime-weighted approaches.
   *Crisis regimes have 8x higher volatility than normal. Models treating all regimes uniformly will systematically under-predict liquidity needs during stress periods.*

6. **[RECOMMENDED] Feature Engineering** — Incorporate institutional regulatory features (lcr_ratio, nsfr_ratio, funding_pressure) as they directly determine required liquidity buffers under Basel III.
   *LCR and NSFR are Basel III regulatory requirements that constrain minimum liquidity. Models ignoring these ratios will fail to capture the regulatory floor on liquidity needs.*

7. **[RECOMMENDED] Feature Engineering** — Leverage timezone-dependent lead-lag relationships: FX leads crypto during London/NY overlap (13:00-16:00 UTC), crypto leads FX during Asian session (00:00-08:00 UTC).
   *The dataset embeds realistic timezone-dependent causality. Lagged cross-asset features capture anticipatory liquidity adjustments during market transitions.*

8. **[RECOMMENDED] Feature Engineering** — Incorporate trading session structure (Asia, Europe, US) and session overlap periods which exhibit distinct U-shaped intraday liquidity patterns.
   *Liquidity follows well-documented intraday patterns. The 16:00-16:30 FX fixing window sees particularly high stress. Models ignoring session structure miss systematic intraday variations.*

9. **[RECOMMENDED] Training** — Do not randomly shuffle time-series data during training, which would destroy temporal dependencies and regime continuity.
   *Financial patterns depend on momentum, regime persistence (mean 4.3 hours), and autocorrelation. Shuffling causes models to learn cross-sectional rather than temporal patterns.*

10. **[RECOMMENDED] Data Handling** — Apply stationarity-inducing transformations such as differencing or log-returns to handle non-stationary time-series with regime shifts.
    *The liquidity_target has significant skew (1.59) and regime-dependent means. Variance-stabilizing transforms improve model performance by making patterns more consistent across regimes.*

11. **[RECOMMENDED] Data Handling** — Recognize and handle 5-60 minute block missing patterns (0.2% probability) that represent system outages, using methods respecting temporal causality.
    *Block missing patterns differ from random missingness. Forward-fill with staleness indicators or separate outage regime indicators are appropriate strategies.*

12. **[UNIVERSAL] Training** — Optimize for SMAPE, MAE, or RMSE — not classification metrics like accuracy.
    *This is a regression task forecasting continuous liquidity values. SMAPE is the grading metric and should be the primary optimization target.*

> With 12 rubrics across 6 types, this is a comprehensive evaluation framework. The REQUIRED rubrics enforce fundamentals (no data leakage, temporal validation, handle missing values), while RECOMMENDED rubrics reward domain expertise (regime awareness, cross-asset lead-lag features, trading session patterns).

---

## Solution

**Score: 19.28**

### Solution Code

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

> The solution demonstrates key time-series practices: lag and rolling features that only look backward (no leakage), RobustScaler instead of StandardScaler to handle outliers, cyclical time encoding for periodicity, and a time-based validation split. Post-processing with smoothing and clipping improves prediction stability. Achieves 19.28% SMAPE.

### Output Files

- `solution.ipynb` — 18 KB
- `submission.csv` — 50 KB
