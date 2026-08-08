import sys
from pathlib import Path

import pandas as pd
import numpy as np
import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, MultiLabelBinarizer, OrdinalEncoder
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

public_dir = Path(sys.argv[1])
submission_out = Path(sys.argv[2])

# Load data
train = pd.read_csv(public_dir / 'train.csv')
test = pd.read_csv(public_dir / 'test.csv')

# Remove target leakage columns (engagement metrics not available at creation time)
LEAKAGE_COLUMNS = [
    'fork_count', 'views', 'downloads', 'comments_count', 'notebook_usage',
    'medal', 'is_featured', 'is_trending', 'engagement_rate',
    'virality_score', 'quality_score',
]
leakage_present = [col for col in LEAKAGE_COLUMNS if col in train.columns]
train = train.drop(columns=leakage_present)
test = test.drop(columns=leakage_present)

# Drop entirely-null and irrelevant columns
drop_cols = ['usability_score', 'file_format', 'column_count',
             'row_count', 'license_type', 'content_type']
train = train.drop(columns=[c for c in drop_cols if c in train.columns])
test = test.drop(columns=[c for c in drop_cols if c in test.columns])

# Define feature types
numerical_columns = [
    "author_followers", "author_notebooks_count", "author_datasets_count",
    "days_since_creation", "update_count", "execution_time_seconds",
    "file_size_mb", "markdown_ratio", "visualization_count", "code_lines",
]
date_columns = ["created_date", "last_updated"]
categorical_columns = [
    "author_tier", "primary_topic", "programming_language",
    "is_competition_related", "uses_gpu",
]
concat_categorical = ["all_topics", "libraries_used"]

# Train/validation split
train, valid = train_test_split(train, test_size=0.2, random_state=42)

# Standardize numerical features
scaler = StandardScaler()
train[numerical_columns] = scaler.fit_transform(train[numerical_columns])
valid[numerical_columns] = scaler.transform(valid[numerical_columns])
test[numerical_columns] = scaler.transform(test[numerical_columns])

# One-hot encode categorical features
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoded_train = encoder.fit_transform(train[categorical_columns])
encoded_valid = encoder.transform(valid[categorical_columns])
encoded_test = encoder.transform(test[categorical_columns])

for df, enc in [(train, encoded_train), (valid, encoded_valid), (test, encoded_test)]:
    enc_df = pd.DataFrame(enc, columns=encoder.get_feature_names_out(), index=df.index)
    df.drop(columns=categorical_columns, inplace=True)
    df[enc_df.columns] = enc_df

# Convert date features to days since epoch
ref = datetime.datetime(1970, 1, 1)
for col in date_columns:
    for df in [train, valid, test]:
        df[col] = (pd.to_datetime(df[col]) - ref).dt.days

# Handle pipe-separated categorical columns
for col in concat_categorical:
    mlb = MultiLabelBinarizer()
    train_split = train[col].fillna('').str.split('|')
    valid_split = valid[col].fillna('').str.split('|')
    test_split = test[col].fillna('').str.split('|')

    enc_tr = pd.DataFrame(
        mlb.fit_transform(train_split),
        columns=[f"{col}_{l}" for l in mlb.classes_], index=train.index
    )
    enc_va = pd.DataFrame(
        mlb.transform(valid_split),
        columns=[f"{col}_{l}" for l in mlb.classes_], index=valid.index
    )
    enc_te = pd.DataFrame(
        mlb.transform(test_split),
        columns=[f"{col}_{l}" for l in mlb.classes_], index=test.index
    )
    train = pd.concat([train.drop(columns=[col]), enc_tr], axis=1)
    valid = pd.concat([valid.drop(columns=[col]), enc_va], axis=1)
    test = pd.concat([test.drop(columns=[col]), enc_te], axis=1)

# TF-IDF on title feature
tfidf = TfidfVectorizer(max_features=256, stop_words='english')
for df, tfidf_data in [
    (train, tfidf.fit_transform(train['title'].fillna(''))),
    (valid, tfidf.transform(valid['title'].fillna(''))),
    (test, tfidf.transform(test['title'].fillna(''))),
]:
    tfidf_df = pd.DataFrame(
        tfidf_data.toarray(),
        columns=[f"title_tfidf_{w}" for w in tfidf.get_feature_names_out()],
        index=df.index,
    )
    df.drop(columns=['title'], inplace=True)
    df[tfidf_df.columns] = tfidf_df

# Ordinal encode author_username
author_enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
train['author_username'] = author_enc.fit_transform(train[['author_username']])
valid['author_username'] = author_enc.transform(valid[['author_username']])
test['author_username'] = author_enc.transform(test[['author_username']])

# Drop ID column
for df in [train, valid, test]:
    df.drop(columns=["content_id"], inplace=True, errors='ignore')

# Train RandomForest on log-transformed target (matches evaluation metric)
X_train = train.drop(columns=['upvotes'])
y_train_log = np.log1p(train['upvotes'])
X_val = valid.drop(columns=['upvotes'])
y_val_log = np.log1p(valid['upvotes'])

rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train_log)

# Validate
val_pred_log = rf.predict(X_val)
mae_log = mean_absolute_error(y_val_log, val_pred_log)
print(f"Validation MAE (log space): {mae_log:.4f}")

# Predict on test and convert back from log space
test_pred = np.maximum(np.expm1(rf.predict(test)), 0)

# Save submission
test_original = pd.read_csv(public_dir / "test.csv")
submission = pd.DataFrame({
    'content_id': test_original['content_id'],
    'upvotes': test_pred,
})
submission_out.parent.mkdir(parents=True, exist_ok=True)
submission.to_csv(submission_out, index=False)
print(f"Submission saved: {len(submission)} predictions")