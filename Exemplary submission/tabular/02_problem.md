# Problem

```text
Notebook Upvote Prediction
Notebook Upvote Prediction

Predict Kaggle notebook upvote counts from metadata including author stats, code metrics, and text features. A tabular regression problem with diverse feature types.

EXEMPLARY
regression
tabular
feature-engineering
text
Dataset
Problem
Solution
Tabular
MEDIUM
Score: 0–999 (minimize)
PROBLEM DESCRIPTION
Docs
Overview

The objective is to predict a Kaggle notebook's upvote count based on various features related to the notebook and its author. Upvotes are a key metric of quality and popularity on Kaggle. Correctly predicting upvote counts can help understand what makes a notebook successful.

This dataset includes numerical, categorical, date, and text features, designed to challenge participants to use features effectively and creatively.

Evaluation

Your goal is to predict the upvote count. Submissions are evaluated using a modified mean absolute error (MAE) in log space:

MAE = min(mean(|log(y_i + 1) - log(ŷ_i + 1)| for all i), 999)

where y_i is the true upvote count and ŷ_i is the predicted upvote count. The log transformation handles the right-skewed distribution and ensures the metric isn't dominated by outliers.

Submission

Submit a CSV file with content_id and the predicted upvote count:

content_id	upvotes
nb_eafe3010	100
nb_05570170	200
nb_699ec80a	300

The problem description explains WHY log transformation is used — handling right-skewed distribution — not just that it's used. This context helps participants choose appropriate model targets and understand the evaluation metric.

GRADING SCRIPT
Docs
python
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
import numpy as np
import pandas as pd
def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    prepared_data = prepare_for_metric(submission, answers)
    mae_log_score = get_mae_log_score(
        prepared_data['submission'], prepared_data['answers']
    )
    return min(mae_log_score, 999)
def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """Validate and prepare submission and answers DataFrames for scoring."""
    if set(submission.columns) != {"content_id", "upvotes"}:
        raise Exception(
            "Submission DataFrame must have 'content_id' and 'upvotes' columns."
CONFIG
Docs
yaml
1
2
3
4
5
6
7
8
name: Notebook Upvote Prediction
difficulty: medium
domain: tabular
grade:
  direction: minimize
  minimum: 0
  maximum: 999
RUBRICS (11 CRITERIA)
Docs
REQUIRED
Data Handling

Identify and remove columns that contain target leakage, such as fork_count, views, downloads, comments_count, notebook_usage, medal, is_featured, is_trending, engagement_rate, virality_score, quality_score. These represent future information not available at notebook creation time.

Target leakage occurs when features contain information not available at prediction time. These columns represent engagement metrics that accumulate after publication, artificially inflating model performance but failing in real-world deployment.

REQUIRED
Data Handling

Use log transformation on the upvote count to handle the right-skewed distribution and match the evaluation metric.

The upvote count is right-skewed, so log transformation matches the evaluation metric and helps the model handle outliers. Training on raw upvote counts would bias the model toward the heavy right tail.

REQUIRED
Data Handling

Split all_topics and libraries_used columns on the pipe delimiter and encode the resulting multi-label values into one-hot or ordinal encoding.

These columns contain multiple categories concatenated with "|". Directly encoding the raw strings would treat each unique combination as a separate category, losing the individual signal from each topic or library.

RECOMMENDED
Agent Behavior

Remove features that are entirely null or irrelevant to notebook upvote prediction, such as usability_score, file_format, column_count, row_count, license_type.

These features are dataset-specific fields with 100% null values for notebook records. Including them wastes computation and could introduce noise.

RECOMMENDED
Data Handling

Convert date features (created_date, last_updated) from dd/mm/yyyy format into numerical representations that preserve temporal information.

Gradient boosting models and neural networks cannot effectively use raw date strings. Converting to days-since-epoch or similar numerical representations preserves temporal ordering.

RECOMMENDED
Data Handling

Handle the text title column using TF-IDF, embeddings, or similar NLP preprocessing.

The title is a strong signal — it's the first thing users see when browsing notebooks. Effective text preprocessing can capture patterns in titles that correlate with upvotes.

RECOMMENDED
Modeling

Implement a gradient boosting model (e.g., XGBoost, LightGBM) for this regression task.

Gradient boosting models are effective for tabular data with mixed feature types and can handle complex non-linear relationships.

RECOMMENDED
Feature Engineering

Create feature interaction terms (e.g., days_since_creation / update_count for average update frequency) to capture cross-feature relationships.

Tree-based models handle interactions less efficiently than neural networks. Manually creating semantically meaningful interactions helps the model capture important patterns.

REQUIRED
Modeling

Model does not show significant overfitting — the gap between training and validation performance is reasonable.

A large train-validation gap indicates the model memorizes training data rather than learning generalizable patterns.

RECOMMENDED
Modeling

Perform hyperparameter search (grid search or Optuna) to optimize model configuration.

Default hyperparameters are rarely optimal. Systematic tuning can significantly improve performance on the specific dataset characteristics.

RECOMMENDED
Modeling

Clip final predictions to ensure non-negative upvote counts.

Upvote counts are strictly non-negative. Models trained in log space can produce negative predictions when converted back, so clipping to zero is necessary.

The rubrics test both technical skills (leakage detection, feature engineering for concatenated categoricals) and practical ML knowledge (log transformation, date handling). Each criterion is specific and verifiable — not vague qualities like "good data handling".

Back to all examples
```
