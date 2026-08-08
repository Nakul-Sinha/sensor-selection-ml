# Dataset

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
DATASET DESCRIPTION
Docs
Kaggle Notebook Upvote Dataset
Overview

The dataset is derived from the Kaggle Success Factors Dataset 100k. It contains 62,733 notebook records with metadata like code lines, markdown ratio, visualization count, libraries used, as well as author information like author tier, followers, and notebook count. The target variable is the upvote count (upvotes).

The dataset was filtered to only include notebook records (not dataset records) since these represent distinct prediction problems. Several features that could leak the target variable (e.g., number of comments, number of forks) were removed.

File Structure
train.csv — Training set with labels (56,762 rows)
test.csv — Test set without upvote column (6,971 rows)
sample_submission.csv — Example submission with random values
Features
Column	Type	Description
content_id	string	Unique notebook identifier
title	string	Notebook title (free text)
author_username	string	Author's Kaggle username
author_tier	categorical	Novice / Contributor / Expert / Master / Grandmaster
author_followers	integer	Number of followers
author_notebooks_count	integer	Notebooks published by author
author_datasets_count	integer	Datasets published by author
primary_topic	categorical	Primary topic (50 categories)
all_topics	string	Pipe-separated topic list
programming_language	categorical	Python / R / Julia / SQL
is_competition_related	boolean	Whether related to a competition
created_date	date	Creation date (dd/mm/yyyy)
last_updated	date	Last update date (dd/mm/yyyy)
days_since_creation	integer	Days since creation
update_count	integer	Number of updates
execution_time_seconds	integer	Execution time in seconds
code_lines	integer	Number of code lines
markdown_ratio	float	Ratio of markdown cells
visualization_count	integer	Number of visualizations
libraries_used	string	Pipe-separated library list
uses_gpu	boolean	Whether notebook uses GPU
file_size_mb	float	File size in MB
Data Characteristics
Missing values: usability_score, file_format, column_count, row_count, license_type are entirely null (these are dataset-only fields)
Target distribution: Right-skewed — most notebooks have few upvotes, with a long tail of highly-upvoted notebooks (median 1, mean 26, max 10,469)
Text feature: The title column is free text requiring NLP preprocessing
Concatenated categoricals: all_topics and libraries_used contain pipe-separated values
Date features: created_date and last_updated are in dd/mm/yyyy format

The dataset description documents the diverse feature types (numerical, categorical, date, pipe-separated, text) and explicitly calls out which columns are entirely null. This helps participants plan their preprocessing pipeline upfront rather than discovering issues during modeling.

FILE STRUCTURE
Docs
Files
raw/data.csv
15.2 MB
PREPARE SCRIPT
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
from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import random
random.seed(42)
def prepare(raw: Path, public: Path, private: Path) -> None:
    data = pd.read_csv(raw / "data.csv")
    # Only use notebook records (dataset records are a distinct prediction problem)
    data = data[data.content_type == "notebook"]
    train, test = train_test_split(data, test_size=0.1, random_state=0)
    test_without_labels = test.drop(columns=["upvotes"])

The prepare script filters to notebook records only and uses a fixed random seed for reproducible splits. The test set has upvotes removed, preventing any label leakage.

Back to all examples
```
