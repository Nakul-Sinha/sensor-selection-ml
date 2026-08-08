from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
import random

random.seed(42)


def prepare(raw: Path, public: Path, private: Path) -> None:
    data = pd.read_csv(raw / "data.csv")

    # Only use notebook records (dataset records are a distinct prediction problem)
    data = data[data.content_type == "notebook"]

    train, test = train_test_split(data, test_size=0.1, random_state=0)
    test_without_labels = test.drop(columns=["upvotes"])

    train.to_csv(public / "train.csv", index=False)
    test_without_labels.to_csv(public / "test.csv", index=False)

    # Private answers
    test = test[["content_id", "upvotes"]]
    test.to_csv(private / "answers.csv", index=False)

    # Sample submission with random values
    sample_submission = test.copy()
    sample_submission["upvotes"] = [
        random.randint(0, 1000) for _ in range(len(sample_submission))
    ]
    sample_submission.to_csv(public / "sample_submission.csv", index=False)