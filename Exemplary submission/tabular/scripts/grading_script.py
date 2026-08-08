import numpy as np
import pandas as pd


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    prepared_data = prepare_for_metric(submission, answers)
    mae_log_score = get_mae_log_score(
        prepared_data['submission'], prepared_data['answers']
    )
    return min(mae_log_score, 999)


def prepare_for_metric(submission: pd.DataFrame, answers: pd.DataFrame) -> dict:
    """Validate and prepare submission and answers DataFrames for scoring."""
    if set(submission.columns) != {"content_id", "upvotes"}:
        raise Exception(
            "Submission DataFrame must have 'content_id' and 'upvotes' columns."
        )

    if len(submission) != len(answers):
        raise Exception(
            "Submission and answers DataFrames must have the same number of rows."
        )

    if not (submission['upvotes'] >= 0).all():
        raise Exception(
            "All upvote predictions in submission DataFrame must be non-negative."
        )

    assert set(answers.columns) == {"content_id", "upvotes"}, (
        "Answers DataFrame must have 'content_id' and 'upvotes' columns."
    )

    submission_sorted = submission.sort_values(by="content_id").reset_index(drop=True)
    answers_sorted = answers.sort_values(by="content_id").reset_index(drop=True)

    assert (
        submission_sorted["content_id"].tolist()
        == answers_sorted["content_id"].tolist()
    ), (
        f"Mismatch in content_ids between submission and answers. "
        f"Mismatched: {len(set(submission_sorted['content_id']) ^ set(answers_sorted['content_id']))}."
    )

    return {"submission": submission_sorted, "answers": answers_sorted}


def get_mae_log_score(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    submission_upvotes = submission['upvotes'].astype(float).values
    answers_upvotes = answers['upvotes'].astype(float).values
    return np.mean(
        np.abs(np.log(submission_upvotes + 1) - np.log(answers_upvotes + 1))
    )