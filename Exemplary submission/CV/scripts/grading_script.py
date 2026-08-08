import math
import pandas as pd


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    answers = answers[["patient_id", "volume"]].copy()
    submission = submission[["patient_id", "volume"]].copy()

    answers["patient_id"] = answers["patient_id"].astype(str).str.strip()
    submission["patient_id"] = submission["patient_id"].astype(str).str.strip()

    answers["volume"] = pd.to_numeric(answers["volume"], errors="coerce")
    submission["volume"] = pd.to_numeric(submission["volume"], errors="coerce")

    gt_map = dict(zip(answers["patient_id"], answers["volume"]))
    pred_map = dict(zip(submission["patient_id"], submission["volume"]))

    patient_ids = sorted(set(gt_map.keys()) | set(pred_map.keys()))

    eps = 1e-4
    scores = []

    for pid in patient_ids:
        g = gt_map.get(pid, None)
        p = pred_map.get(pid, None)

        if g is None or p is None or pd.isna(g) or pd.isna(p):
            raise ValueError(f"Missing or non-numeric volume for patient: {pid}")

        g = float(g)
        p = float(p)

        if (not math.isfinite(g)) or (not math.isfinite(p)) or g < 0 or p < 0:
            raise ValueError(f"Invalid volume for patient: {pid}")

        scores.append(1.0 - (abs(g - p) / (g + p + eps)))

    return float(sum(scores) / len(scores)) if scores else 0.0