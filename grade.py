import numpy as np
import pandas as pd


ID_COLUMN = "id"
ANSWER_COLUMN = "answer"
VALID_ANSWERS = {"A", "B", "C", "D", "E"}
GROUP_COLUMNS = [
    "query_type",
    "split_group",
    "ood_axis",
    "layout_family",
    "lighting_condition",
    "clutter_level",
    "difficulty_tier",
]


def _accuracy(frame: pd.DataFrame) -> float:
    if len(frame) == 0:
        return 0.0
    return float((frame["answer_true"] == frame["answer_pred"]).mean())


def _worst_group_accuracy(frame: pd.DataFrame, group_col: str) -> float:
    scores = []
    for _, group in frame.groupby(group_col):
        scores.append(_accuracy(group))
    return float(min(scores)) if scores else 0.0


def grade(submission: pd.DataFrame, answers: pd.DataFrame) -> float:
    expected_columns = [ID_COLUMN, ANSWER_COLUMN]
    if list(submission.columns) != expected_columns:
        raise Exception("submission must contain exactly two columns: id, answer")
    required_answer_cols = {ID_COLUMN, ANSWER_COLUMN, *GROUP_COLUMNS}
    if not required_answer_cols.issubset(answers.columns):
        raise Exception(f"answers missing required columns: {sorted(required_answer_cols - set(answers.columns))}")

    if submission[ID_COLUMN].isna().any():
        raise Exception("submission contains missing ids")
    if submission[ID_COLUMN].duplicated().any():
        raise Exception("submission contains duplicate ids")
    if set(submission[ID_COLUMN]) != set(answers[ID_COLUMN]):
        raise Exception("submission ids must exactly match the private test ids")
    if submission[ANSWER_COLUMN].isna().any():
        raise Exception("submission contains missing answers")
    unknown = set(submission[ANSWER_COLUMN]) - VALID_ANSWERS
    if unknown:
        raise Exception(f"answers must be one of A, B, C, D, E; found {sorted(unknown)}")

    aligned = answers.merge(
        submission,
        on=ID_COLUMN,
        suffixes=("_true", "_pred"),
        validate="one_to_one",
    )
    overall = _accuracy(aligned)
    score = (
        0.50 * overall
        + 0.18 * _worst_group_accuracy(aligned, "query_type")
        + 0.10 * _worst_group_accuracy(aligned, "split_group")
        + 0.07 * _worst_group_accuracy(aligned, "ood_axis")
        + 0.05 * _worst_group_accuracy(aligned, "layout_family")
        + 0.03 * _worst_group_accuracy(aligned, "lighting_condition")
        + 0.02 * _worst_group_accuracy(aligned, "clutter_level")
        + 0.05 * _worst_group_accuracy(aligned, "difficulty_tier")
    )
    return float(np.clip(score, 0.0, 1.0))
