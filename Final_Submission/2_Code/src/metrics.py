"""Metrics for Task 11 Subtask 1 validation experiments."""

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def task_metrics(
    labels: Sequence[int], predictions: Sequence[int], plausibility: Sequence[int]
) -> dict[str, float]:
    """Compute accuracy, macro-F1, and a transparent content-effect estimate.

    Labels use 1 for valid and 0 for invalid. Plausibility uses 1 for plausible
    and 0 for implausible. Content effect is the mean absolute accuracy gap
    across the two within-plausibility and two within-validity comparisons.
    """
    labels_array = np.asarray(labels, dtype=int)
    predictions_array = np.asarray(predictions, dtype=int)
    plausibility_array = np.asarray(plausibility, dtype=int)

    accuracies: dict[tuple[int, int], float] = {}
    for validity in (0, 1):
        for plausible in (0, 1):
            mask = (labels_array == validity) & (plausibility_array == plausible)
            if mask.any():
                accuracies[(validity, plausible)] = float(
                    accuracy_score(labels_array[mask], predictions_array[mask])
                )

    gaps: list[float] = []
    for plausible in (0, 1):
        pair = [accuracies.get((validity, plausible)) for validity in (0, 1)]
        if all(value is not None for value in pair):
            gaps.append(abs(pair[0] - pair[1]))
    for validity in (0, 1):
        pair = [accuracies.get((validity, plausible)) for plausible in (0, 1)]
        if all(value is not None for value in pair):
            gaps.append(abs(pair[0] - pair[1]))

    content_effect = float(np.mean(gaps)) if gaps else float("nan")
    return {
        "accuracy": float(accuracy_score(labels_array, predictions_array)),
        "macro_f1": float(f1_score(labels_array, predictions_array, average="macro")),
        "content_effect_estimate": content_effect,
        "task_score_estimate": float(
            accuracy_score(labels_array, predictions_array)
            / (1.0 + np.log1p(content_effect))
        ) if np.isfinite(content_effect) else float("nan"),
    }

