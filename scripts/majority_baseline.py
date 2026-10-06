"""Score a majority-class baseline on the fixed local validation split."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import task_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/task11_subtask1_train.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/majority-baseline.json"))
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = json.loads(args.data.read_text(encoding="utf-8"))
    strata = [f"{int(row['validity'])}_{int(row['plausibility'])}" for row in records]
    train_rows, validation_rows = train_test_split(
        records,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=strata,
    )
    majority_label = int(
        np.bincount([int(row["validity"]) for row in train_rows], minlength=2).argmax()
    )
    labels = [int(row["validity"]) for row in validation_rows]
    plausibility = [int(row["plausibility"]) for row in validation_rows]
    predictions = [majority_label] * len(validation_rows)
    metrics = task_metrics(labels, predictions, plausibility)
    result = {
        "model": "majority-class baseline",
        "seed": args.seed,
        "train_examples": len(train_rows),
        "validation_examples": len(validation_rows),
        "predicted_label": majority_label,
        **metrics,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
