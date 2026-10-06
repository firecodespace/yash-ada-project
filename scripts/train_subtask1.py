"""Fine-tune a sequence classifier on Task 11 Subtask 1."""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.metrics import task_metrics


class SyllogismDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records = records
        self.encodings = tokenizer(
            [record["syllogism"] for record in records],
            truncation=True,
            padding=True,
            max_length=max_length,
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[index]) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(int(self.records[index]["validity"]))
        return item


def read_records(path: Path) -> list[dict]:
    records = json.loads(path.read_text(encoding="utf-8"))
    required = {"syllogism", "validity", "plausibility"}
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty JSON array in {path}")
    for index, record in enumerate(records):
        if not required.issubset(record):
            raise ValueError(f"Record {index} is missing fields: {required - record.keys()}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/raw/task11_subtask1_train.json"))
    parser.add_argument("--model", default="xlm-roberta-base")
    parser.add_argument("--output", type=Path, default=Path("outputs/subtask1-xlm-roberta"))
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    records = read_records(args.data)

    # Preserve all four validity/plausibility groups in both splits.
    strata = [f"{int(row['validity'])}_{int(row['plausibility'])}" for row in records]
    train_rows, validation_rows = train_test_split(
        records,
        test_size=args.validation_size,
        random_state=args.seed,
        stratify=strata,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)
    train_dataset = SyllogismDataset(train_rows, tokenizer, args.max_length)
    validation_dataset = SyllogismDataset(validation_rows, tokenizer, args.max_length)
    validation_plausibility = np.asarray([int(row["plausibility"]) for row in validation_rows])

    def compute_metrics(evaluation):
        logits, labels = evaluation
        predictions = np.argmax(logits, axis=-1)
        return task_metrics(labels, predictions, validation_plausibility)

    training_args = TrainingArguments(
        output_dir=str(args.output / "checkpoints"),
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=20,
        report_to="none",
        seed=args.seed,
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    pretrained_metrics = trainer.evaluate()
    trainer.train()
    finetuned_metrics = trainer.evaluate()
    args.output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.output / "model"))
    tokenizer.save_pretrained(str(args.output / "model"))
    (args.output / "metrics.json").write_text(
        json.dumps({
            "model": args.model,
            "seed": args.seed,
            "train_examples": len(train_rows),
            "validation_examples": len(validation_rows),
            "pretrained_validation": pretrained_metrics,
            "finetuned_validation": finetuned_metrics,
        }, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({
        "pretrained_validation": pretrained_metrics,
        "finetuned_validation": finetuned_metrics,
    }, indent=2))


if __name__ == "__main__":
    main()
