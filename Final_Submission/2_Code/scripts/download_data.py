"""Download the official Task 11 Subtask 1 training data."""

import argparse
from pathlib import Path

import requests


DATA_URL = (
    "https://raw.githubusercontent.com/neuro-symbolic-ai/"
    "semeval_2026_task_11/main/train_data/subtask%201/train_data.json"
)
DEFAULT_DESTINATION = Path("data/raw/task11_subtask1_train.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_DESTINATION)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(DATA_URL, timeout=60)
    response.raise_for_status()
    args.output.write_bytes(response.content)
    print(f"Saved official training data to {args.output} ({len(response.content):,} bytes)")


if __name__ == "__main__":
    main()

