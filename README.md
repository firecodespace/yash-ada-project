# SemEval 2026 Task 11: Content-Independent Syllogistic Reasoning

This project studies whether fine-tuning an open multilingual model helps it judge the **formal validity** of syllogisms, independently of whether their claims sound plausible in the real world.

## Research question

Can a small fine-tuned model improve validity accuracy while reducing the content effect—the tendency to let plausibility influence its answer?

## Initial scope

The first milestone implements Task 11 Subtask 1: binary validity classification on the official English training data. It fine-tunes `xlm-roberta-base` and reports accuracy, macro-F1, and a validation estimate of content effect. The later multilingual extension can evaluate transfer to the official Subtask 3 languages: German, Spanish, French, Italian, Dutch, Portuguese, Russian, Chinese, Swahili, Bengali, and Telugu.

The official training set is English only. The competition evaluation phase has ended; the released data and evaluation materials remain useful for a reproducible course project. We will treat a stronger result as an experimental objective, not a guaranteed outcome.

The released Subtask 1 training file contains 960 examples, balanced across the four combinations of validity and plausibility (240 examples each).

## Setup

Python 3.10+ is recommended. Create an environment, install the requirements, and download the official data:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_data.py
```

Fine-tune the starter model:

```bash
python scripts/majority_baseline.py
python scripts/train_subtask1.py
```

For Colab, upload/clone this repository, install `requirements.txt`, enable a GPU runtime, and run the same two Python commands in notebook cells. Training automatically uses mixed precision when CUDA is available. To change model or training settings:

```bash
python scripts/train_subtask1.py --model xlm-roberta-base --epochs 4 --batch-size 8 --seed 42
```

The majority baseline is saved to `outputs/majority-baseline.json`. The fine-tuned model and matched pretrained/fine-tuned metrics are saved under `outputs/subtask1-xlm-roberta/`. Raw data and generated outputs are git-ignored.

## Project workflow

1. Establish the majority and untuned-model baselines.
2. Fine-tune with a fixed, stratified train/validation split.
3. Compare multiple seeds and model/training choices.
4. Inspect performance by plausibility group and classify error patterns.
5. Extend the best controlled experiment to multilingual Task 11 Subtask 3.

Do not use the official test labels to tune hyperparameters. Record every run in [RESULTS.md](RESULTS.md). The in-repo content-effect value is an estimate for development; use the official Task 11 evaluation kit for final task scores and confirm its exact ranking calculation before reporting competition-style results.

## Repository layout

```text
data/raw/                 Downloaded official data (not committed)
outputs/                  Checkpoints and run metrics (not committed)
scripts/download_data.py  Fetch the official Subtask 1 training JSON
scripts/train_subtask1.py Fine-tune and evaluate a sequence classifier
src/metrics.py            Accuracy, macro-F1, and content-effect estimate
RESULTS.md                Experiment log
```

## References

- [SemEval-2026 Task 11 overview](https://sites.google.com/view/semeval-2026-task-11)
- [Official dataset and evaluation repository](https://github.com/neuro-symbolic-ai/semeval_2026_task_11)
