# Can a model learn to reason past what sounds plausible?

This is an ADA class project based on **SemEval 2026 Task 11**. We fine-tune a language model to decide whether a syllogism is logically valid. The key idea is to judge the logic of the statements, even when their content sounds strange or unrealistic.

For example, these premises may sound unrealistic:

> All planets are dogs. All dogs are animals. Therefore, all planets are animals.

The argument is logically valid: if both premises are true, the conclusion must follow. A model should judge the structure of the argument, not whether planets really are dogs.

## What we are trying to find out

Does fine-tuning help a model predict logical validity, and does it make the model less influenced by whether an argument sounds plausible?

The project currently covers **Subtask 1**, which asks for a true/false validity prediction in English. The official training file has 960 examples. Each example includes the syllogism, its validity label, and a plausibility label. We use the plausibility label to measure whether plausibility appears to affect predictions.

The SemEval competition period has ended, but its released dataset and evaluation materials can still be used for this course project.

## What we built

We compare three starting points:

1. **Majority baseline:** always predicts the most common validity label in the training portion.
2. **Pretrained XLM-R:** uses `xlm-roberta-base` before task-specific training.
3. **Fine-tuned XLM-R:** trains that model on the Task 11 examples, then evaluates it on held-out examples.

The dataset is split into 80% training and 20% validation data. The split is stratified so all four validity/plausibility combinations are represented. The training script measures the pretrained and fine-tuned model on the same validation examples and saves the model and metrics.

## Current results

The first two fine-tuning runs improved over their corresponding pretrained-model validation results:

| Run | Seed | Pretrained accuracy | Fine-tuned accuracy | Fine-tuned macro-F1 | Content-effect estimate |
|---|---:|---:|---:|---:|---:|
| 1 | 42 | 50.0% | 60.9% | 0.608 | 0.109 |
| 2 | 123 | 50.0% | 75.5% | 0.755 | 0.070 |

Each validation set contains 192 examples. Changing the seed also changes the train/validation split, so these two runs are **not a controlled comparison with each other**. The numbers show promising initial results, not a final claim about performance on unseen test data. See [RESULTS.md](RESULTS.md) for the experiment log.

### What the metrics mean

- **Accuracy:** fraction of validation answers predicted correctly.
- **Macro-F1:** averages performance on valid and invalid examples, so one class cannot dominate the score.
- **Content-effect estimate:** a local estimate of how much accuracy differs across plausibility and validity groups. Lower is better. It is useful for tracking our experiments, but it is not a verified official Task 11 score.
- **Task score:** the official evaluation kit must be used for an authoritative competition-style result.

## Run it on Windows

Open PowerShell in the project folder. Python 3.10 or newer is recommended.

```powershell
cd D:\yash-ada-project
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, allow it for the current terminal and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

Download the official training data if it is not already present:

```powershell
python scripts/download_data.py
```

The data file should be at `data\raw\task11_subtask1_train.json`. Run the majority baseline:

```powershell
python scripts/majority_baseline.py
```

Then run fine-tuning:

```powershell
python scripts/train_subtask1.py --epochs 3 --batch-size 8 --seed 42
```

The first run downloads the public XLM-R model from Hugging Face. If the model is already cached and you need to run without internet, add `--offline`:

```powershell
python scripts/train_subtask1.py --epochs 3 --batch-size 8 --seed 42 --offline
```

The fine-tuned model and metrics are written to `outputs\subtask1-xlm-roberta\`. A GPU makes training faster; the first local run completed on CPU. If you get an out-of-memory error, try `--batch-size 4`.

## Run it in Google Colab

1. Upload or clone the project into Colab.
2. In **Runtime → Change runtime type**, select a GPU runtime.
3. Install dependencies and run the scripts from the project directory:

   ```bash
   pip install -r requirements.txt
   python scripts/download_data.py
   python scripts/majority_baseline.py
   python scripts/train_subtask1.py --epochs 3 --batch-size 8 --seed 42
   ```

## How we did it

1. Downloaded the official Task 11 Subtask 1 JSON from the organizers' GitHub repository.
2. Split the 960 labeled examples into training and validation sets, preserving the four label groups.
3. Loaded `xlm-roberta-base` and added a binary classification layer for valid/invalid predictions. That new layer starts with random weights; fine-tuning learns it from the task data.
4. Measured the pretrained model, fine-tuned for three epochs, then measured the fine-tuned model on the same validation examples.
5. Saved the model and metrics, then recorded the run in `RESULTS.md`.

## Next steps

The current script uses one seed for both the data split and model training. For a stronger experiment, separate those settings: keep the validation split fixed, train with several different seeds, and report the average and spread of the results. Then inspect errors and use the official evaluation kit before making final claims. A multilingual extension using Task 11 Subtask 3 is a possible stretch goal.

## Files in this project

```text
README.md                  Project explanation and instructions
RESULTS.md                 Experiment log
data/raw/                  Downloaded training data (git-ignored)
scripts/download_data.py   Downloads the official training JSON
scripts/majority_baseline.py  Runs the simple majority baseline
scripts/train_subtask1.py  Evaluates and fine-tunes XLM-R
src/metrics.py             Local validation metrics
outputs/                   Saved models and metrics (git-ignored)
```

## Official resources

- [SemEval 2026 Task 11 overview](https://sites.google.com/view/semeval-2026-task-11)
- [Official dataset and evaluation code](https://github.com/neuro-symbolic-ai/semeval_2026_task_11)
- [XLM-R base model](https://huggingface.co/xlm-roberta-base)
