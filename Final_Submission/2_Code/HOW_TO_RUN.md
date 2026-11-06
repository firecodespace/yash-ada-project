# How to run the code

Run every command from inside this `2_Code` folder. Paths to the datasets and the trained model point to the sibling folders `4_Datasets` and `5_Trained_Model`.

## Requirements

- Python 3.11, an NVIDIA GPU with 6 GB or more (the models were trained on an RTX 4050, 6 GB).
- Packages: `pip install -r requirements.txt` and `transformers==4.46.3 peft==0.13.2 bitsandbytes reportlab matplotlib`.
- The first run downloads `Qwen/Qwen2.5-1.5B` from Hugging Face (about 3 GB).

## Files

| File | Purpose |
|---|---|
| `scripts/train_lora.py` | **Main method.** LoRA fine-tuning of Qwen2.5 on the real data (plus optional synthetic data) with a fixed validation split. |
| `scripts/gen_synthetic.py` | Generates exactly-labelled syllogisms from the classical mood/figure table. |
| `src/logic.py` | Exact validity checker by enumerating all 256 Venn-region worlds. |
| `src/metrics.py` | Accuracy, macro-F1 and the content-effect estimate. |
| `scripts/eval_synthetic.py` | Scores a saved adapter on a synthetic test set. |
| `scripts/predict.py` | Command-line prediction with the saved adapter. |
| `scripts/demo.py`, `scripts/demo.html` | Live demo web page (type a syllogism or click an example). |
| `scripts/train_subtask1.py`, `scripts/majority_baseline.py`, `scripts/download_data.py` | Baselines (XLM-R, majority class) and data download from the starting project. |
| `scripts/make_figures.py`, `scripts/build_report.py` | Build the report figures and PDF from saved results (these expect the full project folder layout, see the GitHub repo). |
| `scripts/run_all.sh` | The seed, ablation and 3B-model runs used in the report. |

## Run the live demo (about 30 seconds to load, then about 0.1 s per check)

```powershell
python scripts/demo.py --adapter ..\5_Trained_Model
```

Open http://127.0.0.1:8000, then click an example or type two premises and a conclusion.

## Classify one syllogism

Point the script at the saved model with the `ADAPTER` environment variable, then pass the text:

```powershell
$env:ADAPTER = "..\5_Trained_Model"
python scripts/predict.py "All dogs are animals. All animals breathe. Therefore, all dogs breathe."
```

## Re-create the final model

```powershell
python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --epochs 3 --data ..\4_Datasets\task11_subtask1_train.json --extra-train ..\4_Datasets\synthetic\syn_train.json --extra-eval ..\4_Datasets\synthetic\syn_test.json --output outputs\lora-qwen15b-syn
```

About 17 minutes on an RTX 4050. It prints the validation accuracy and the synthetic test accuracy and saves the adapter.

## Re-create the synthetic data

```powershell
python scripts/gen_synthetic.py --n 1800 --output ..\4_Datasets\synthetic\syn_train.json
python scripts/gen_synthetic.py --n 600 --test --output ..\4_Datasets\synthetic\syn_test.json
```

## Check the exact logic checker

```powershell
python -c "from src.logic import verdict; print(verdict(1, 'AAA'), verdict(2, 'AII'))"
```

Prints `valid invalid`: figure 1 mood AAA is the classic Barbara form, while AII in figure 2 is invalid.
