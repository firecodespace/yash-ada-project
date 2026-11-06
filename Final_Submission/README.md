# Logic over Plausibility: Fine-Tuning Language Models to Judge Syllogistic Validity

**SemEval 2026 Task 11, Subtask 1** | ADA course project | Yash, Enrollment A2305224323
Amity University Noida | Guide: Dr Dipika Jain | B.Tech CSE, Batch 2024-28

A syllogism is a two-premise argument. It is *valid* if the conclusion must follow from the premises, whether or not the statements are true. We fine-tune a language model (Qwen2.5-1.5B with LoRA) to judge validity from the logical form alone, even when the content sounds absurd.

**Headline result (192-example validation split):** the original XLM-R baseline scored 60.9%. The final model scores **94.8%** (three training seeds of the base setup: 93.1% +/- 0.9). On held-out synthetic syllogisms it scores 99.2%.

## What is in this folder

| Folder / file | Contents |
|---|---|
| `1_Project_Report/` | **Project_Report_ADA.pdf**: the full project report (start here). |
| `2_Code/` | All project code: training, synthetic-data generator, exact logic checker, metrics, demo web app, figure and report builders. See `2_Code/HOW_TO_RUN.md`. |
| `3_Results/` | Results table, per-run metrics (JSON), synthetic-data evaluation, all report figures, and the experiment log. |
| `4_Datasets/` | The official Task 11 Subtask 1 training data (960 examples) and our synthetic training/test sets. |
| `5_Trained_Model/` | The final trained LoRA adapter (used by the demo). |
| `STUDY_GUIDE.md` | Plain-language explanation of every concept with examples. |

## Quick tour (5 minutes)

1. Read the abstract and sections 4, 5 and 7 of the report in `1_Project_Report/`.
2. Look at `3_Results/results_table.md` and the figures in `3_Results/figures/`.
3. Skim `2_Code/scripts/train_lora.py` (the main method) and `2_Code/scripts/gen_synthetic.py` (synthetic data).
4. Optional: run the live demo (see `2_Code/HOW_TO_RUN.md`). It needs a computer with the Python environment and an NVIDIA GPU.

## Honest notes

- All scores are **validation estimates** on 192 held-out examples (standard error about 1.8 points), not official competition test scores. The content-effect score is our own approximation.
- The synthetic test set reuses the same sentence templates as the synthetic training data, so it shows robustness to new words, not to entirely new phrasings.
- The official dataset belongs to the SemEval 2026 Task 11 organisers: https://github.com/neuro-symbolic-ai/semeval_2026_task_11
- Full project history and code: https://github.com/firecodespace/yash-ada-project
