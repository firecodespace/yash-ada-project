# Experiment Results

Use one row per run. Report the validation split seed and model revision so results can be reproduced. Do not add official test results until the final configuration is fixed.

| Run | Date | Model | Method | Seed | Validation accuracy | Macro-F1 | Content effect estimate | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | 2026-10-07 | Majority class | Predict most frequent training label | 42 | 0.500 | 0.333 | 0.500 | 768 train / 192 validation examples; local score estimate 0.356. |
| 2 | 2026-10-07 | xlm-roberta-base | Full fine-tuning, 3 epochs, learning rate 2e-5 | 42 | 0.609 | 0.608 | 0.109 | 192 validation examples; local score estimate 0.552; CPU run. |
| 3 | 2026-10-07 | xlm-roberta-base | Full fine-tuning, 3 epochs, learning rate 2e-5 | 123 | 0.755 | 0.755 | 0.070 | 192 validation examples; local score estimate 0.708; CPU run. |
| 4 | 2026-11-05 | Qwen2.5-1.5B | LoRA r=16, prompt, 4 epochs, lr 1e-4 | 42 | 0.943 | 0.943 | 0.011 | Fixed split (split seed 42); GPU. |
| 5 | 2026-11-05 | Qwen2.5-1.5B | Same, training seed 123 | 123 | 0.922 | 0.922 | 0.073 | Same split as run 4. |
| 6 | 2026-11-05 | Qwen2.5-1.5B | Same, training seed 7 | 7 | 0.927 | 0.927 | 0.031 | Same split as run 4. |
| 7 | 2026-11-05 | Qwen2.5-1.5B | Ablation: no task prompt | 42 | 0.813 | 0.812 | 0.035 | Prompt is worth about 13 points. |
| 8 | 2026-11-05 | Qwen2.5-3B | 4-bit NF4 LoRA, prompt | 42 | 0.943 | 0.943 | 0.022 | Same accuracy as 1.5B. |
| 9 | 2026-11-05 | Qwen2.5-1.5B | LoRA on real + 1,800 synthetic examples, 3 epochs (**final model**) | 42 | 0.948 | 0.948 | 0.022 | Held-out synthetic set: 0.992 (real-only model: 0.815). |

## Metric notes

- **Accuracy / macro-F1:** measured on the local held-out validation split.
- **Content effect estimate:** mean absolute gap between subgroup accuracies, comparing validity groups within each plausibility group and plausibility groups within each validity group. Lower is better. This development estimate should be checked against the official evaluation kit before final reporting.
- **Task score:** use the official Task 11 evaluation kit for authoritative task scoring.

## Run protocol

Keep the split seed fixed when comparing methods. Repeat promising runs with multiple training seeds and report mean and standard deviation. Track model name, hyperparameters, runtime, and GPU alongside scores in the Notes column or an added table.

Runs 4-8 share one validation split (split seed 42), which is also the split of the XLM-R run 2 above, so they are directly comparable. Mean of the three 1.5B seeds: 0.931 +/- 0.009 accuracy.
