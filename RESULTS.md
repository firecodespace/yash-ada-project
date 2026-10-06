# Experiment Results

Use one row per run. Report the validation split seed and model revision so results can be reproduced. Do not add official test results until the final configuration is fixed.

| Run | Date | Model | Method | Seed | Validation accuracy | Macro-F1 | Content effect estimate | Notes |
|---|---|---|---|---:|---:|---:|---:|---|
| 1 | 2026-10-07 | Majority class | Predict most frequent training label | 42 | 0.500 | 0.333 | 0.500 | 768 train / 192 validation examples; deterministic stratified split. |
| — | 2026-10-07 | xlm-roberta-base | Fine-tuning launch attempted; model-weight download stalled before training | 42 | — | — | — | Official data is present; no neural-model score yet. |

## Metric notes

- **Accuracy / macro-F1:** measured on the local held-out validation split.
- **Content effect estimate:** mean absolute gap between subgroup accuracies, comparing validity groups within each plausibility group and plausibility groups within each validity group. Lower is better. This development estimate should be checked against the official evaluation kit before final reporting.
- **Task score:** use the official Task 11 evaluation kit for authoritative task scoring.

## Run protocol

Keep the split seed fixed when comparing methods. Repeat promising runs with multiple training seeds and report mean and standard deviation. Track model name, hyperparameters, runtime, and GPU alongside scores in the Notes column or an added table.
