# Results on the fixed 192-example validation split

| System | Accuracy | Macro-F1 | Content effect (lower is better) |
|---|---:|---:|---:|
| Majority class baseline | 0.500 | 0.333 | 0.500 |
| XLM-R base, fine-tuned (starting point, seed 42 split) | 0.609 | 0.608 | 0.109 |
| Qwen2.5-1.5B + LoRA, seed 42 | 0.943 | 0.943 | 0.011 |
| Qwen2.5-1.5B + LoRA, seed 123 | 0.922 | 0.922 | 0.073 |
| Qwen2.5-1.5B + LoRA, seed 7 | 0.927 | 0.927 | 0.031 |
| Qwen2.5-1.5B + LoRA, no task prompt (ablation) | 0.812 | 0.812 | 0.035 |
| Qwen2.5-3B 4-bit + LoRA | 0.943 | 0.943 | 0.022 |
| Qwen2.5-1.5B + LoRA + synthetic data (FINAL MODEL) | 0.948 | 0.948 | 0.022 |
