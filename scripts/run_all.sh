#!/bin/bash
# Extra seeds, prompt ablation and a larger 4-bit model, all on the fixed split (split-seed 42).
P=.venv311/Scripts/python
$P scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --output outputs/lora-qwen15b-s123 --seed 123 > outputs/qwen15b-s123.log 2>&1
$P scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --output outputs/lora-qwen15b-s7 --seed 7 > outputs/qwen15b-s7.log 2>&1
$P scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --output outputs/lora-qwen15b-noprompt --no-prompt > outputs/qwen15b-noprompt.log 2>&1
$P scripts/train_lora.py --model Qwen/Qwen2.5-3B --load-4bit --batch-size 4 --grad-accum 4 --output outputs/lora-qwen3b-4bit > outputs/qwen3b.log 2>&1
