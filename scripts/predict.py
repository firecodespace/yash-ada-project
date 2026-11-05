"""Classify syllogisms with the saved LoRA adapter.

Usage: python scripts/predict.py "All planets are dogs. All dogs are animals. Therefore, all planets are animals."
"""
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import PROMPT

ADAPTER = Path("outputs/lora-qwen15b-syn/adapter")
BASE = "Qwen/Qwen2.5-1.5B"

tok = AutoTokenizer.from_pretrained(ADAPTER)
model = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=2, torch_dtype=torch.bfloat16)
model.config.pad_token_id = tok.pad_token_id
model = PeftModel.from_pretrained(model, ADAPTER).eval().cuda() if torch.cuda.is_available() else PeftModel.from_pretrained(model, ADAPTER).eval()
for text in sys.argv[1:]:
    enc = tok(PROMPT.format(s=text), return_tensors="pt").to(model.device)
    with torch.no_grad():
        p = model(**enc).logits.float().softmax(-1)[0]
    print(json.dumps({"syllogism": text, "valid": bool(p[1] > p[0]), "p_valid": round(float(p[1]), 3)}))
