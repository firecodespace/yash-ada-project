"""Evaluate a saved adapter on a synthetic JSON set (accuracy overall, by validity, by figure)."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_lora import PROMPT

adapter, data, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
tok = AutoTokenizer.from_pretrained(adapter)
m = AutoModelForSequenceClassification.from_pretrained("Qwen/Qwen2.5-1.5B", num_labels=2, torch_dtype=torch.bfloat16)
m.config.pad_token_id = tok.pad_token_id
m = PeftModel.from_pretrained(m, adapter).eval().cuda()
rows = json.loads(data.read_text(encoding="utf-8"))
pred = []
for i in range(0, len(rows), 16):
    e = tok([PROMPT.format(s=r["syllogism"]) for r in rows[i:i + 16]], return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        pred += m(**e).logits.argmax(-1).tolist()
y = np.array([int(r["validity"]) for r in rows]); p = np.array(pred); ok = y == p
res = {"accuracy": float(ok.mean()), "valid_acc": float(ok[y == 1].mean()), "invalid_acc": float(ok[y == 0].mean()),
       "by_figure": {str(f): float(ok[[r["figure"] == f for r in rows]].mean()) for f in (1, 2, 3, 4)}}
out.write_text(json.dumps(res, indent=1)); print(res)
