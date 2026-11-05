"""LoRA fine-tuning of a decoder LLM (sequence-classification head) on Task 11 Subtask 1.

The validation split is fixed (--split-seed) so runs with different training seeds,
models and prompts are directly comparable.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    set_seed,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import task_metrics

PROMPT = (
    "Task: decide whether the conclusion of this syllogism follows logically from its "
    "premises. Judge only the logical form; ignore whether the statements are true in "
    "the real world.\nSyllogism: {s}\nIs the argument logically valid?"
)


class DS(torch.utils.data.Dataset):
    def __init__(self, rows, tok, max_len, prompt):
        texts = [PROMPT.format(s=r["syllogism"]) if prompt else r["syllogism"] for r in rows]
        self.enc = tok(texts, truncation=True, max_length=max_len)
        self.labels = [int(r["validity"]) for r in rows]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: v[i] for k, v in self.enc.items()}
        item["labels"] = self.labels[i]
        return item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/raw/task11_subtask1_train.json"))
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--epochs", type=float, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--no-prompt", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--extra-train", type=Path, help="synthetic training JSON added to the training rows")
    ap.add_argument("--extra-eval", type=Path, help="synthetic held-out JSON evaluated after training")
    ap.add_argument("--full", action="store_true", help="train on all data (no validation)")
    args = ap.parse_args()

    set_seed(args.seed)
    rows = json.loads(args.data.read_text(encoding="utf-8"))
    strata = [f"{int(r['validity'])}_{int(r['plausibility'])}" for r in rows]
    train_rows, val_rows = train_test_split(
        rows, test_size=0.2, random_state=args.split_seed, stratify=strata
    )
    if args.full:
        train_rows = rows
    if args.extra_train:
        train_rows = train_rows + json.loads(args.extra_train.read_text(encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    kwargs = {"num_labels": 2, "torch_dtype": torch.bfloat16}
    if args.load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": 0}
    model = AutoModelForSequenceClassification.from_pretrained(args.model, **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    if args.load_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    cfg = LoraConfig(
        task_type=TaskType.SEQ_CLS, r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, cfg)
    for n, p in model.named_parameters():  # score head in fp32 for stable training
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()

    prompt = not args.no_prompt
    train_ds = DS(train_rows, tok, 256, prompt)
    val_ds = DS(val_rows, tok, 256, prompt)
    plaus = np.asarray([int(r["plausibility"]) for r in val_rows])

    def compute(ev):
        logits, labels = ev
        return task_metrics(labels, np.argmax(logits, -1), plaus)

    targs = TrainingArguments(
        output_dir=str(args.output / "ckpt"), learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=16,
        gradient_accumulation_steps=args.grad_accum, num_train_epochs=args.epochs,
        warmup_ratio=0.1, lr_scheduler_type="cosine", weight_decay=0.01,
        eval_strategy="no" if args.full else "epoch", save_strategy="no", logging_steps=10,
        report_to="none", seed=args.seed, bf16=True, remove_unused_columns=False,
        gradient_checkpointing=args.load_4bit,
    )
    from transformers import DataCollatorWithPadding
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute, data_collator=DataCollatorWithPadding(tok))
    t0 = time.time()
    trainer.train()
    minutes = (time.time() - t0) / 60
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output / "adapter"))
    tok.save_pretrained(str(args.output / "adapter"))
    out = {"model": args.model, "seed": args.seed, "split_seed": args.split_seed,
           "lr": args.lr, "epochs": args.epochs, "rank": args.rank, "prompt": prompt,
           "train_minutes": minutes, "history": trainer.state.log_history}
    if not args.full:
        pred = trainer.predict(val_ds)
        logits = pred.predictions
        p = np.argmax(logits, -1)
        out["final"] = task_metrics(pred.label_ids, p, plaus)
        out["val_predictions"] = [
            {"id": r["id"], "label": int(r["validity"]), "plaus": int(r["plausibility"]),
             "pred": int(x), "logit_diff": float(l[1] - l[0]), "text": r["syllogism"]}
            for r, x, l in zip(val_rows, p, logits)]
        print(json.dumps(out["final"], indent=2))
        if args.extra_eval:
            trainer.compute_metrics = None  # metrics are tied to the 192 validation rows
            syn = json.loads(args.extra_eval.read_text(encoding="utf-8"))
            sp = trainer.predict(DS(syn, tok, 256, prompt))
            sy = np.argmax(sp.predictions, -1)
            out["synthetic_accuracy"] = float((sy == sp.label_ids).mean())
            out["synthetic_by_validity"] = {
                str(v): float((sy == sp.label_ids)[sp.label_ids == v].mean()) for v in (0, 1)}
            print("synthetic held-out accuracy", out["synthetic_accuracy"])
    (args.output / "metrics.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
