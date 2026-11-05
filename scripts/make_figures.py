"""Build report figures and a results summary from outputs/*/metrics.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figs"
FIG.mkdir(parents=True, exist_ok=True)
C = {"blue": "#1f4e9c", "gold": "#f2b700", "grey": "#8a8f98", "red": "#c0392b", "green": "#2e8b57"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

runs = {}
for p in sorted((ROOT / "outputs").glob("lora-*/metrics.json")):
    runs[p.parent.name.replace("lora-", "")] = json.loads(p.read_text())
data = json.loads((ROOT / "data/raw/task11_subtask1_train.json").read_text())

summary = {k: v["final"] | {"train_minutes": v["train_minutes"]} for k, v in runs.items() if "final" in v}
(ROOT / "report" / "summary.json").write_text(json.dumps(summary, indent=1))

# 1. dataset composition
fig, ax = plt.subplots(figsize=(5, 3))
groups = [("Valid", True), ("Invalid", False)]
x = np.arange(2); w = 0.35
for i, (pl, col) in enumerate([("Plausible", C["blue"]), ("Implausible", C["gold"])]):
    vals = [sum(d["validity"] == v and d["plausibility"] == (i == 0) for d in data) for _, v in groups]
    b = ax.bar(x + (i - .5) * w, vals, w, label=pl, color=col)
    ax.bar_label(b)
ax.set_xticks(x, [g for g, _ in groups]); ax.set_ylabel("Examples"); ax.legend(frameon=False)
ax.set_title("Training data: validity vs plausibility (960 examples)")
fig.tight_layout(); fig.savefig(FIG / "data_dist.png"); plt.close(fig)

# 2. model comparison
s42 = runs["qwen15b-s42"]["final"]["accuracy"]
models = [("Majority\nbaseline", .500, C["grey"]), ("XLM-R base\n(pretrained)", .500, C["grey"]),
          ("XLM-R base\n(fine-tuned)", .609, C["gold"]), ("Qwen2.5-1.5B\nLoRA", s42, C["blue"])]
if "qwen3b-4bit" in runs:
    models.append(("Qwen2.5-3B\n4-bit LoRA", runs["qwen3b-4bit"]["final"]["accuracy"], C["green"]))
fig, ax = plt.subplots(figsize=(6.2, 3.3))
b = ax.bar([m[0] for m in models], [m[1] for m in models], color=[m[2] for m in models])
ax.bar_label(b, fmt="%.3f"); ax.set_ylim(0, 1.05); ax.set_ylabel("Validation accuracy")
ax.set_title("Model comparison on the fixed validation split")
fig.tight_layout(); fig.savefig(FIG / "model_comparison.png"); plt.close(fig)

# 3. training curves (seed 42)
h = runs["qwen15b-s42"]["history"]
tr = [(e["epoch"], e["loss"]) for e in h if "loss" in e]
ev = [(e["epoch"], e["eval_accuracy"], e["eval_loss"]) for e in h if "eval_accuracy" in e]
fig, ax = plt.subplots(1, 2, figsize=(7, 3))
ax[0].plot(*zip(*tr), color=C["blue"], label="train loss")
ax[0].plot([e[0] for e in ev], [e[2] for e in ev], "o-", color=C["gold"], label="val loss")
ax[0].set_xlabel("Epoch"); ax[0].set_ylabel("Loss"); ax[0].legend(frameon=False)
ax[1].plot([e[0] for e in ev], [e[1] for e in ev], "o-", color=C["green"])
ax[1].set_xlabel("Epoch"); ax[1].set_ylabel("Validation accuracy"); ax[1].set_ylim(.4, 1)
fig.suptitle("Qwen2.5-1.5B LoRA training dynamics (seed 42)")
fig.tight_layout(); fig.savefig(FIG / "training_curves.png"); plt.close(fig)

# 4. confusion matrix + subgroup accuracy (seed 42)
pr = runs["qwen15b-s42"]["val_predictions"]
cm = np.zeros((2, 2), int)
for r in pr: cm[r["label"], r["pred"]] += 1
fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
ax[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2): ax[0].text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
ax[0].set_xticks([0, 1], ["Invalid", "Valid"]); ax[0].set_yticks([0, 1], ["Invalid", "Valid"])
ax[0].set_xlabel("Predicted"); ax[0].set_ylabel("True"); ax[0].set_title("Confusion matrix")
names, accs = [], []
for v in (1, 0):
    for p in (1, 0):
        g = [r for r in pr if r["label"] == v and r["plaus"] == p]
        names.append(f"{'Valid' if v else 'Invalid'}\n{'plaus.' if p else 'implaus.'}")
        accs.append(np.mean([r["label"] == r["pred"] for r in g]))
b = ax[1].bar(names, accs, color=[C["blue"], C["gold"]] * 2); ax[1].bar_label(b, fmt="%.2f")
ax[1].set_ylim(0, 1.1); ax[1].set_title("Accuracy by subgroup")
fig.tight_layout(); fig.savefig(FIG / "confusion_subgroups.png"); plt.close(fig)

# 5. seeds / ablations
labels = {"qwen15b-s42": "seed 42", "qwen15b-s123": "seed 123", "qwen15b-s7": "seed 7", "qwen15b-noprompt": "no task\nprompt", "qwen3b-4bit": "Qwen 3B\n4-bit", "qwen15b-syn": "real +\nsynthetic"}
ks = [k for k in labels if k in runs and "final" in runs[k]]
fig, ax = plt.subplots(figsize=(6.2, 3.1))
b = ax.bar([labels[k] for k in ks], [runs[k]["final"]["accuracy"] for k in ks], color=C["blue"])
ax.bar_label(b, fmt="%.3f"); ax.set_ylim(.5, 1.02); ax.set_ylabel("Validation accuracy")
ax.set_title("Seed robustness and ablations")
fig.tight_layout(); fig.savefig(FIG / "seeds_ablation.png"); plt.close(fig)

# 6. pipeline diagram
fig, ax = plt.subplots(figsize=(7.4, 2.2)); ax.axis("off")
steps = ["Task 11\nJSON (960)", "Stratified\n80/20 split", "Prompt\nwrapper", "Qwen2.5 +\nLoRA adapters", "Classification\nhead", "Valid /\nInvalid"]
for i, s in enumerate(steps):
    ax.add_patch(plt.Rectangle((i * 1.25, 0.2), 1.05, 0.6, color=C["blue"] if i != 3 else C["gold"], zorder=2))
    ax.text(i * 1.25 + .525, .5, s, ha="center", va="center", color="white" if i != 3 else "black", fontsize=8)
    if i < 5: ax.annotate("", (i * 1.25 + 1.25, .5), (i * 1.25 + 1.05, .5), arrowprops=dict(arrowstyle="->"))
ax.set_xlim(-.1, 7.3); ax.set_ylim(0, 1)
fig.tight_layout(); fig.savefig(FIG / "pipeline.png"); plt.close(fig)
# 7. confidence histogram
d_ok = [r["logit_diff"] * (1 if r["label"] else -1) for r in pr if r["label"] == r["pred"]]
d_bad = [r["logit_diff"] * (1 if r["label"] else -1) for r in pr if r["label"] != r["pred"]]
fig, ax = plt.subplots(figsize=(6, 3))
bins = np.linspace(-6, 10, 25)
ax.hist(d_ok, bins, color=C["blue"], label=f"correct ({len(d_ok)})"); ax.hist(d_bad, bins, color=C["red"], label=f"wrong ({len(d_bad)})")
ax.axvline(0, color="k", lw=.8); ax.set_xlabel("Logit margin towards the true class"); ax.set_ylabel("Examples"); ax.legend(frameon=False)
fig.tight_layout(); fig.savefig(FIG / "confidence.png"); plt.close(fig)
# 8. synthetic robustness
if (ROOT / "outputs/syn_eval_realonly.json").exists() and "qwen15b-syn" in runs:
    ro = json.loads((ROOT / "outputs/syn_eval_realonly.json").read_text())
    sy = json.loads((ROOT / "outputs/syn_eval_syn.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.1))
    names = ["Real data only", "Real + synthetic"]
    b = ax[0].bar(names, [ro["accuracy"], sy["accuracy"]], color=[C["grey"], C["blue"]]); ax[0].bar_label(b, fmt="%.3f")
    ax[0].set_ylim(0, 1.1); ax[0].set_title("Held-out synthetic syllogisms (600)")
    b = ax[1].bar(names, [runs["qwen15b-s42"]["final"]["accuracy"], runs["qwen15b-syn"]["final"]["accuracy"]], color=[C["grey"], C["blue"]]); ax[1].bar_label(b, fmt="%.3f")
    ax[1].set_ylim(0, 1.1); ax[1].set_title("Real validation split (192)")
    fig.tight_layout(); fig.savefig(FIG / "synthetic_robustness.png"); plt.close(fig)
print(json.dumps(summary, indent=1))
