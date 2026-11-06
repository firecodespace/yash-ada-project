"""Build report/Project_Report_ADA.pdf from outputs/ and report/figs/."""
import json
from pathlib import Path

import numpy as np
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (CondPageBreak, BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate,
                                Paragraph, Preformatted, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "report" / "figs"
OUT = ROOT / "report" / "Project_Report_ADA.pdf"
R = {p.parent.name.replace("lora-", ""): json.loads(p.read_text()) for p in (ROOT / "outputs").glob("lora-*/metrics.json")}
F = lambda k, m="accuracy": R[k]["final"][m]
RO = json.loads((ROOT / "outputs/syn_eval_realonly.json").read_text())
SY = json.loads((ROOT / "outputs/syn_eval_syn.json").read_text())
seeds = ["qwen15b-s42", "qwen15b-s123", "qwen15b-s7"]
acc = [F(k) for k in seeds]
ce = [F(k, "content_effect_estimate") for k in seeds]
sc = [F(k, "task_score_estimate") for k in seeds]
m_acc, s_acc = np.mean(acc), np.std(acc)
m_ce, m_sc = np.mean(ce), np.mean(sc)

base = "Times-Roman"
body = ParagraphStyle("b", fontName=base, fontSize=11, leading=15.5, alignment=TA_JUSTIFY, spaceAfter=7)
h1 = ParagraphStyle("h1", fontName="Times-Bold", fontSize=15, leading=19, spaceBefore=4, spaceAfter=9, keepWithNext=1)
h2 = ParagraphStyle("h2", fontName="Times-Bold", fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=5, keepWithNext=1)
cap = ParagraphStyle("c", fontName="Times-Italic", fontSize=10, leading=13, alignment=TA_CENTER, spaceAfter=10, spaceBefore=3)
bul = ParagraphStyle("bl", parent=body, leftIndent=18, bulletIndent=6, spaceAfter=3)
cen = ParagraphStyle("cn", fontName=base, fontSize=11.5, leading=14.5, alignment=TA_CENTER)
cenb = ParagraphStyle("cb", parent=cen, fontName="Times-Bold", fontSize=14, leading=19)
cell = ParagraphStyle("cell", fontName=base, fontSize=9.5, leading=12)
cellh = ParagraphStyle("cellh", fontName="Times-Bold", fontSize=9.5, leading=12, textColor=colors.white)
code = ParagraphStyle("code", fontName="Courier", fontSize=8.8, leading=11.2, leftIndent=8,
                      backColor=colors.HexColor("#f3f4f7"), borderPadding=6, spaceAfter=10)

P = lambda t: Paragraph(t, body)
H1 = lambda t: Paragraph(t, h1)
H2 = lambda t: Paragraph(t, h2)
B = lambda items: [Paragraph(i, bul, bulletText="•") for i in items]
_t = [0]
_f = [0]


def fig(name, width, caption):
    _f[0] += 1
    img = Image(str(FIG / name))
    r = img.imageHeight / img.imageWidth
    img.drawWidth, img.drawHeight = width * cm, width * cm * r
    return KeepTogether([img, Paragraph(f"Figure {_f[0]}: {caption}", cap)])


def table(rows, widths, caption):
    _t[0] += 1
    data = [[Paragraph(str(c), cellh) for c in rows[0]]] + [[Paragraph(str(c), cell) for c in r] for r in rows[1:]]
    t = Table(data, colWidths=[w * cm for w in widths], repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f4e9c")),
                           ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef2fa")]),
                           ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#b8c0d0")),
                           ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
    return KeepTogether([t, Paragraph(f"Table {_t[0]}: {caption}", cap)])


def on_page(c, d):
    c.saveState()
    c.setFont(base, 10)
    c.drawCentredString(A4[0] / 2, 1.3 * cm, str(d.page - 1))
    c.restoreState()


def cover_page(c, d):
    c.saveState()
    c.setLineWidth(0.8)
    c.rect(1.2 * cm, 1.2 * cm, A4[0] - 2.4 * cm, A4[1] - 2.4 * cm)
    c.restoreState()


doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=2.3 * cm, rightMargin=2.3 * cm,
                      topMargin=2.2 * cm, bottomMargin=2.2 * cm,
                      title="Project Report - Logic over Plausibility (SemEval 2026 Task 11)", author="Yash")
fr = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="f")
doc.addPageTemplates([PageTemplate(id="cover", frames=[fr], onPage=cover_page, autoNextPageTemplate="main"),
                      PageTemplate(id="main", frames=[fr], onPage=on_page)])

S = []
# ---------------- cover
logo = Image(str(ROOT / "assets" / "amity_logo.png"), width=5.4 * cm * 277 / 311, height=5.4 * cm)
S += [Spacer(1, 0.2 * cm), Paragraph("Project report", cen), Paragraph("on", cen), Spacer(1, 4),
      Paragraph("Logic over Plausibility: Fine-Tuning Language Models to Judge Syllogistic Validity<br/>"
                "(SemEval 2026 Task 11, Subtask 1)", cenb),
      Spacer(1, 4), Paragraph("Course: Analysis and Design of Algorithms (ADA)", cen),
      Paragraph("Submitted to", cen), Paragraph("Amity University Noida, Uttar Pradesh", cen),
      Spacer(1, 10), logo, Spacer(1, 22),
      Paragraph("In partial fulfillment of the requirements for the award of the degree of", cen),
      Paragraph("Bachelor of Technology", cen), Paragraph("in", cen),
      Paragraph("Computer Science and Engineering", cen), Spacer(1, 6), Paragraph("by", cen),
      Paragraph("Name: Yash", cen), Paragraph("Enrollment number: A2305224323", cen), Spacer(1, 8),
      Spacer(1, 6), Paragraph("Under the guidance of", cen), Paragraph("Dr Dipika Jain", cen), Spacer(1, 22),
      Paragraph("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING<br/>AMITY SCHOOL OF ENGINEERING &amp; TECHNOLOGY<br/>"
                "AMITY UNIVERSITY NOIDA, UTTAR PRADESH", cen),
      Spacer(1, 22), Paragraph("BATCH: 2024-28", cen), PageBreak()]

# ---------------- abstract
S += [H1("ABSTRACT"),
      P("A syllogism is a short argument made of two premises and a conclusion. Whether it is <i>valid</i> depends only on its "
        "logical form, not on whether its statements are true or sensible: &ldquo;All planets are dogs. All dogs are animals. "
        "Therefore all planets are animals&rdquo; is valid even though the premises are absurd. Language models, like people, are "
        "prone to the <i>content effect</i>: they accept arguments that sound believable and reject ones that sound strange. "
        "SemEval 2026 Task 11 asks systems to judge validity while staying robust to plausibility."),
      P(f"This project fine-tunes language models on the 960 labelled examples of Subtask 1. The starting repository used "
        f"XLM-RoBERTa-base and reached 60.9% and 75.5% validation accuracy on two runs (with different splits). We replace it with a "
        f"decoder language model (Qwen2.5) adapted with LoRA, a method that trains about 1.2% of the weights, together with a prompt "
        f"that tells the model to ignore real-world truth. On a fixed 192-example validation split the final system reaches "
        f"<b>{m_acc*100:.1f}% &plusmn; {s_acc*100:.1f}%</b> accuracy over three training seeds (best {max(acc)*100:.1f}%), with a "
        f"content-effect estimate of {m_ce:.3f}, against 50.0% for the majority baseline and 60.9% for fine-tuned XLM-R on the same split. "
        f"A 3B model in 4-bit precision matches the best 1.5B run at {F('qwen3b-4bit')*100:.1f}%. Removing the task prompt costs "
        f"{(F('qwen15b-s42')-F('qwen15b-noprompt'))*100:.1f} points, showing that instructing the model to ignore content matters. "
        f"During testing we found that the real-data-only model fails some plain textbook syllogisms (81.5% on a synthetic test set), so we added an exact syllogism generator, "
        f"verified against brute-force logic, and trained on real plus synthetic data: this lifts the synthetic test to {SY['accuracy']*100:.1f}% and gives {F('qwen15b-syn')*100:.1f}% on the real validation split. "
        "The report covers the algorithms and their complexity, the experimental design, results, subgroup and error analysis, "
        "and the limits of what the numbers can claim."),
      PageBreak()]

# ---------------- 1 intro
S += [H1("1. INTRODUCTION"), H2("1.1 Overview"),
      P("Deductive reasoning is a core test of whether a language model understands structure rather than memorising patterns. "
        "A categorical syllogism uses statements such as <i>all A are B</i>, <i>no A is B</i>, <i>some A are B</i> and "
        "<i>some A are not B</i>. Two premises share a middle term and a conclusion links the remaining two terms. "
        "The conclusion is valid if it must be true whenever both premises are true."),
      P("Humans and language models both show <i>belief bias</i>. A valid argument with an unbelievable conclusion is often "
        "rejected, and an invalid argument with a believable conclusion is often accepted [7]. SemEval 2026 Task 11 isolates this problem. "
        "Each example carries two independent labels: validity (does the conclusion follow?) and plausibility (does the content "
        "sound realistic?). A good system must get validity right in all four combinations."),
      fig("pipeline.png", 15.5, "Overview of the system: from the Task 11 JSON file to a valid / invalid decision."),
      H2("1.2 Why this is an algorithms problem"),
      P("The course is about the analysis and design of algorithms, so the report treats the pipeline as an algorithmic system. "
        "Validity of a syllogism can be decided exactly by a classical algorithm (enumerate the possible set relationships between "
        "three terms), but that algorithm needs the sentence parsed into a logical form first, and the text varies widely "
        "(&ldquo;It is impossible for something to be an organism and a creature at the same time&rdquo;). A learned classifier avoids "
        "hand-written parsing. We analyse its running time and memory, compare design alternatives (encoder versus decoder models, "
        "full fine-tuning versus parameter-efficient LoRA, quantisation) and measure which design is most accurate for the compute it uses.")]

# ---------------- 2 objectives
S += [H1("2. PROBLEM STATEMENT AND OBJECTIVES"),
      P("<b>Problem.</b> Given the text of a syllogism, predict whether it is logically valid (1) or invalid (0). The decision "
        "must depend on logical structure and not on how plausible the content sounds."),
      P("<b>Objectives.</b>"),
      *B(["Establish honest baselines (majority class, pretrained and fine-tuned XLM-R) on one fixed validation split.",
          "Design a stronger approach and beat those baselines by a clear margin.",
          "Measure robustness: variation across training seeds, and accuracy in each validity &times; plausibility subgroup.",
          "Run ablations (task prompt, model size) to explain <i>why</i> the system works.",
          "Analyse errors and state plainly what the results do and do not show.",
          "Deliver reproducible code, a saved final model and this report."]),
      H2("2.1 Evaluation measures"),
      P("<b>Accuracy</b> is the share of correct predictions. <b>Macro-F1</b> averages F1 over the valid and invalid classes. "
        "The <b>content-effect estimate</b> is the mean absolute accuracy gap between valid and invalid examples within each "
        "plausibility group, and between plausible and implausible examples within each validity group; lower is better. "
        "The <b>task score estimate</b> is accuracy / (1 + ln(1 + content effect)). The content effect and task score are local "
        "approximations written for this project (<font face='Courier'>src/metrics.py</font>); they are not the official competition "
        "scorer, so they are used only to compare our own runs.")]

# ---------------- 3 data
S += [H1("3. DATASET AND VALIDATION PROTOCOL"),
      P("The data is the official Task 11 Subtask 1 training file [1]: 960 English syllogisms. Each record has an id, the syllogism text, "
        "a boolean <i>validity</i> and a boolean <i>plausibility</i>. The text is one string holding two premises and a conclusion, "
        "about 25 words on average, phrased in many different ways so that a model cannot rely on a fixed template."),
      table([["Validity", "Plausible", "Implausible", "Total"], ["Valid", "240", "240", "480"],
             ["Invalid", "234", "246", "480"], ["Total", "474", "486", "960"]],
            [4, 3.5, 3.5, 3.5], "Composition of the 960 training examples. The data is almost perfectly balanced."),
      fig("data_dist.png", 9.5, "Counts per validity and plausibility group."),
      H2("3.1 Fixed split"),
      P("The examples are split 80% / 20% (768 train, 192 validation) with stratification over the four groups, so each group appears "
        "in both parts in the same proportion. The original script tied the split to the training seed, so its two runs were not "
        "comparable with each other. <b>We fixed the split seed at 42</b> and varied only the training seed, which makes the numbers "
        "in this report comparable. Settings such as the number of epochs and the prompt were chosen once in advance, not tuned per run, "
        "but the figures remain validation estimates and not a final test score.")]

# ---------------- 4 methodology
S += [H1("4. METHODOLOGY AND ALGORITHMS"), H2("4.1 Baselines"),
      P("<b>Majority class</b> always predicts the more common training label; on balanced data it scores about 50%. "
        "<b>XLM-RoBERTa-base</b> [4] (278M parameters, encoder) is evaluated before training (its new classification head is random, "
        "giving 50%) and after full fine-tuning for three epochs, as in the starting repository."),
      H2("4.2 Decoder language model with a classification head"),
      P("The proposed model is Qwen2.5 [5] (1.5B and 3B parameters), a decoder-only transformer [6] pre-trained on large text and code "
        "corpora that include a lot of step-by-step reasoning. We attach a linear <i>sequence-classification head</i> to the hidden "
        "state of the last token and train it to output valid or invalid. A decoder reads the sentence left to right and the last "
        "token summarises it, so the classifier can use what the model already knows about reasoning."),
      H2("4.3 Task prompt"),
      P("Each syllogism is wrapped in a fixed instruction before being tokenised:"),
      Preformatted("Task: decide whether the conclusion of this syllogism follows\nlogically from its premises. Judge only the logical form; ignore\n"
                   "whether the statements are true in the real world.\nSyllogism: <text>\nIs the argument logically valid?", code),
      P("The prompt puts the content-effect warning directly in the input, so the final-token representation is computed with the "
        "right question in mind. The ablation in Section 6 measures its value."),
      H2("4.4 LoRA: low-rank adaptation"),
      P("Full fine-tuning updates every weight matrix W of size d<sub>out</sub> &times; d<sub>in</sub>. LoRA [2] freezes W and learns a "
        "low-rank update, so the layer computes <i>y = Wx + (&alpha;/r)&middot;B(Ax)</i>, where A is r &times; d<sub>in</sub>, "
        "B is d<sub>out</sub> &times; r and the rank r is small (here 16, &alpha; = 32). Adapters are added to all attention "
        "(q, k, v, o) and feed-forward (gate, up, down) projections. For the 1.5B model this trains <b>18.5M of 1,562M parameters "
        "(1.18%)</b>; for the 3B model it is <b>29.9M of 3,116M (0.96%)</b>. Gradients and optimiser state are only needed for these few "
        "parameters, which is what allows a 1.5B&ndash;3B model to train on a 6 GB laptop GPU."),
      H2("4.5 Training algorithm"),
      Preformatted("""input: examples D (768), model f = frozen weights + LoRA adapters theta
mini-batches of 8, accumulated over 2 steps -> effective batch 16
for epoch = 1..4:
    for each accumulated batch:
        logits = f(prompt(x); theta)                 # forward pass, bf16
        loss   = cross_entropy(logits, y)
        theta  = AdamW(theta, grad(loss), lr_t)      # only adapters + head
    lr_t: 10% linear warm-up to 1e-4, then cosine decay
    evaluate on the fixed 192 validation examples""", code),
      H2("4.6 Complexity analysis"),
      P("Let n be the sequence length in tokens (about 60 after adding the prompt), L the number of layers, d the hidden width and r the "
        "LoRA rank. One transformer layer costs <b>O(n&sup2;d + n d&sup2;)</b> time: the n&sup2;d term is self-attention and the nd&sup2; term "
        "is the projections and feed-forward network. Since n &asymp; 60 is far smaller than d (1536 for the 1.5B model), the d&sup2; term "
        "dominates, so a forward pass is about <b>O(L &middot; n &middot; d&sup2;)</b>, linear in sequence length. LoRA adds "
        "O(n&middot;r&middot;d) per adapted matrix, a small overhead because r &lt;&lt; d. Training for E epochs on N examples costs "
        "O(E&middot;N&middot;L&middot;n&middot;d&sup2;), with the backward pass about twice the forward pass. Optimiser memory falls from O(P) "
        "for full fine-tuning to O(P<sub>LoRA</sub>) with P<sub>LoRA</sub> &asymp; 1.2% of P, and 4-bit NF4 quantisation [3] reduces the "
        "frozen weights from 16 to about 4 bits each (roughly a 4&times; saving) at the cost of dequantising weights in every pass. "
        "Inference classifies one syllogism with a single forward pass and no autoregressive decoding."),
      H2("4.7 Exact classical algorithm for comparison"),
      P("A syllogism over terms S, M and P can be decided exactly by model enumeration. Each of the three terms is a set, and the eight Venn regions "
        "(in or out of S, M, P) are either empty or non-empty, giving 2<super>8</super> = 256 candidate worlds. A world satisfies a premise such as "
        "<i>all S are M</i> when the regions S&and;&not;M are empty, and <i>some S are M</i> when at least one region in S&and;M is non-empty. The argument is valid "
        "if every world that satisfies both premises also satisfies the conclusion. The check is constant time, O(256) per argument, so the hard part "
        "is not the logic but turning free text into the formal form, which is the step the learned model absorbs implicitly. Our pipeline therefore "
        "trades a guaranteed-correct but brittle parser for a classifier that tolerates wording variety but can make errors (Section 9). "
        "Combining the two is listed as future work."),
      table([["Setting", "Qwen2.5-1.5B (main)", "Qwen2.5-3B (4-bit)"],
             ["Trainable / total parameters", "18.5M / 1,562M (1.18%)", "29.9M / 3,116M (0.96%)"],
             ["Weights precision", "bf16", "4-bit NF4 + double quant, bf16 compute"],
             ["Batch (per step x accumulation)", "8 x 2 = 16", "4 x 4 = 16"],
             ["Learning rate / schedule", "1e-4, cosine, 10% warm-up", "1e-4, cosine, 10% warm-up"],
             ["Epochs / LoRA rank / alpha", "4 / 16 / 32", "4 / 16 / 32"],
             ["Training time (RTX 4050, 6 GB)", f"{R['qwen15b-s42']['train_minutes']:.1f} min",
              f"{R['qwen3b-4bit']['train_minutes']:.1f} min"]],
            [5.5, 5.3, 5.3], "Training configuration.")]

# ---------------- 5 results
rows = [["System", "Accuracy", "Macro-F1", "Content effect (lower better)", "Task score est."],
        ["Majority class", "0.500", "0.333", "0.500", "0.356"],
        ["XLM-R base, pretrained", "0.500", "-", "-", "-"],
        ["XLM-R base, fine-tuned (seed 42)", "0.609", "0.608", "0.109", "0.552"]]
for k, n in [("qwen15b-s42", "Qwen2.5-1.5B LoRA, seed 42"), ("qwen15b-s123", "Qwen2.5-1.5B LoRA, seed 123"),
             ("qwen15b-s7", "Qwen2.5-1.5B LoRA, seed 7")]:
    rows.append([n, f"{F(k):.3f}", f"{F(k,'macro_f1'):.3f}", f"{F(k,'content_effect_estimate'):.3f}", f"{F(k,'task_score_estimate'):.3f}"])
rows.append(["<b>Qwen2.5-1.5B LoRA, mean of 3 seeds</b>", f"<b>{m_acc:.3f}</b> &plusmn; {s_acc:.3f}",
             f"{np.mean([F(k,'macro_f1') for k in seeds]):.3f}", f"{m_ce:.3f}", f"{m_sc:.3f}"])
k = "qwen3b-4bit"
rows.append(["Qwen2.5-3B 4-bit LoRA, seed 42", f"{F(k):.3f}", f"{F(k,'macro_f1'):.3f}",
             f"{F(k,'content_effect_estimate'):.3f}", f"{F(k,'task_score_estimate'):.3f}"])
S += [PageBreak(), H1("5. RESULTS AND PERFORMANCE ANALYSIS"),
      P("All Qwen numbers come from the project code on the same 192 validation examples. The XLM-R and majority numbers are from the "
        "starting repository's seed-42 runs, which use the same stratified split (split seed 42), so the comparison is like for like."),
      table(rows, [5.9, 2.5, 2.2, 3.2, 2.5], "Main results on the fixed validation split."),
      fig("model_comparison.png", 12.5, "Validation accuracy of each system."),
      P(f"<b>Interpretation.</b> LoRA-tuned Qwen2.5-1.5B beats fine-tuned XLM-R by about {(m_acc-0.609)*100:.0f} points and the majority "
        f"baseline by about {(m_acc-0.5)*100:.0f} points. The three seeds range from {min(acc)*100:.1f}% to {max(acc)*100:.1f}%, so the "
        f"improvement is far larger than seed-to-seed noise. With 192 examples one example is worth 0.52 points, and a 95% binomial interval "
        f"around 93% is roughly &plusmn;3.6 points, so differences of one or two points between our own variants (for example 1.5B versus 3B) "
        "should not be over-interpreted. The large gap to XLM-R is plausibly explained by scale and pre-training: the decoder has seen far "
        "more text and code involving reasoning, while XLM-R-base is a small multilingual encoder. We did not run a controlled test that separates these causes."),
      fig("training_curves.png", 14.5, "Loss and validation accuracy per epoch for the seed-42 run."),
      P("Validation accuracy improves over the four epochs and the validation loss flattens near the end, so four epochs is a "
        "reasonable stopping point; we did not tune it per run.")]

# ---------------- 6 ablation
nop = F("qwen15b-noprompt")
S += [H1("6. ABLATIONS AND ROBUSTNESS"),
      fig("seeds_ablation.png", 12.5, "Accuracy across seeds, without the task prompt, and with the larger 4-bit model."),
      table([["Variant", "Accuracy", "Content effect", "Observation"],
             ["Full prompt (seed 42)", f"{F('qwen15b-s42'):.3f}", f"{F('qwen15b-s42','content_effect_estimate'):.3f}", "Reference"],
             ["No task prompt (raw syllogism)", f"{nop:.3f}", f"{F('qwen15b-noprompt','content_effect_estimate'):.3f}",
              f"{(F('qwen15b-s42')-nop)*100:.1f} points lower"],
             ["Qwen2.5-3B, 4-bit", f"{F('qwen3b-4bit'):.3f}", f"{F('qwen3b-4bit','content_effect_estimate'):.3f}", "Same accuracy as 1.5B"],
             ["Seeds 42 / 123 / 7", " / ".join(f"{a:.3f}" for a in acc), " / ".join(f"{c:.3f}" for c in ce), f"Std {s_acc:.3f}"]],
            [4.8, 4.4, 3.8, 3.1], "Ablation summary."),
      P(f"<b>Task prompt.</b> Without the instruction, accuracy drops from {F('qwen15b-s42')*100:.1f}% to {nop*100:.1f}%. The prompt costs a few "
        "dozen tokens but supplies the key hint that only form matters. The no-prompt run is a single seed, so the exact size of the gap is "
        "uncertain, but it is much larger than the seed spread. <b>Model size.</b> Doubling the parameter count (with 4-bit weights) did not "
        "help on this validation set (94.3% for both), so the extra memory and compute were not worth it; a larger validation set would be "
        "needed to detect a smaller difference. <b>Seeds.</b> A spread of a few points across seeds is typical for 768 training examples, "
        "so we report the mean and spread instead of the best run alone.")]

# ---------------- 6b synthetic
S += [H1("7. ROBUSTNESS TO PLAIN SYLLOGISMS AND SYNTHETIC DATA"),
      P("Validation accuracy is measured on examples written in the dataset's own style. A manual check of the saved model showed that this can hide a weakness: "
        "the argument &ldquo;All planets are dogs. All dogs are animals. Therefore, all planets are animals&rdquo; (the example from the task description) was rated "
        "<i>invalid</i> with very high confidence, and so was &ldquo;No fish are mammals. All whales are mammals. Therefore, no whales are fish&rdquo;. "
        "The model had partly learned the dataset's phrasing rather than the logical form."),
      P("<b>Generator.</b> <font face='Courier'>scripts/gen_synthetic.py</font> builds syllogisms from the classical table of moods and figures (A, E, I, O statements in four figures, 256 forms in all). "
        "Valid examples come only from the 15 forms valid under the modern reading, and invalid examples only from forms invalid under every reading, so no label depends on how empty terms are treated. "
        "We checked this table against the brute-force algorithm of Section 4.7: over all 256 Venn-region worlds, the 15 forms are exactly the valid ones, and the 9 excluded forms are valid only if every term is assumed non-empty. "
        "Terms are real nouns or invented words, and each statement type has four wordings, so content is often absurd. The training set has 1,800 generated examples; "
        "a separate test set of 600 uses held-out words so that no term seen in training appears in it."),
      table([["Model (seed 42)", "Real validation accuracy", "Synthetic test accuracy", "Valid / invalid on synthetic"],
             ["Real data only", f"{F('qwen15b-s42'):.3f}", f"{RO['accuracy']:.3f}", f"{RO['valid_acc']:.2f} / {RO['invalid_acc']:.2f}"],
             ["Real + 1,800 synthetic (3 epochs)", f"{F('qwen15b-syn'):.3f}", f"{SY['accuracy']:.3f}", f"{SY['valid_acc']:.2f} / {SY['invalid_acc']:.2f}"]],
            [5.2, 3.8, 3.6, 3.6], "Effect of adding synthetic training data."),
      fig("synthetic_robustness.png", 14, "Accuracy on held-out synthetic syllogisms (left) and the real validation split (right)."),
      P(f"Adding synthetic data raises accuracy on the synthetic test from {RO['accuracy']*100:.1f}% to {SY['accuracy']*100:.1f}% without hurting the real validation score "
        f"({F('qwen15b-s42')*100:.1f}% to {F('qwen15b-syn')*100:.1f}%; one seed each, so the one-example difference is not meaningful). All five hand-written probes, including the planets, whales and "
        "abstract A-B-C arguments, are now classified correctly, as is one invalid probe (&ldquo;Some cats are pets. Some pets are dogs. Therefore, some cats are dogs&rdquo;). "
        "<b>Caveat.</b> The synthetic test shares its sentence templates with the synthetic training data and uses only held-out words, so 99% reflects robustness to new content words and not to entirely new phrasings. "
        "The five-probe check is anecdotal. The released model is this real + synthetic adapter.")]

# ---------------- 7 subgroup analysis
ps = R["qwen15b-s42"]["val_predictions"]


def g(v, p, preds=ps):
    s = [x for x in preds if x["label"] == v and x["plaus"] == p]
    return np.mean([x["label"] == x["pred"] for x in s]), len(s)


rows = [["Subgroup (validity, plausibility)", "n (validation)", "Seed 42", "Seed 123", "Seed 7"]]
for v, p, n in [(1, 1, "Valid, plausible"), (1, 0, "Valid, implausible"), (0, 1, "Invalid, plausible"), (0, 0, "Invalid, implausible")]:
    rows.append([n, g(v, p)[1]] + [f"{g(v, p, R[k]['val_predictions'])[0]:.3f}" for k in seeds])
S += [H1("8. CONTENT-EFFECT AND SUBGROUP ANALYSIS"),
      fig("confusion_subgroups.png", 15.5, "Seed-42 confusion matrix (left) and accuracy per subgroup (right; blue = plausible, gold = implausible)."),
      table(rows, [5.8, 2.6, 2.4, 2.4, 2.4], "Accuracy per subgroup for each seed."),
      P("The key question is whether plausibility changes accuracy. For seed 42 the four subgroups lie between 93.6% and 95.8%; implausible valid "
        "arguments are even slightly easier than plausible ones, the opposite of the human belief bias. The content-effect estimate is 0.011, "
        "close to zero. Seeds 123 and 7 show larger gaps (0.073 and 0.031), mostly because valid-plausible accuracy dips to 83% and 90%. "
        "With about 48 examples per subgroup, one example moves a subgroup by about 2 points, so these gaps are within noise and no consistent "
        "plausibility bias can be claimed in either direction. Errors in seed 42 are balanced between classes (6 false valid, 5 false invalid), "
        "so the model has no strong lean towards saying <i>valid</i> or <i>invalid</i>.")]

# ---------------- 8 error analysis
S += [H1("9. ERROR ANALYSIS"),
      fig("confidence.png", 12.5, "Model confidence (logit difference towards the true class) for correct and wrong validation predictions, seed 42."),
      P("The seed-42 model misclassifies 11 of 192 examples. Representative cases are shown below (true label, model verdict, and the model's "
        "confidence as the logit difference valid minus invalid)."),
      table([["Syllogism (abridged)", "True", "Model", "Logit diff"],
             ["Every car is a vehicle. At least one motorized vehicle is not a vehicle. There is at least one motorized vehicle that is not a car.", "Invalid", "Valid", "+5.7"],
             ["There are no animals that are also plants. At least one tree is a plant. At least one tree is not an animal.", "Invalid", "Valid", "+3.7"],
             ["Any animal that is a lion is a carnivore. No gazelle is a carnivore. It is impossible for a gazelle to be a lion.", "Valid", "Invalid", "-4.7"],
             ["There are no fiction books that are not non-fiction. Anything that is a novel is a fiction book. Therefore, some novels are non-fiction.", "Valid", "Invalid", "-4.4"],
             ["Every single liquid is wet. No sand is a liquid. Consequently, there are no things made of sand which are wet.", "Invalid", "Valid", "+1.3"]],
            [10.2, 1.8, 1.8, 2.3], "Selected misclassified validation examples (seed 42)."),
      P("Three patterns appear. <b>(1) Double negation and unusual phrasing.</b> &ldquo;No fiction books that are not non-fiction&rdquo; needs two "
        "negations to be unwound, and stacked negation is a known weak spot for language models. <b>(2) Ordinary-language quantifiers.</b> "
        "&ldquo;At least one motorized vehicle is not a vehicle&rdquo; contradicts the first premise, yet the model treats the argument as valid. "
        "<b>(3) Low-confidence mistakes.</b> Five of the eleven errors have an absolute logit difference below 1.5, so a confidence threshold "
        "could flag them for review. A caution: some labels look debatable to us (the lion and gazelle argument has the standard valid "
        "modus tollens form, yet the model rejects it confidently), so part of the remaining error may be label noise or the dataset's conventions "
        "for existential import, which we cannot verify without the official solutions.")]

# ---------------- 9 testing
S += [H1("10. TESTING, VALIDATION AND REPRODUCIBILITY"),
      table([["Check", "Outcome"],
             ["Split is stratified and identical across all runs (split seed 42)", "768 train / 192 validation"],
             ["Metrics code (accuracy, macro-F1, content effect)", "Same <font face='Courier'>src/metrics.py</font> for all systems"],
             ["Repeated training seeds", "3 seeds for the main model"],
             ["Ablations", "Prompt removal, 3B / 4-bit model, synthetic data"],
             ["Label generator", "Mood table verified against brute-force enumeration (0 mismatches)"],
             ["Final model", "Real + synthetic LoRA adapter in <font face='Courier'>outputs/lora-qwen15b-syn/adapter</font>; scored on the held-out validation split and a held-out synthetic set"],
             ["Figures and tables", "Generated by <font face='Courier'>scripts/make_figures.py</font> and <font face='Courier'>scripts/build_report.py</font> from saved metrics"]],
            [8.5, 7.6], "Validation checklist."),
      P("<b>Reproduce.</b> Install the requirements, run <font face='Courier'>python scripts/download_data.py</font>, then "
        "<font face='Courier'>python scripts/train_lora.py --model Qwen/Qwen2.5-1.5B --output outputs/lora-qwen15b-s42</font>. "
        "<font face='Courier'>scripts/run_all.sh</font> repeats the seed and ablation runs. Commands and results are also in the repository README and RESULTS.md.")]

# ---------------- 10 adv/lim
S += [H1("11. ADVANTAGES, LIMITATIONS AND FUTURE SCOPE"), H2("11.1 Advantages"),
      *B(["Large, consistent gain over the baselines at modest cost: about 10 minutes of training on a laptop GPU.",
          "Parameter-efficient: only about 1% of the weights are trained, so the saved adapter is small.",
          "Small content effect in the main run: validity is judged mostly on form, not plausibility.",
          "Reproducible, with a fixed split and reported seed variation."]),
      H2("11.2 Limitations"),
      *B(["Results are on a 192-example validation split, not the hidden competition test set, so they are estimates (standard error about 1.7 points).",
          "The content-effect and task scores are local approximations, not the official scorer.",
          "Only English Subtask 1 is covered; the multilingual subtasks are not attempted.",
          "Hyper-parameters were chosen once and not tuned, and the prompt was written by hand.",
          "The synthetic test shares templates with the synthetic training data, so it does not prove robustness to unseen phrasings.",
          "Some labels appear debatable, which may cap attainable accuracy."]),
      H2("11.3 Future scope"),
      *B(["Cross-validation over several splits and ensembling of seeds for a tighter, more stable estimate.",
          "Score with the official evaluation kit on held-out data.",
          "Data augmentation by recombining terms across logical forms, and nonsense-word syllogisms to push the content effect lower.",
          "Multilingual extension (Subtask 3) using multilingual decoder models.",
          "Evaluate on synthetic data with entirely new sentence templates, and average several seeds.",
          "Neuro-symbolic checking: parse into a logical form and verify it with a classical algorithm."])]

# ---------------- appendix
S += [H1("APPENDIX: REPOSITORY STRUCTURE"),
      Preformatted("""README.md                     Project explanation and run instructions
RESULTS.md                    Experiment log with every run
assets/amity_logo.png         Logo used on the report cover
scripts/download_data.py      Downloads the official Task 11 training JSON
scripts/majority_baseline.py  Majority-class baseline
scripts/train_subtask1.py     XLM-R baseline (original script)
scripts/train_lora.py         LoRA fine-tuning of Qwen2.5 (main method)
scripts/run_all.sh            Extra seeds, prompt ablation, 3B 4-bit run
scripts/predict.py            Classify new syllogisms with the saved adapter
scripts/gen_synthetic.py      Exact syllogism generator (mood/figure table)
scripts/eval_synthetic.py     Evaluate an adapter on a synthetic set
scripts/make_figures.py       Builds report figures from saved metrics
scripts/build_report.py       Builds this PDF
src/metrics.py                Accuracy, macro-F1, content-effect estimate
report/                       Report PDF, figures, summary.json""", code),
      PageBreak()]

# ---------------- 11 conclusion + refs
S += [H1("12. CONCLUSION"),
      P(f"We rebuilt the Task 11 Subtask 1 system around a LoRA-tuned decoder language model with an instruction prompt. On a fixed validation split "
        f"it reaches {m_acc*100:.1f}% &plusmn; {s_acc*100:.1f}% accuracy (best {max(acc)*100:.1f}%), compared with 60.9% for fine-tuned XLM-R and 50% "
        "for the majority class, and its accuracy shows no consistent dependence on plausibility. A manual check exposed that the real-data-only model fails some plain syllogisms; adding verified synthetic data fixed this (81.5% to 99.2% on a held-out synthetic set) with no loss on the real validation split. The prompt ablation shows the instruction matters, "
        "and a larger quantised model gives no extra gain here. The work shows that a small, carefully trained and carefully evaluated model can judge "
        "the logic of an argument even when its content is absurd, while the limits (small validation set, local metrics, possible label noise) keep us "
        "from over-claiming. The code, results and final adapter are in the project repository."),
      PageBreak(), H1("13. REFERENCES"),
      *[Paragraph(t, ParagraphStyle("r", parent=body, leftIndent=22, firstLineIndent=-22)) for t in [
          "[1] SemEval 2026 Task 11 organisers, task website and dataset (Subtask 1 training data). https://github.com/neuro-symbolic-ai/semeval_2026_task_11",
          "[2] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang and W. Chen, &ldquo;LoRA: Low-Rank Adaptation of Large Language Models,&rdquo; in ICLR, 2022.",
          "[3] T. Dettmers, A. Pagnoni, A. Holtzman and L. Zettlemoyer, &ldquo;QLoRA: Efficient Finetuning of Quantized LLMs,&rdquo; in NeurIPS, 2023.",
          "[4] A. Conneau et al., &ldquo;Unsupervised Cross-lingual Representation Learning at Scale,&rdquo; in ACL, 2020 (XLM-RoBERTa).",
          "[5] Qwen Team, &ldquo;Qwen2.5 Technical Report,&rdquo; arXiv:2412.15115, 2024.",
          "[6] A. Vaswani et al., &ldquo;Attention Is All You Need,&rdquo; in NeurIPS, 2017.",
          "[7] J. St. B. T. Evans, J. L. Barston and P. Pollard, &ldquo;On the conflict between logic and belief in syllogistic reasoning,&rdquo; Memory &amp; Cognition, vol. 11, pp. 295-306, 1983.",
          "[8] I. Loshchilov and F. Hutter, &ldquo;Decoupled Weight Decay Regularization,&rdquo; in ICLR, 2019 (AdamW).",
          "[9] T. Wolf et al., &ldquo;Transformers: State-of-the-Art Natural Language Processing,&rdquo; in EMNLP System Demonstrations, 2020.",
          "[10] S. Mangrulkar et al., &ldquo;PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods,&rdquo; Hugging Face, 2022."]]]

out = []
for f in S:
    if isinstance(f, Paragraph) and f.style.name in ("h1", "h2"):
        out.append(CondPageBreak(7 * cm if f.style.name == "h1" else 5 * cm))
    out.append(f)
doc.build(out)
print("wrote", OUT)
