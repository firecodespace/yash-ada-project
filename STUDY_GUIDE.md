# Study guide: concepts with examples

Use this to prepare for your assessment. Each concept has a plain explanation and an example you can say out loud.

## 1. Syllogism, validity and plausibility

A **syllogism** has two premises and a conclusion about three terms.

> All dogs are animals. All planets are dogs. Therefore, all planets are animals.

- **Valid** means: *if both premises were true, the conclusion would have to be true.* It says nothing about whether the premises are actually true.
- **Plausible** means the content sounds realistic.

The two are independent, giving four kinds of example:

| | Plausible content | Implausible content |
|---|---|---|
| **Valid** | All cats are mammals. All mammals are animals. So all cats are animals. | All planets are dogs. All dogs are animals. So all planets are animals. |
| **Invalid** | All cats are mammals. All mammals are animals. So some animals are not cats. | All planets are dogs. Some dogs are tables. So all planets are tables. |

**Content effect (belief bias):** people and models tend to say "valid" when it *sounds* right and "invalid" when it sounds odd. The task tests whether the model judges form only.

## 2. The four statement types (A, E, I, O)

| Letter | Meaning | Example |
|---|---|---|
| A | All S are P | All dogs are animals |
| E | No S are P | No fish are mammals |
| I | Some S are P | Some pets are dogs |
| O | Some S are not P | Some animals are not cats |

A syllogism's **mood** is the three letters (major premise, minor premise, conclusion), e.g. **AAA** = "All… All… Therefore all…".
The **figure** (1-4) says where the middle term M sits in the premises. 4 letters^3 x 4 figures = 256 possible forms.

**Famous forms:** AAA in figure 1 is *Barbara* ("All M are P, all S are M, so all S are P"). EAE in figure 1 is *Celarent*.

A classic **invalid** pattern, "undistributed middle": *Some cats are pets. Some pets are dogs. So some cats are dogs.* The pets that are dogs could be different from the pets that are cats.

## 3. The exact classical algorithm (`src/logic.py`)

Think of a Venn diagram of three circles S, M, P. It has **8 regions**. Each region is either empty or has members, so there are 2^8 = **256 possible worlds**.

1. For each world, check whether both premises are true.
2. If both are true but the conclusion is false in some world, the argument is **invalid** (that world is a counterexample).
3. If no such world exists, it is **valid**.

Cost: O(256) per argument, which is constant time. So the logic is easy. The hard part is turning free English into S, M, P and A/E/I/O. A learned model does that step implicitly.

**Existential import:** traditional logic assumes every term has members. Under that assumption 24 forms are valid; under the modern reading only **15** are. Nine forms (such as "All M are P, all M are S, so some S are P") are valid only if M is non-empty. My generator uses only the 15, plus forms that are invalid either way, so no label depends on this convention.

## 4. Why a language model, and what "fine-tuning" means

A pretrained model like Qwen2.5 has read huge amounts of text. **Fine-tuning** continues training on our 960 labelled examples so it specializes in "valid or invalid".

We add a **classification head**: one linear layer on the last token's hidden vector that outputs two scores (invalid, valid). A softmax turns them into probabilities.

> Example: scores (-2.8, +6.6) become about 99.99% "valid".

## 5. LoRA (low-rank adaptation)

A model has weight matrices W (say 1536 x 1536 = 2.4 million numbers). Updating all of them needs lots of GPU memory.

LoRA **freezes W** and learns two small matrices: A (16 x 1536) and B (1536 x 16). The layer computes:

`y = W x + (alpha/r) * B(A x)`

The learned part has 2 x 16 x 1536 = about 49 thousand numbers instead of 2.4 million. Across the whole model that is **18.5M trainable of 1,562M total (1.18%)**. That is why it trains on a 6 GB laptop GPU in about 10 minutes.

## 6. The prompt

Each input is wrapped as: *"Judge only the logical form; ignore whether the statements are true in the real world. Syllogism: ... Is the argument logically valid?"*

This hint matters. Without it the same model dropped from 94.3% to 81.3%.

## 7. Training loop in plain words

- Epoch = one full pass over the training data. We use 3-4.
- Batch of 8, accumulated over 2 steps (effective batch 16).
- Learning rate 1e-4 with 10% warm-up, then cosine decay.
- Optimizer AdamW, precision bf16.
- Loss = cross-entropy: it punishes confident wrong answers heavily.

## 8. Metrics, with arithmetic

Validation set: 192 examples. We got 182 right:

- **Accuracy** = 182 / 192 = 94.8%.
- **Macro-F1:** compute F1 for "valid" and for "invalid", then average, so neither class can dominate.
- **Confusion matrix** (seed 42 run): 90 invalid correct, 6 false "valid", 5 false "invalid", 91 valid correct.
- **Content effect:** average accuracy gap between subgroups. Our seed-42 subgroups were 0.94, 0.96, 0.94, 0.94, so the gap is about 0.011 (near zero = good).
- **Standard error:** sqrt(0.93 x 0.07 / 192) = about 1.8 points. So 94.3% vs 94.8% is **not** a real difference.

## 9. Complexity (the ADA part)

Let n = tokens (~60), d = hidden width (1536), L = layers.

- Self-attention costs O(n^2 d); feed-forward and projections cost O(n d^2).
- Since n is much smaller than d, the d^2 term dominates, so one forward pass is about **O(L n d^2)**, linear in sentence length.
- LoRA adds only O(n r d) per adapted matrix, where r = 16.
- Optimizer memory drops from O(P) to about 1.2% of P.
- 4-bit quantization (the 3B run) stores frozen weights in about 4 bits instead of 16, roughly 4x less memory.

## 10. The robustness story (tell this one)

1. First model: 94.3% on validation.
2. I tested it by hand. It rated "All planets are dogs, all dogs are animals, so all planets are animals" as **invalid**, and the same for the whales example. Validation accuracy had hidden this because the test examples share the dataset's wording.
3. Fix: generate 1,800 examples from the 256-form table, verified against brute force, with real or invented words.
4. Result: held-out synthetic accuracy **81.5% to 99.2%**, real validation **94.3% to 94.8%**.
5. Honest caveat: the synthetic test reuses the same sentence templates, so it shows robustness to new *words*, not new *phrasings*.

## 11. Questions to expect, with short answers

- **Why a decoder LLM, not XLM-R?** Much more pretraining on reasoning-like text; XLM-R is a small encoder. Result: 60.9% vs about 93%. I did not run a controlled test separating size from architecture.
- **Is this the official score?** No. It is a local validation estimate on 192 examples.
- **Why three seeds?** One run can be lucky. We got 94.3, 92.2 and 92.7, so we report 93.1 +/- 0.9.
- **Why didn't the 3B model help?** 94.3% for both; with 192 examples, differences under about 3.6 points are noise.
- **Biggest weakness?** Small validation set, a synthetic test that shares templates, some debatable labels, and only English Subtask 1.
- **What would you do next?** Test on entirely new sentence templates, cross-validate, score with the official kit, and combine the model with a parser plus the exact algorithm.

## 12. Files to read, in order

1. `scripts/train_lora.py`: data split, prompt, LoRA, training, evaluation.
2. `scripts/gen_synthetic.py` and `src/logic.py`: the synthetic data and the exact checker.
3. `src/metrics.py`: accuracy, macro-F1, content effect.
4. `scripts/demo.py` + `scripts/demo.html`: the live demo.
5. `report/Project_Report_ADA.pdf`: sections 4, 5 and 7.

## 13. Running the live demo

```powershell
.venv311\Scripts\python scripts\demo.py
```

Wait for "Ready: http://127.0.0.1:8000" (about 20-30 seconds to load once), then open that address. Use the example chips, the "Generate and compare" panel (model vs exact algorithm), and the "Test 40 random syllogisms" button. Each check takes about 0.1 s.
