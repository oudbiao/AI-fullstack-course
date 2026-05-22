---
title: "6.8.4 Project: Generative Models in Practice [Optional]"
description: "Build a generative project review loop with sample checkpoints, quality notes, diversity checks, failure cases, and portfolio presentation."
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "generative project, GAN, VAE, generation quality, diversity, evaluation"
---
:::tip[Section Overview]
A generative project is not finished when it produces one nice sample. You need to show quality, diversity, stability, failures, and why one checkpoint is worth keeping.
:::
## Learning Objectives

- Explain why generative projects need different evaluation from classification.
- Track quality and diversity together.
- Build a small checkpoint review table.
- Identify mode collapse and blurry-output failure modes.
- Package generated samples as project evidence.

---

## See the Evaluation Loop First

![Generative model project evaluation loop](/img/course/ch06-project-generative-eval-loop-en.webp)

```text
train -> sample checkpoints -> review quality + diversity -> keep failures -> choose next step
```

For a practice project, choose a generation target that is:

- visually inspectable;
- small enough to train or simulate;
- easy to compare across checkpoints.

Digits, icons, simple shapes, or tiny grayscale patterns are better first projects than open-ended photorealistic generation.

## Lab: Checkpoint Review Dashboard

Create `generative_review_dashboard.py`:

```python
checkpoints = [
    {"epoch": 1, "quality": 0.20, "diversity": 0.80, "note": "mostly noise"},
    {"epoch": 10, "quality": 0.45, "diversity": 0.72, "note": "outlines appear"},
    {"epoch": 30, "quality": 0.68, "diversity": 0.60, "note": "usable but varied"},
    {"epoch": 60, "quality": 0.75, "diversity": 0.48, "note": "possible collapse"},
]

print("generation_review")
for row in checkpoints:
    status = "candidate" if row["quality"] >= 0.6 and row["diversity"] >= 0.55 else "review"
    print(
        f"epoch={row['epoch']:03d} "
        f"quality={row['quality']:.2f} "
        f"diversity={row['diversity']:.2f} "
        f"status={status}"
    )

selected = max(
    [row for row in checkpoints if row["diversity"] >= 0.55],
    key=lambda row: row["quality"],
)
print("selected_epoch:", selected["epoch"])
```

Run it:

```bash
python generative_review_dashboard.py
```

Expected output:

```text
generation_review
epoch=001 quality=0.20 diversity=0.80 status=review
epoch=010 quality=0.45 diversity=0.72 status=review
epoch=030 quality=0.68 diversity=0.60 status=candidate
epoch=060 quality=0.75 diversity=0.48 status=review
selected_epoch: 30
```

![Checkpoint review result map for generative models](/img/course/ch06-generative-checkpoint-selection-result-map-en.webp)

Why not pick epoch 60? Because quality is higher but diversity is lower. A good generative project does not select only the prettiest sample.

## What to Save

| Evidence | Why |
|---|---|
| samples by checkpoint | shows training progression |
| failure samples | reveals limits honestly |
| diversity notes | catches repeated outputs |
| quality notes | explains visual improvements |
| training logs | shows stability or collapse |
| final selection rule | makes the choice reproducible |

## Quality, Diversity, Stability

| Dimension | Good sign | Warning sign |
|---|---|---|
| Quality | samples look like target data | noisy, blurry, broken structure |
| Diversity | samples vary meaningfully | repeated outputs or one dominant style |
| Stability | checkpoints improve gradually | sudden collapse or oscillation |
| Interpretability | failures are documented | only best samples are shown |

The common trade-off:

```text
best-looking single sample != best project checkpoint
```

## Project Upgrade Path

| Version | What to add |
|---|---|
| basic | one model, fixed sampling seed, checkpoint samples |
| standard | quality/diversity table and failure samples |
| challenge | compare VAE, GAN, or diffusion-style outputs |
| portfolio | clear story: data, model, samples, failures, next step |

## Evidence to Keep

A generative project should leave this minimum evidence:

```text
checkpoint_samples: fixed-seed samples across epochs
quality_note: what improved visually
diversity_note: whether outputs repeat
failure_sample: blurry, broken, collapsed, or unrealistic output
selection_rule: why this checkpoint was kept
next_action: data, objective, architecture, or sampling change
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| showing only best samples | show average and failure samples too |
| ignoring diversity | track repeated outputs or unique patterns |
| comparing checkpoints by memory | use the same fixed seed set |
| using a dataset too complex at first | start with small visual targets |
| not explaining model choice | state why VAE, GAN, or another method fits the goal |

## Exercises

1. Add an epoch `90` with quality `0.80` and diversity `0.30`. Should it be selected?
2. Add a `failure` field to each checkpoint.
3. Write a 4-row table for your own generative project idea.
4. Explain mode collapse using the checkpoint table.
5. Draft a portfolio section titled “Why I selected this checkpoint.”

<details>
<summary>Project reference and review notes</summary>

1. Usually no, unless the project values quality far more than diversity. A diversity score of `0.30` is a warning sign for repeated or narrow outputs.
2. The `failure` field should name visible problems such as repetition, artifacts, prompt mismatch, unsafe output, or poor diversity.
3. A useful table has rows for idea, data/source, evaluation signal, and main risk. The table should help someone judge whether the project can be evaluated.
4. Mode collapse means the model produces a small set of similar outputs. In the checkpoint table, it looks like acceptable quality with low diversity.
5. The portfolio section should justify the selected checkpoint with evidence: quality, diversity, failure notes, sample outputs, and why rejected checkpoints were weaker.

</details>

## Key Takeaways

- Generative projects need evaluation stories, not just galleries.
- Quality and diversity must be read together.
- Failure samples make the project more credible.
- A clear checkpoint selection rule is part of the deliverable.
