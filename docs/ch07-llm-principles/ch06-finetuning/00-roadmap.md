---
title: "7.6.1 Finetuning Roadmap: Data, LoRA, Evaluation"
sidebar_position: 0
description: "A concise hands-on roadmap for finetuning: decide when training is worth it, prepare examples, understand LoRA/QLoRA/PEFT, and evaluate against a prompt baseline."
keywords: [finetuning guide, LoRA, QLoRA, PEFT, LLM finetuning]
---

# 7.6.1 Finetuning Roadmap: Data, LoRA, Evaluation

Finetuning changes model behavior by training on examples. It is useful for stable task patterns, repeated formats, domain style, or behavior habits. It is usually not the first fix for missing private knowledge; that is often a RAG problem.

## 7.6.1.1 See the Decision Loop First

![Relationship diagram of the large model finetuning chapter](/img/course/ch07-finetuning-chapter-flow-en.png)

![Finetuning decision and evaluation loop diagram](/img/course/ch07-finetuning-decision-loop-en.png)

![Fine-tuning engineering workflow comic](/img/course/ch07-finetuning-engineering-loop-en.png)

Key terms: LoRA means low-rank adapters, QLoRA means quantized LoRA, and PEFT means parameter-efficient fine-tuning. They reduce cost by training a small set of extra parameters instead of every model weight.

## 7.6.1.2 Run a Finetuning Route Check

Use this check before you start training. A finetuning run without a prompt baseline, validation set, and failure log is hard to judge.

```python
case = {
    "private_facts": False,
    "format_drift": True,
    "stable_task": True,
    "labeled_examples": 120,
}

if case["private_facts"]:
    route = "RAG first"
elif case["format_drift"] and case["stable_task"] and case["labeled_examples"] >= 50:
    route = "fine-tuning candidate"
else:
    route = "prompt baseline first"

print("route:", route)
print("minimum_before_training:", ["prompt baseline", "validation set", "failure log"])
```

Expected output:

```text
route: fine-tuning candidate
minimum_before_training: ['prompt baseline', 'validation set', 'failure log']
```

Change one value at a time and rerun it. For example, set `private_facts` to `True`; the decision should move to RAG first.

## 7.6.1.3 Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Finetuning overview | Write when to use prompt, RAG, or finetuning |
| 2 | LoRA / QLoRA | Explain what parameters are trained and why cost drops |
| 3 | Other PEFT methods | Know that full finetuning is not the only path |
| 4 | Finetuning practice | Prepare train/validation examples and one run command |
| 5 | Data labeling | Audit samples for format, duplicates, leakage, and edge cases |

## 7.6.1.4 Pass Check

You pass this chapter when you can say why finetuning is worth trying, show the baseline it beats, and keep a validation set that was not used for training.

The exit mini project is a small instruction-tuning plan: choose one fixed task, prepare dozens to hundreds of examples, define a prompt baseline, and compare format stability or accuracy after a LoRA/QLoRA run.
