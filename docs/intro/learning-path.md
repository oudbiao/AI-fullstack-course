---
sidebar_position: 3
title: "0.4 Recommended Learning Path"
description: "Choose one practical learning path first, then enter Chapter 1."
keywords: [AI learning path, AI full-stack path, LLM learning path, Agent learning path, RAG learning path]
---

# 0.4 Recommended Learning Path

![Recommended Learning Path Selection Diagram](/img/course/intro-learning-path-selection-en.webp)

If you are unsure, choose the beginner path: **Chapter 1 -> Chapter 9 in order, one small output per stage, then choose one specialization from Chapters 10-12 only when you need it.**

| Your goal | First route |
|---|---|
| I am new | Follow Chapters 1-9 before branching |
| I already code | Skim 1-6, focus on 7-9, then pick a specialization |
| I need a portfolio | Keep README, screenshots, logs, metrics, traces, failures |
| I care about models | Spend more time on math, ML, DL, Transformer, then choose CV/NLP/multimodal depth |

## Pick A Route

| Route | Best for | Chapters | What to produce |
|---|---|---|---|
| Beginner full path | New learners or career switchers | 1 -> 9, then one of 10-12 | One runnable mini project after each main track, plus one specialization demo |
| Builder path | Developers who want LLM apps quickly | skim 1-6, focus 7 -> 9, then 12 if multimodal is needed | RAG app, Agent trace, evaluation notes, safety boundary |
| Model path | Learners who want deeper ML intuition | 1 -> 7, then 10 or 11 based on data type | Model experiments, metric comparisons, failure analysis |
| Portfolio path | Job-focused learners | 1 -> 9 with stronger README work, then one capstone direction | A public project story with setup, screenshots, logs, metrics, traces, limits |

## Stage Exit Checks

Do not judge progress by pages read. Judge it by evidence.

| Stage | Chapters | Minimum evidence | Deeper evidence for experienced learners |
|---|---|---|---|
| Foundations | 1-3 | A reproducible project folder, Python scripts, cleaned data, charts | README rerun test, edge cases, data quality notes |
| Model understanding | 4-6 | One model experiment with a metric and failure samples | Bias/variance notes, ablation, training diagnosis |
| LLM applications | 7-9 | Prompt tests, RAG retrieval trace, Agent tool trace | Fixed eval set, safety boundary, cost/latency notes |
| Specialization | 10-12 | One vision, NLP, or multimodal demo with saved inputs and outputs | Domain metric, review checklist, deployment constraint |

The specialization chapters are not a reward for finishing everything. They are a deliberate branch: choose them when the product needs images, text pipelines, multimodal assets, or domain-specific evaluation.

## Weekly Loop

Use the same loop every week:

```text
read briefly -> run one thing -> change one condition -> record evidence -> write one reflection
```

The reflection can be short. Good examples:

- What failed first?
- What input changed the output most?
- What evidence would convince another developer?
- What would break if this became a real user-facing feature?

## When To Skip Or Slow Down

Skip only when you can pass the chapter check without guessing. Slow down when you cannot explain the output, cannot rerun the code, or cannot tell whether the result is good. Experienced learners should slow down on evaluation, failure modes, and production constraints even when the demo feels easy.

Do not switch routes every week. Read briefly, run something, keep evidence, then continue to [Chapter 1](/ch01-tools).
