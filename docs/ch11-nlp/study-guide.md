---
title: "11.0 Learning Checklist: Natural Language Processing"
sidebar_position: 1
description: "A compact checklist for Chapter 11: text cleaning, tokenization, representation, labels, extraction, generation, metrics, and portfolio evidence."
keywords: [NLP checklist, text classification, information extraction, BERT, GPT, text evaluation]
---

# 11.0 Learning Checklist: Natural Language Processing

Use this page as a printable checklist. If you need the full explanation, return to the [Chapter 11 entry page](./index.md).

![NLP portfolio evidence pack](/img/course/ch11-nlp-evidence-pack-en.png)

## Two-Hour First Pass

| Time box | Do this | Stop when you can say |
|---|---|---|
| 20 min | Read the text-to-task pipeline | "NLP starts with raw text and ends with evaluable outputs." |
| 25 min | Run the label evaluation script | "I can compare predicted labels with expected labels." |
| 25 min | Skim 11.1 text preprocessing | "Cleaning can help or harm depending on meaning." |
| 25 min | Skim classification, extraction, and generation roadmaps | "The task is defined by the output." |
| 25 min | Read the task output map | "I can choose metrics from the output type." |

## Required Evidence

| Evidence | Minimum version |
|---|---|
| `text_cleaning.py` | cleaning, tokenization, before/after examples |
| `label_guide.md` | label definitions, boundary cases, positive and negative examples |
| `classification_report.md` | metrics, confusion matrix or error table, model comparison |
| `extraction_examples.jsonl` | source text, extracted fields, validation result |
| `failure_cases.md` | confusing labels, missing fields, unsupported facts, bad summaries |
| `README.md` | task goal, run command, input/output, metrics, limitations |

## Exit Questions

- Can you explain how raw text becomes tokens and model input?
- Can you define label boundaries before training or prompting?
- Can you decide whether a task needs classification, extraction, retrieval, or generation?
- Can you evaluate factual consistency for summaries or answers?
- Can you explain when a traditional NLP method is enough and when an LLM is helpful?

If the answer is yes, you can use NLP ideas more confidently in Prompt, RAG, Agent memory, and multimodal work.
