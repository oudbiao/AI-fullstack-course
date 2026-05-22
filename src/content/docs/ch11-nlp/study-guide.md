---
title: "11.0 Learning Checklist: Natural Language Processing"
description: "A compact checklist for Chapter 11: text cleaning, tokenization, representation, labels, extraction, generation, metrics, and portfolio evidence."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "NLP checklist, text classification, information extraction, BERT, GPT, text evaluation"
---

# 11.0 Learning Checklist: Natural Language Processing

Use this page as a printable checklist. If you need the full explanation, return to the [Chapter 11 entry page](./index.md).

![NLP portfolio evidence pack](/img/course/ch11-nlp-evidence-pack-en.webp)

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

## Quality Gates

| Gate | Pass condition |
|---|---|
| Label/schema boundary | Labels or fields include positive, negative, and edge examples. |
| Baseline | Rule, TF-IDF, simple model, or LLM baseline runs on the same fixed eval cases. |
| Factuality | Generated summaries or answers are checked against source evidence, not only fluency. |
| Error review | Confusion, missing fields, unsupported facts, and bad summaries have a cause and next test. |

## Exit Questions

- Can you explain how raw text becomes tokens and model input?
- Can you define label boundaries before training or prompting?
- Can you decide whether a task needs classification, extraction, retrieval, or generation?
- Can you evaluate factual consistency for summaries or answers?
- Can you explain when a traditional NLP method is enough and when an LLM is helpful?

If the answer is yes, you can use NLP ideas more confidently in Prompt, RAG, Agent memory, and multimodal work.

<details>
<summary>Check reasoning and explanation</summary>

1. A strong answer explains the path from raw text to tokens, representation, model input, prediction, metric, and failure case.
2. Label boundaries are ready only when you have positive examples, negative examples, edge cases, and a written rule for disagreements.
3. Choose classification for fixed labels, extraction for fields, retrieval for evidence lookup, generation for new text, and hybrids when outputs require multiple steps.
4. Factual consistency means each generated summary or answer can be traced to source evidence; fluency alone is not enough.
5. Traditional NLP is enough when the task is small, transparent, and stable; LLMs help when language variation, generation, or reasoning over context dominates.

</details>

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
task_output: label, entity fields, summary, answer, retrieval result, or semantic graph
artifacts: raw text, processed text, predictions, metrics, and failure cases
metric: accuracy/F1, precision/recall, retrieval hit rate, faithfulness, or schema validity
failure_check: unclear labels, over-cleaning, boundary errors, hallucination, or unsupported answer
Expected_output: reproducible text pipeline folder with metrics and examples
```
