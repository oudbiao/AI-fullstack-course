---
title: "11.7.1 Project Roadmap: Build an Evaluatable NLP Pipeline"
sidebar_position: 0
description: "A concise hands-on roadmap for NLP projects: define task boundaries, build baselines, evaluate outputs, analyze failures, and package evidence."
keywords: [NLP Project Guide, QA, Summarization, Information Extraction, NLP Portfolio]
---

# 11.7.1 Project Roadmap: Build an Evaluatable NLP Pipeline

An NLP project is not a fluent paragraph. It is a clear task boundary, data source, baseline, evaluation method, failure analysis, and structured deliverable.

## See the Project Evidence Loop First

![NLP project delivery loop](/img/course/ch11-projects-delivery-loop-en.webp)

![NLP evidence pack diagram](/img/course/ch11-nlp-evidence-pack-en.webp)

![Workshop text to artifacts pipeline map](/img/course/ch11-workshop-text-to-artifacts-pipeline-map-en.webp)

Start with information extraction or classification for clear labels. Move to summarization and QA when you can evaluate factuality, refusal, citations, and boundaries.

## Run a Project Readiness Check

```python
project = {
    "task": "information extraction",
    "has_schema": True,
    "has_baseline": True,
    "has_eval_cases": True,
    "has_failure_case": True,
}

ready = all(project[key] for key in ["has_schema", "has_baseline", "has_eval_cases", "has_failure_case"])

print("task:", project["task"])
print("portfolio_ready:", ready)
```

Expected output:

```text
task: information extraction
portfolio_ready: True
```

If labels, fields, or knowledge boundaries are unclear, fix the task definition before changing models.

## Learn in This Order

| Step | Project | Evidence |
|---|---|---|
| 1 | Information extraction | Schema, field boundaries, precision/recall, failure examples |
| 2 | Text classification | Labels, baseline, F1, ambiguity cases |
| 3 | Summarization | Compression, factuality, readability, missing facts |
| 4 | QA | Retrieval, citation, refusal, no-answer evaluation |
| 5 | Hands-on workshop | Reproducible mini pipeline before larger project pages |

Run [11.7.6 Hands-on: Build a Reproducible NLP Mini Pipeline](./05-hands-on-nlp-workshop.md) before expanding the project.

## Project Deliverable Standards

| Deliverable | Minimum Requirement | Stronger Portfolio Version |
|---|---|---|
| README | Goal, run command, dependencies, examples | Add task boundary, data source, trade-offs, review summary |
| Label/schema | Labels, entity boundaries, or output fields | Add positive, negative, boundary examples, consistency notes |
| Baseline | Keyword, TF-IDF, rule, or simple model | Add model comparison and error attribution |
| Evaluation | Accuracy, recall, F1, human score, or factuality check | Add analysis by label, length, domain, and noise type |
| Failure case | At least 1 real failure | Add cause, fix action, regression check |
| Presentation | Screenshot or short GIF proving it runs | Build a clear text-understanding project page |

## Pass Check

You pass this chapter when your NLP project has a task definition, data examples, evaluation metric, baseline, failure case, and next-step improvement plan.
