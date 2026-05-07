---
sidebar_position: 10
title: "AI Learning Assistant Repository Template"
description: "A compact repository structure, README, eval, and trace template for the course-wide AI Learning Assistant project."
keywords: [AI Learning Assistant, project template, portfolio project, RAG project template, Agent project template]
---

# AI Learning Assistant Repository Template

![AI Learning Assistant repository evidence cabinet](/img/course/intro-ai-assistant-repo-evidence-cabinet-en.png)

This template is not a directory decoration. It is an evidence cabinet: code, data, logs, evaluations, and screenshots all explain whether the project can be run and reviewed.

## Minimum Directory Structure

```text
ai-learning-assistant/
  README.md
  requirements.txt
  .env.example
  src/
    app/
    rag/
    agent/
  data/
    raw/
    processed/
  evals/
    questions.jsonl
    results/
  logs/
    traces/
    failures/
  docs/
    screenshots/
    decisions.md
  tests/
```

Start small. In Chapters 1-3, you only need `README.md`, `src/`, `data/`, and `docs/screenshots/`. Add `evals/`, `logs/`, `rag/`, and `agent/` when the course reaches those capabilities.

## What Each Folder Proves

| Folder | Proof |
| --- | --- |
| `src/` | The system has runnable code |
| `data/` | Inputs and materials are explicit |
| `evals/` | Results can be judged |
| `logs/` | Failures and traces can be reviewed |
| `docs/` | Others can understand the project |
| `tests/` | Fixes can be checked again |

## Minimum README

````md
# AI Learning Assistant

## Goal
What learning problem does this assistant solve?

## Current Version
v0.x:

## How to Run
```bash
pip install -r requirements.txt
python -m src.app.cli
```

## Example
Input:
Output:

## Evaluation
What fixed questions, metrics, or manual checks are used?

## Failure Samples
What failed and what will be changed next?
````

## Minimal Eval and Trace Examples

```jsonl
{"id":"q001","question":"Why does RAG need citations?","expected_sources":["ch08-rag"],"ideal_points":["grounding","evaluation","failure cases"]}
```

```json
{
  "run_id": "demo-001",
  "user_input": "Help me review RAG",
  "steps": [
    {"action": "retrieve", "sources": ["ch08-rag"]},
    {"action": "generate_plan", "status": "ok"}
  ],
  "failure": null
}
```

When presenting the project, show the repository as evidence: run command, sample data, eval cases, trace logs, failure notes, and screenshots.
