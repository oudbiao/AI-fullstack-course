---
sidebar_position: 10
title: "AI Learning Assistant Repository Template"
description: "A compact repository structure, README, evaluation, and trace template for the course-wide AI Learning Assistant project."
keywords: [AI Learning Assistant, project template, portfolio project, RAG project template, Agent project template]
---

# AI Learning Assistant Repository Template

![AI Learning Assistant repository evidence cabinet](/img/course/intro-ai-assistant-repo-evidence-cabinet-en.png)

Treat the repository as an evidence cabinet. Every folder should prove one thing: the project can run, be reviewed, be evaluated, or be improved.

## 1. Start With This

```text
ai-learning-assistant/
  README.md
  requirements.txt
  .env.example
  src/
  data/
  evals/
  logs/
  docs/
  tests/
```

In chapters 1-3, use only `README.md`, `src/`, `data/`, and `docs/`. Add `evals/`, `logs/`, RAG, and Agent files when the course reaches those topics.

## 2. Folder Proof

| Folder | Proof |
|---|---|
| `src/` | The project has runnable code |
| `data/` | Inputs and materials are explicit |
| `evals/` | Results can be judged again |
| `logs/` | Failures and traces can be reviewed |
| `docs/` | Screenshots and decisions are visible |
| `tests/` | Fixes can be checked later |

## 3. Minimum README

````md
# AI Learning Assistant

## Goal

## How to Run
```bash
pip install -r requirements.txt
python -m src.app
```

## Example

## Evaluation

## Known Failure

## Next Step
````

## 4. First Eval and Trace Files

```jsonl
{"id":"q001","question":"Why does RAG need citations?","expected_sources":["ch08-rag"]}
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

When presenting the project, show the command, sample input, eval case, trace, failure note, and screenshot.
