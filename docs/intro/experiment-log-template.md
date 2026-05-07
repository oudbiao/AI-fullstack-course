---
sidebar_position: 13
title: "Experiment Log and README Templates"
description: "Copyable README, experiment log, and failure sample templates for turning practice into portfolio evidence."
keywords: [experiment log template, README template, AI project retrospective, portfolio]
---

# Experiment Log and README Templates

![AI product experiment metrics loop](/img/course/elective-ai-product-experiment-metrics-loop-en.png)

Use these templates when a project has real commands, outputs, metrics, or failures. Keep them short; a template that nobody fills in is just noise.

## Minimum README Template

````md
# Project Name

## Goal
What problem does this solve? Who is it for?

## How to Run
```bash
python main.py
```

## Example Input and Output
Input:

Output:

## Evaluation or Check
How do you know the result is acceptable?

## Failure Sample
What failed, why, and how will you test the fix?

## Next Step
What changes in the next version?
````

## Experiment Log Template

| Field | Fill in |
| --- | --- |
| `experiment_id` | `rag_exp_003` |
| Goal | What you want to verify |
| Baseline | What you compare against |
| Change | The one main thing changed this time |
| Config | Model, prompt, retrieval, agent, or training settings |
| Metrics | Accuracy, Hit@k, citation_ok, latency, cost, or manual score |
| Result | What improved and what got worse |
| Decision | Keep, reject, or retry with changes |

## Failure Sample Template

| Field | Fill in |
| --- | --- |
| Input | The exact input that failed |
| Expected | What should have happened |
| Actual | What happened |
| Layer | environment / data / model / prompt / RAG / Agent / deployment |
| Cause | Best current explanation |
| Fix | What you changed |
| Regression check | How you will know it does not return |

## Recommended Files

```text
reports/
  failure_cases.md
  improvement_record.md
evals/
  eval_questions.csv
  citation_check.csv
logs/
  llm_calls.jsonl
  retrieval_logs.jsonl
  agent_traces.jsonl
```

The point of logging is not paperwork. It proves that you can evaluate, debug, and improve a system rather than only show a successful demo.
