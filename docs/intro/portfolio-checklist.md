---
sidebar_position: 21
title: "Portfolio Acceptance Checklist"
description: "A compact checklist for turning AI course projects into portfolio work that is runnable, reproducible, evaluable, and explainable."
keywords: [portfolio acceptance, AI project, README, project retrospective, capstone project]
---

# Portfolio Acceptance Checklist

![Portfolio acceptance ladder](/img/course/intro-portfolio-acceptance-ladder-en.png)

A portfolio project does not need many features. It needs evidence: it runs, others can reproduce it, the output can be evaluated, and you can explain the trade-offs.

## 1. Acceptance Ladder

| Level | Minimum evidence | Not ready if... |
|---|---|---|
| Practice | Code runs locally and prints a result | Only screenshots exist |
| Project | README, run command, sample inputs and outputs | Another person cannot reproduce it |
| Portfolio | Evaluation set, logs, failure samples, demo screenshots | It only shows success cases |
| Interview | Architecture, metrics, limits, cost, security, alternatives | You cannot explain why choices were made |

Each stage project should reach at least Project level. A capstone should aim for Portfolio level.

## 2. One-Minute Quick Check

Before sharing a project, confirm these eight items:

| Item | Passes when... |
|---|---|
| README | The goal, user, and feature list are clear |
| Run command | A fresh reader can run the smallest flow |
| Dependencies | `requirements.txt`, `pyproject.toml`, or `package.json` exists |
| Examples | 1-3 real inputs and outputs are shown |
| Screenshot or demo | The user can see what the project does |
| Evaluation | There is a fixed question set, metric, or manual scoring rule |
| Failure samples | Known bad cases and causes are recorded |
| Next step | The next improvement is specific, not “optimize later” |

If three or more items are missing, keep it as practice code for now.

## 3. AI Evidence by Project Type

| Project type | Evidence to keep |
|---|---|
| Machine learning | Baseline, train/test split, metric, error examples |
| Deep learning | Training log, validation curve, overfitting check |
| Prompt assistant | Prompt versions, input/output examples, failure cases |
| RAG | Document source, chunking choice, retrieval logs, citation check |
| Agent | Tool schema, execution trace, stop condition, permission boundary |
| Multimodal | Input format, processing flow, human review, quality criteria |

The core of a RAG project is reliable retrieval and citations. The core of an Agent project is traceable execution and controllable actions.

## 4. Three-Minute Demo Script

Use this order when presenting:

1. Who the user is and what problem they have.
2. What the smallest working feature does.
3. Where AI enters the workflow.
4. How you tested the output.
5. What failed before and what you changed.
6. What you would improve next.

## 5. Minimum README Shape

```md
# Project Name

## Goal

## How to Run

## Example Inputs and Outputs

## Evaluation

## Known Failures

## Next Step
```

Short and reproducible beats long and vague.
