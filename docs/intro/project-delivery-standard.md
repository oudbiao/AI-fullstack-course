---
sidebar_position: 16
title: "Project Delivery Standard"
description: "The minimum evidence every stage project should deliver: goal, run command, examples, evaluation, failures, and next step."
keywords: [project page template, README standard, project delivery, portfolio, AI project]
---

# Project Delivery Standard

![Project quick reference map](/img/course/appendix-project-quick-reference-map-en.png)

This page answers: “What must I hand in?” Use it after finishing a stage project.

## 1. Delivery Loop

| Step | Evidence |
|---|---|
| Goal | Who uses it, what problem it solves, what output it gives |
| Run | Installation notes, environment variables, one smallest run command |
| Example | Real input and real output |
| Evaluate | Metric, checklist, fixed questions, or manual review rule |
| Failure | One reproducible bad case and likely cause |
| Next | One specific improvement for the next version |

If time is short, keep the run command, example output, and failure case first.

## 2. Minimum README

```md
# Project Name

## Goal

## How to Run

## Example Input and Output

## Evaluation

## Known Failure

## Next Step
```

## 3. Extra Evidence for AI Projects

| Project | Add this |
|---|---|
| Prompt | Prompt version, fixed test samples, structured validation |
| RAG | chunking choice, retrieval logs, citation check |
| Agent | tool schema, trace, max steps, permission boundary |
| Multimodal | source materials, generation settings, human review checklist |

Keep the delivery small but concrete. A short README that someone can run is stronger than a long plan.
