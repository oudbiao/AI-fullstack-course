---
sidebar_position: 8
title: "End-to-End Project: AI Learning Assistant"
description: "A compact overview of the course-wide AI Learning Assistant project and how each stage adds one visible capability."
keywords: [AI learning assistant, AI project main line, portfolio, RAG project, Agent project]
---

# End-to-End Project: AI Learning Assistant

![AI Learning Assistant version roadmap](/img/course/ai-learning-assistant-roadmap-en.png)

The AI Learning Assistant is the recommended project thread for the whole course. It keeps your work from becoming scattered: every stage adds one small ability to the same product.

Use this page as the overview. For implementation details, see the [version roadmap](/intro/ai-learning-assistant-version-roadmap) and the [repository template](/intro/ai-learning-assistant-template).

## One Product, One Ability per Stage

| Stage | Ability added | Evidence |
| --- | --- | --- |
| 1 Tools | Project workspace exists | README, Git commit, run screenshot |
| 2 Python | Assistant records learning tasks | CLI output, `tasks.json`, error handling |
| 3 Data | Assistant summarizes learning records | cleaned data, charts, conclusions |
| 4-6 Models | Assistant classifies or predicts learning risk | baseline, metrics, error samples |
| 7 LLM/Prompt | Assistant generates plans or review cards | prompt versions, schema checks |
| 8 RAG | Assistant answers from course materials | retrieval logs, citations, eval questions |
| 9 Agent | Assistant plans steps and calls tools | trace, permission table, stop rule |
| 10-12 Extensions | Assistant handles one specialized medium | before/after examples and review notes |
| Capstone | Assistant becomes a demoable product | demo, README, evaluation report |

## The Product Rule

Do not build a huge product at the beginning. After each stage, add only:

1. One user action.
2. One run command.
3. One sample input and output.
4. One failure sample.
5. One next-step note.

This gives you a portfolio story without adding a separate project for every chapter.

## Final Standard

The final assistant does not need every feature. It should have a clear loop:

```text
learning goal or material -> context reading -> retrieval or tool use -> answer or plan -> evidence, logs, evaluation
```

If another person can run it, inspect examples, see failures, and understand your next improvement, the project is portfolio-ready.
