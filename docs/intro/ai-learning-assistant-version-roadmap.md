---
sidebar_position: 10
title: "AI Learning Assistant Version Roadmap"
description: "A compact version roadmap that turns each course stage into one runnable, evaluable AI Learning Assistant release."
keywords: [AI Learning Assistant, version roadmap, AI portfolio, RAG project, Agent project]
---

# AI Learning Assistant Version Roadmap

![AI Learning Assistant version roadmap](/img/course/ai-learning-assistant-roadmap-en.png)

Each version should add one capability and leave four kinds of evidence:

| Required evidence | Why it matters |
| --- | --- |
| Run command | Proves the feature is reproducible |
| Sample input/output | Shows what the feature does |
| Failure sample | Shows the boundary |
| Evaluation or check | Prevents a lucky demo |

## Version Cards

| Version | Course stage | Capability | Minimum evidence |
| --- | --- | --- | --- |
| v0.1 | Chapter 1 | Project skeleton | README, dependency note, first commit |
| v0.2 | Chapter 2 | CLI learning task assistant | `tasks.json`, add/view/complete commands, broken-file handling |
| v0.3 | Chapter 3 | Learning data analysis | charts, cleaning note, one conclusion |
| v0.4 | Chapters 4-5 | Question classification or risk baseline | rule or ML baseline, metrics, error samples |
| v0.5 | Chapter 6 | Training diagnosis experiment | config, loss curve, failed run note |
| v0.7 | Chapter 7 | Prompt learning assistant | prompt versions, schema checks, failed outputs |
| v0.8 | Chapter 8 | RAG course Q&A | document list, retrieval logs, citation checks |
| v0.9 | Chapter 9 | Learning planning Agent | tool trace, permission rule, stop condition |
| v1.0 | Chapters 10-12 / capstone | Demoable product or extension | demo, README, evaluation report |

Version numbers are flexible. The evidence is not.

## Version Note Template

```md
## v0.x Version Name

Goal:
Run command:
Sample input:
Sample output:
Evaluation/check:
Failure sample:
Next version:
```

When presenting the project, explain the version line instead of only showing the final screenshot. A strong story is: “I started with a CLI, then added data, evaluation, RAG, traces, and deployment evidence.”
