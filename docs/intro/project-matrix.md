---
sidebar_position: 12
title: "Full-Course Project Matrix"
description: "A compact matrix showing the smallest useful project for each course stage and the portfolio evidence to keep."
keywords: [AI project matrix, portfolio, stage projects, AI full-stack project]
---

# Full-Course Project Matrix

![Project portfolio roadmap](/img/course/intro-project-portfolio-roadmap-en.png)

Use this page to answer one question: “What should I build at each stage?” Keep the first pass small. Upgrade only the projects you want to show in a portfolio.

## 1. Stage Project Map

| Stage | Smallest useful project | Evidence to keep |
|---|---|---|
| 1 Tools | Create a repo and run Python | README, command record, screenshots |
| 2 Python | CLI task manager or simple API | Run command, sample input/output |
| 3 Data | Analyze one CSV and draw charts | Cleaning notes, charts, conclusion |
| 4 AI Math | Vector, probability, or gradient mini experiment | Diagram, tiny code, explanation |
| 5 Machine Learning | Baseline classification or regression | Metric, baseline, error samples |
| 6 Deep Learning | PyTorch training loop | Config, curve, failed samples |
| 7 LLM and Prompt | Prompt assistant or review-card generator | Prompt versions, output comparison |
| 8 RAG | Markdown Q&A with citations | Questions, retrieval logs, citation check |
| 9 Agent | Tool-calling planning Agent | Tool schema, execution trace, limits |
| 10 Vision | Classification, OCR, or detection experiment | Labels, metrics, visual results |
| 11 NLP | Text classification or extraction project | Label rules, metrics, error cases |
| 12 Multimodal | Image, audio, video, or multimodal workflow | Source materials, generation logs, review rules |

## 2. Upgrade Rule

| Version | Goal | Add this evidence |
|---|---|---|
| Minimum | Prove it runs | README, command, sample output |
| Standard | Prove others can reproduce it | Dependency file, config notes, logs |
| Portfolio | Prove it can be evaluated and explained | Evaluation set, failures, screenshots, retrospective |

Challenge projects are optional. They are useful after the main path is stable, not before the first working loop exists.

## 3. Suggested Repository Shape

```text
ai-fullstack-portfolio/
├── ch01-tools/
├── ch02-python-cli/
├── ch03-data-report/
├── ch05-ml-baseline/
├── ch08-rag-assistant/
├── ch09-agent-planner/
└── final-ai-app/
```

Every folder should include a README, source code, sample input or data, output evidence, one failure sample, and one next step.
