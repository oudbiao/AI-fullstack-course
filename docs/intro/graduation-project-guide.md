---
sidebar_position: 16
title: "Final Project Design Guide"
description: "A compact guide for designing a closed-loop AI full-stack capstone project that can be run, evaluated, reviewed, and improved."
keywords: [final project, AI full-stack project, portfolio, project design]
---

# Final Project Design Guide

![Final project closed-loop design diagram](/img/course/intro-graduation-project-loop-en.png)

A final project should prove that you can turn a real problem into a runnable, evaluable AI system. It does not need many features. It needs a complete loop.

## Final Project Loop

| Layer | Minimum requirement |
| --- | --- |
| Problem | Who is the user, and what pain point is solved? |
| Materials/data | Where does the input come from, and how is it cleaned? |
| AI capability | Prompt, RAG, Agent, model, or multimodal workflow |
| Interaction | CLI, notebook, API, or simple web entry |
| Evaluation/logs | Fixed tests, failure samples, citations, traces, or metrics |
| Demo/review | README, screenshots, demo script, next-step plan |

## Choose a Topic

| You have | Good topic | Evaluation focus |
| --- | --- | --- |
| Course docs or knowledge base | RAG / AI learning assistant | retrieval hit, citation support, no-answer handling |
| CSV, Excel, or business data | Data analysis Agent | correctness, chart explanation, safe code |
| Images, screenshots, PDFs | Multimodal assistant | parsing quality, uncertainty, human review |
| Domain text | NLP / vertical assistant | extraction accuracy, label boundaries, factual consistency |
| No clear idea | AI learning assistant | reproducibility, logs, evaluation, demo |

## Minimum Version

For an AI learning assistant, the minimum final project can be:

```text
Markdown docs -> chunk/index -> user question -> retrieved sources -> answer with citation -> Q&A log -> 10 eval questions
```

No complex UI, multi-Agent system, or long-term memory is required for the first version.

## README Must Include

| Section | Must answer |
| --- | --- |
| Goal | What problem and user? |
| Run | What commands and environment variables? |
| Example | What input and output? |
| Architecture | Where are LLM, RAG, Agent, data, and logs? |
| Evaluation | What fixed tests or metrics? |
| Failure cases | Where does it break? |
| Next steps | What will improve next and why? |

## Demo Script

| Time | Show |
| --- | --- |
| 1 minute | Problem and user |
| 2 minutes | Architecture and data flow |
| 3 minutes | One successful example |
| 2 minutes | One failure example and diagnosis |
| 2 minutes | Evaluation result and next step |

Being able to show one failure clearly is often stronger than hiding all failures. It proves engineering judgment.
