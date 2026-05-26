---
title: "0.5 Capstone Project Thread: Course Knowledge Assistant"
description: "Use one course knowledge assistant to connect Chapters 1-13 into a portfolio-ready AI project."
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AI portfolio project, AI full-stack project, RAG project, Agent project, open-source LLM deployment"
---
![Project lens map](/img/course/appendix-ai-project-lens-map-en.webp)

If you do not have your own project yet, default to a **course knowledge assistant**. It is not extra homework. It is one portfolio thread that grows through the course: each chapter adds one layer of capability until you have an AI application that can be explained, rerun, evaluated, and deployed.

## Final Shape

By the end, this project should be able to:

- read course notes, PDFs, web excerpts, or your own learning records;
- clean data while preserving source, time, fields, and quality notes;
- answer with Prompt, RAG, or Agent workflows while keeping retrieval and tool traces;
- keep fixed eval questions, failure samples, cost/latency notes, and safety boundaries;
- optionally connect images, OCR, multimodal assets, or a local open-source model runtime;
- let a reviewer rerun the core path from the README.

## Directory Template

```tree
capstone-course-assistant/
  README.md
  data/
    raw/
    processed/
  notebooks/
  src/
    cli.py
    data_pipeline.py
    evals.py
    rag.py
    agent_tools.py
  reports/
    evidence_log.md
    failure_cases.md
    eval_results.csv
    runtime_notes.md
```

On day one, only create the folder and README. The other files should appear naturally as the chapters add capability.

## Portfolio Submission Template

Use the same final package format after every major stage. This keeps the project reviewable instead of becoming a pile of demos.

```text
README.md                  what it does, how to run, what is not supported
run.sh or commands.md       exact rerun path
data_note.md                source, fields, cleaning rules, privacy notes
eval_cases.csv              fixed questions or inputs used for comparison
failure_cases.md            at least one honest failure and suspected cause
screenshots/ or outputs/    visible result, chart, trace, or API response
release_note.md             what changed this chapter and what to test next
```

Minimum version: README, one run command, one output, and one failure note. Strong portfolio version: fixed eval set, before/after comparison, cost or latency note, safety boundary, and a short demo script.

## Growth By Chapter

**Chapters 1-3: reproducible workbench**
Keep environment commands, Git commits, a Python CLI, sample data, cleaning rules, charts, and data quality notes.

**Chapters 4-6: model evidence**
Use a small classification, regression, or representation experiment to practice baselines, metrics, failure samples, and training diagnosis. The goal is not a high score; the goal is evidence-based model judgment.

**Chapter 7: LLM behavior control**
Fix 5-10 questions, then compare prompts, structured outputs, token/context limits, and failure samples. Optionally run mini GPT-2 to understand training and generation.

**Chapter 8: RAG grounded answers**
Chunk course material, add metadata, retrieve evidence, and generate cited answers. Save top-k chunks before reading the final answer.

**Chapter 9: Agent tool loop**
Expose only a few safe tools, such as reading files, listing folders, or generating reports. Keep tool schemas, traces, safety blocks, and rollback notes.

**Chapters 10-12: product-specific extensions**
Use Chapter 10 for images or OCR, Chapter 11 for labels, extraction, or summaries, and Chapter 12 for PDF, image, audio, video, or creative-package workflows.

**Chapter 13: open-source model runtime**
Start with a small model to run local inference, evaluation, and an OpenAI-style API. With GPU access, try vLLM or SGLang. Keep the model license, environment report, first run, eval table, and stop procedure.

## Change One Thing Per Chapter

At the end of each chapter, answer four questions:

- What new capability did the project gain?
- What command reruns it?
- What evidence proves it works?
- What failure sample keeps the claim honest?

If you cannot answer, add evidence before adding features.

## Evidence to Keep

Keep this page's proof of learning as a project-thread evidence card:

```text
project_name: course knowledge assistant or your own replacement project
chapter_growth_rule: add one capability per chapter, not a pile of demos
rerun_path: README command, script, notebook cell, or service endpoint
review_bundle: data note, eval cases, trace, failure note, and release note
expected_output: one project thread that grows from setup to RAG, Agent, and runtime evidence
```

## Minimum Pass Standard

After the main route, this project should include:

- a runnable README;
- a small dataset or document set;
- fixed evaluation questions;
- a Prompt/RAG/Agent trace;
- failure cases and an improvement plan;
- a note explaining when to use a cloud API and when to use an open-source model runtime.

The goal is not the largest system. The goal is a system that makes another person believe you understand the AI engineering loop.
