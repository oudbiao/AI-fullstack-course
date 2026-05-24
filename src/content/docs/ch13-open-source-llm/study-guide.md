---
title: "13.0 Learning Checklist: Open-Source LLM Deployment"
description: "A compact checklist for Chapter 13: model selection, runtime choice, environment checks, serving evidence, evaluation, and fine-tuning decisions."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM checklist, local model deployment, LoRA checklist, vLLM checklist"
---
Use this page as a printable checklist. If you need the full explanation, return to the [Chapter 13 entry page](/ch13-open-source-llm/).

![Chapter 13 open-source LLM study checklist](/img/course/ch13-open-source-llm-study-checklist-en.webp)

If you have not run the lab yet, first complete [13.1 Compute Routes: Local CPU, Free Colab, Rented GPU](/ch13-open-source-llm/compute-routes/), then [13.2 Hands-on: Run and Serve an Open-Source LLM](/ch13-open-source-llm/hands-on-open-llm-lab/). Use [13.3 Model and Runtime Decision](/ch13-open-source-llm/model-runtime-decision/) and [13.4 Serving, Evaluation, and Release Runbook](/ch13-open-source-llm/serving-evaluation-runbook/) to finish the deployment evidence.

## Two-Hour First Pass

1. **20 min: Choose the compute route**
   Stop when you can say, "This run belongs on local CPU, free Colab, or rented GPU, and I know what that route cannot prove."

2. **20 min: Run the environment check**
   Stop when you can say, "I know whether this machine has usable CUDA or only CPU."

3. **25 min: Run the runbook script**
   Stop when you can say, "I can choose a runtime from hardware and project constraints."

4. **25 min: Build a five-prompt eval table**
   Stop when you can say, "I can compare model behavior before changing runtime or tuning."

5. **30 min: Write the adaptation decision**
   Stop when you can say, "I can explain why I chose Prompt, RAG, quantization, LoRA, or no tuning."

6. **30 min: Write the release runbook**
   Stop when you can say, "Another engineer can start, test, stop, and roll back this service."

## Required Evidence

- `environment_report.txt`: Python, torch, CUDA/device, platform, disk or instance note.
- `compute_route.md`: local CPU, free Colab, or rented GPU choice with fallback and stop rule.
- `model_decision.md`: model, size, license, source, reason, rejected alternatives.
- `open_llm_runbook.json`: runtime choice, adaptation choice, required evidence.
- `first_run.md`: exact command, prompt, output, latency or memory note.
- `eval_cases.csv`: at least five prompts, expected behavior, pass/fail, notes.
- `README.md`: setup, run, evaluate, stop server, rollback or shutdown.

## Quality Gates

- **Reproducibility**: another engineer can identify model version, runtime, command, and environment.
- **Safety**: license, privacy, auth, logging, and shutdown are checked before sharing.
- **Evaluation**: runtime or tuning changes are compared on the same eval cases.
- **Cost control**: free notebook limits or GPU rental time, memory, latency, and stop procedure are recorded.
- **Adaptation**: fine-tuning is justified by repeated evidence, not one disappointing answer.

## Exit Questions

- Can you explain why you chose this model size and license?
- Can you explain why this run belongs on local CPU, free Colab, or rented GPU?
- Can you say why this runtime is enough for the current project?
- Can you run or reproduce the environment check?
- Can you compare outputs with the same five prompts after a change?
- Can you defend the adaptation choice: Prompt, RAG, quantization, LoRA, or full fine-tune?

If the answer is yes, you can treat open-source LLMs as an engineering option instead of a collection of random model demos.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
environment_report: Python, torch, CUDA/device, platform, and hardware/cost note
compute_route: local CPU / free Colab / rented GPU, fallback, stop rule
model_decision: selected model, license, size, source, and rejected alternatives
runtime_contract: command or endpoint, request format, response format, and error path
evaluation: fixed prompts, outputs, pass/fail notes, latency or memory note
adaptation_choice: Prompt/RAG/quantization/LoRA/full fine-tune decision with reason
```
