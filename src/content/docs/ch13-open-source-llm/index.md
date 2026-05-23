---
title: "13 Open-Source LLM Deployment and Fine-Tuning"
description: "Learn how to choose, run, serve, evaluate, and lightly fine-tune open-source LLMs with reproducible environment notes, runtime decisions, and delivery evidence."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM, local LLM deployment, vLLM, SGLang, Transformers, LoRA, model serving"
---
![Open-source LLM runtime deployment loop](/img/course/ch13-open-source-llm-runtime-loop-en.webp)

Chapter 13 turns open-source model use into an engineering workflow. The goal is not to collect every model name. The goal is to choose a model, run it in a known environment, expose it through a stable interface, evaluate behavior, and keep enough evidence that another engineer can reproduce the result.

Use [Datawhale Self-LLM](https://github.com/datawhalechina/self-llm) as a broad reference library. This chapter provides the course path around it: smaller choices, fewer moving parts, and explicit pass checks.

If you want to follow commands immediately, start with [13.1 Hands-on: Run and Serve an Open-Source LLM](/ch13-open-source-llm/hands-on-open-llm-lab/). This page explains the route; 13.1 provides commands, code, outputs, and failure handling.

## Where This Fits

By now you can build LLM, RAG, and Agent workflows. This chapter answers a different question:

> What changes when the model is no longer a cloud API, but something you download, host, quantize, serve, or fine-tune yourself?

Open-source LLM work is mostly systems work. You must control hardware, drivers, model files, runtime, API contract, logs, evaluation cases, and rollback.

## The Deployment Loop

1. **Select**
   Decide model family, license, size, context length, language, and modality. Keep the model card, license note, and reason for choosing it.

2. **Prepare**
   Confirm GPU/CPU, CUDA, PyTorch, disk, network, and secrets. Keep the environment report and cost estimate.

3. **Run**
   Choose Transformers, Ollama, llama.cpp, vLLM, SGLang, or a vendor runtime. Keep the exact command, model path, and first response.

4. **Serve**
   Wrap the model as an OpenAI-compatible API, internal SDK, or batch script. Keep a request/response sample and error path.

5. **Evaluate**
   Fix prompts, RAG cases, safety cases, latency, and cost checks. Keep the eval table and failure notes.

6. **Adapt**
   Choose among Prompt, RAG, quantization, LoRA, or full fine-tuning. Keep the decision memo, adapter artifact, and before/after notes.

7. **Release**
   Package the README, container, runbook, monitoring, and shutdown plan. Keep the deployment checklist and rollback notes.

## Learning Order And Task List

1. Pick one model and one runtime; stop when you have a written model/runtime decision.
2. Verify the environment; stop when Python, PyTorch, CUDA or CPU status is saved.
3. Run one local inference; stop when you have prompt, output, command, and model version.
4. Wrap it as an API or script; stop when you have one repeatable request/response.
5. Run a tiny evaluation set; stop when you have at least five prompts and pass/fail notes.
6. Decide whether to fine-tune; stop when you can justify no tuning, LoRA, or full training.
7. Package the runbook; stop when README, commands, cost, limits, and shutdown are written.

The stage deliverables are a runnable runbook, environment report, five-case evaluation table, model/runtime decision memo, and README with shutdown or rollback notes.

## First Runnable Loop: Build a Model Runbook

This offline script does not download a model. It teaches the planning habit you need before renting a GPU or starting a long download.

Create `ch13_open_llm_runbook.py` and run it with Python 3.10 or later.

```python
import json
from pathlib import Path


project = {
    "task": "course assistant",
    "privacy": "local documents may be private",
    "expected_users": "small internal group",
    "latency_target_seconds": 4,
    "available_vram_gb": 24,
    "needs_fine_tuning": False,
}


def choose_runtime(info):
    if info["available_vram_gb"] >= 24:
        return {
            "runtime": "vLLM or SGLang",
            "model_size": "7B to 14B instruct model",
            "why": "enough VRAM for a practical server and OpenAI-compatible API",
        }
    if info["available_vram_gb"] >= 8:
        return {
            "runtime": "Transformers or Ollama",
            "model_size": "1B to 7B instruct model, possibly quantized",
            "why": "simpler setup and acceptable for a small lab",
        }
    return {
        "runtime": "CPU quantized runtime or cloud API fallback",
        "model_size": "small quantized model",
        "why": "local GPU memory is too limited for serving a larger model",
    }


def choose_adaptation(info):
    if info["needs_fine_tuning"]:
        return "prepare a LoRA experiment with a fixed eval set first"
    if info["privacy"] == "local documents may be private":
        return "try RAG before fine-tuning"
    return "start with prompt and decoding settings"


plan = {
    "project": project["task"],
    "runtime_choice": choose_runtime(project),
    "adaptation_choice": choose_adaptation(project),
    "minimum_evidence": [
        "environment report",
        "model card and license note",
        "first prompt/output",
        "five-case evaluation table",
        "latency and memory note",
        "shutdown or rollback step",
    ],
}

Path("open_llm_runbook.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
print(json.dumps(plan, indent=2))
```

Expected output:

```text
{
  "project": "course assistant",
  "runtime_choice": {
    "runtime": "vLLM or SGLang",
    "model_size": "7B to 14B instruct model",
    "why": "enough VRAM for a practical server and OpenAI-compatible API"
  },
  "adaptation_choice": "try RAG before fine-tuning",
  "minimum_evidence": [
    "environment report",
    "model card and license note",
    "first prompt/output",
    "five-case evaluation table",
    "latency and memory note",
    "shutdown or rollback step"
  ]
}
```

The output should become your deployment runbook. Before running a large model, change the VRAM number, privacy requirement, task, and fine-tuning flag.

## Read the Runbook Line by Line

1. **`project = {...}`**
   This is the project constraint card. It turns "I want to run a model" into hardware, privacy, users, latency, and tuning requirements. Change `task`, `privacy`, `available_vram_gb`, and `needs_fine_tuning` first.

2. **`choose_runtime(info)`**
   This is the runtime decision rule. It prevents you from renting a GPU or downloading a model before checking memory. Adjust the VRAM thresholds after you know the real instance or local machine.

3. **`choose_adaptation(info)`**
   This is the fine-tuning gate. Private knowledge should usually try RAG before training. Set `needs_fine_tuning` to `True` only after fixed eval cases keep failing.

4. **`plan = {...}`**
   This is the deployment checklist that connects model choice, runtime choice, and required evidence. Add project-specific evidence such as auth, logging, or rollback notes.

5. **`write_text(...)` and `print(...)`**
   These lines save the same plan to disk and terminal output, so the run can be reviewed later. Commit or archive `open_llm_runbook.json` with the rest of the experiment notes.

If you can explain each row, you understand the script. If a row feels vague, edit the project card and run it again before touching a GPU.

## Minimal Environment Check

Run this before downloading a model:

```bash
python -V
python - <<'PY'
import platform
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
except Exception as exc:
    print("torch check failed:", repr(exc))
print("platform:", platform.platform())
PY
```

Save the output. If the environment is not visible, the model result is not reproducible.

## Renting a GPU Without Losing the Thread

Treat a rented GPU as a short experiment, not a permanent computer. Free or no-rental paths come first: a local quantized model, a school or company GPU, or a notebook platform's free quota if available. If those are too slow or unavailable, rent only enough time to prove one model/runtime loop.

1. **Define the run**
   Write the target model size, runtime, expected first prompt, and maximum rental time. Save `gpu_plan.md` with stop time and budget guardrail.

2. **Pick the instance**
   Choose Linux, enough VRAM for the model, enough disk for weights, and SSH access. Save instance type, VRAM, disk, and hourly price note.

3. **Lock access down**
   Use SSH keys, keep the model API private by default, and open only the ports you need. Save a security note and exposed ports.

4. **Prepare the environment**
   Run `python -V`, torch/CUDA check, `nvidia-smi` if available, and disk check. Save `environment_report.txt`.

5. **Run one model path**
   Download or mount one model, run one prompt, then save command, output, and failure notes. Save `first_run.md`.

6. **Stop and archive**
   Copy runbook, logs, eval cases, and README back to your project, then stop or destroy the instance. Save a shutdown screenshot or stop note.

The most important command is often the last one: stop the machine. A successful experiment that keeps billing silently is still an engineering failure.

## Runtime Choices

**Transformers**

Use it for learning, debugging, and custom Python pipelines. Avoid treating it as the final server when you immediately need high-throughput serving.

**Ollama / LM Studio**

Use it for local demos, laptop testing, and non-engineer handoff. Avoid it when you need precise production control.

**llama.cpp**

Use it for CPU or quantized edge experiments. Avoid it when you need standard GPU server features.

**vLLM**

Use it for OpenAI-compatible high-throughput API serving. Avoid it when your GPU or dependency setup is not ready.

**SGLang**

Use it for structured generation, serving, and agentic workloads. Avoid it when you only need the simplest possible first run.

**Cloud model API**

Use it for product prototypes with low operational burden. Avoid it when privacy, cost, or latency requires local control.

Start with the simplest runtime that proves the product behavior. Upgrade only when latency, cost, privacy, or throughput demands it.

## Fine-Tuning Decision

Do not fine-tune just because the model gave one bad answer.

**Missing private knowledge**

Try RAG first. Fine-tune only if retrieval is correct but behavior is still wrong.

**Wrong output format**

Try schema constraints, a parser, and examples first. Fine-tune only if many fixed cases still fail.

**Wrong tone or role**

Try the system prompt and examples first. Fine-tune only if the same style issue repeats across many examples.

**Domain terms weak**

Try a glossary, RAG, and few-shot examples first. Fine-tune only if you have enough labeled domain examples.

**Too slow or expensive**

Try a smaller model, quantization, and batching first. Move toward training only if behavior is good and runtime still fails constraints.

For most course projects, LoRA is the first serious adaptation method. Full fine-tuning is a later engineering choice, not the default.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
model_choice: selected model, size, license, and reason
runtime_choice: Transformers/Ollama/llama.cpp/vLLM/SGLang/API and why
environment: Python, torch, CUDA/device, disk, and cost estimate
first_run: exact command, prompt, output, latency, memory note
adaptation_decision: Prompt/RAG/quantization/LoRA/full fine-tune decision
expected_output: runbook, evaluation table, README, and rollback or shutdown note
```

## Common Failures

- Downloading a large model before checking disk, network, and VRAM.
- Treating a successful chat response as deployment evidence.
- Ignoring the model license or data-use restriction.
- Fine-tuning without a fixed before/after evaluation set.
- Exposing a local model server without auth, logging, and shutdown rules.
- Forgetting to stop a rented GPU after the experiment.

## Pass Check

You pass this chapter when you can choose a model/runtime pair for one project, run the environment check, produce `open_llm_runbook.json`, explain whether Prompt, RAG, quantization, LoRA, or full fine-tuning is the right next step, and write a README command another engineer can follow.
