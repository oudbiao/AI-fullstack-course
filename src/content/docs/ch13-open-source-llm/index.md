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

## Where This Fits

By now you can build LLM, RAG, and Agent workflows. This chapter answers a different question:

> What changes when the model is no longer a cloud API, but something you download, host, quantize, serve, or fine-tune yourself?

Open-source LLM work is mostly systems work. You must control hardware, drivers, model files, runtime, API contract, logs, evaluation cases, and rollback.

## The Deployment Loop

| Step | Decision | Evidence to keep |
|---|---|---|
| Select | model family, license, size, context, language, modality | model card, license note, why this model |
| Prepare | GPU/CPU, CUDA, PyTorch, disk, network, secrets | environment report, cost estimate |
| Run | Transformers, Ollama, llama.cpp, vLLM, SGLang, or vendor runtime | exact command, model path, first response |
| Serve | OpenAI-compatible API, internal SDK, or batch script | request/response sample, error path |
| Evaluate | fixed prompts, RAG cases, safety cases, latency/cost | eval table, failure notes |
| Adapt | Prompt, RAG, quantization, LoRA, or full fine-tune | decision memo, adapter artifact, before/after |
| Release | README, container, runbook, monitoring, shutdown plan | deployment checklist, rollback notes |

## Learning Order And Task List

| Order | Do this | Stop when you have |
|---|---|---|
| 1 | Pick one model and one runtime | a written model/runtime decision |
| 2 | Verify the environment | Python, PyTorch, CUDA or CPU status saved |
| 3 | Run one local inference | prompt, output, command, model version |
| 4 | Wrap it as an API or script | one repeatable request/response |
| 5 | Run a tiny evaluation set | at least five prompts and pass/fail notes |
| 6 | Decide whether to fine-tune | a reasoned choice: no tuning, LoRA, or full training |
| 7 | Package the runbook | README, commands, cost, limits, shutdown |

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

| Code part | What it means | What you change first |
|---|---|---|
| `project = {...}` | The project constraint card. It turns "I want to run a model" into hardware, privacy, users, latency, and tuning requirements. | Change `task`, `privacy`, `available_vram_gb`, and `needs_fine_tuning`. |
| `choose_runtime(info)` | The runtime decision rule. It prevents you from renting a GPU or downloading a model before checking memory. | Adjust VRAM thresholds after you know the real instance or local machine. |
| `choose_adaptation(info)` | The fine-tuning gate. Private knowledge should usually try RAG before training. | Set `needs_fine_tuning` to `True` only after fixed eval cases keep failing. |
| `plan = {...}` | The deployment checklist that connects model choice, runtime choice, and required evidence. | Add project-specific evidence such as auth, logging, or rollback notes. |
| `write_text(...)` and `print(...)` | The script saves the same plan to disk and terminal output, so the run can be reviewed later. | Commit or archive `open_llm_runbook.json` with the rest of the experiment notes. |

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

| Step | Action | Evidence to save |
|---|---|---|
| 1. Define the run | Write the target model size, runtime, expected first prompt, and maximum rental time. | `gpu_plan.md` with stop time and budget guardrail |
| 2. Pick the instance | Choose Linux, enough VRAM for the model, enough disk for weights, and SSH access. | instance type, VRAM, disk, hourly price note |
| 3. Lock access down | Use SSH keys, keep the model API private by default, and open only the ports you need. | security note and exposed ports |
| 4. Prepare the environment | Run `python -V`, torch/CUDA check, `nvidia-smi` if available, and disk check. | `environment_report.txt` |
| 5. Run one model path | Download or mount one model, run one prompt, then save command, output, and failure notes. | `first_run.md` |
| 6. Stop and archive | Copy runbook, logs, eval cases, and README back to your project; then stop or destroy the instance. | shutdown screenshot or stop note |

The most important command is often the last one: stop the machine. A successful experiment that keeps billing silently is still an engineering failure.

## Runtime Choices

| Runtime | Use when | Avoid when |
|---|---|---|
| Transformers | learning, debugging, custom Python pipelines | you need high-throughput serving immediately |
| Ollama / LM Studio | local demo, laptop testing, non-engineer handoff | you need precise production control |
| llama.cpp | CPU or quantized edge experiments | you need standard GPU server features |
| vLLM | OpenAI-compatible high-throughput API serving | your GPU or dependency setup is not ready |
| SGLang | structured generation, serving, and agentic workloads | you need the simplest possible first run |
| Cloud model API | product prototype with low ops burden | privacy, cost, or latency requires local control |

Start with the simplest runtime that proves the product behavior. Upgrade only when latency, cost, privacy, or throughput demands it.

## Fine-Tuning Decision

Do not fine-tune just because the model gave one bad answer.

| Symptom | Try first | Fine-tune only if |
|---|---|---|
| Missing private knowledge | RAG | retrieval is correct but behavior is still wrong |
| Wrong output format | schema, parser, examples | many fixed cases still fail |
| Wrong tone or role | system prompt and examples | the same style issue repeats across many examples |
| Domain terms weak | glossary, RAG, few-shot | you have enough labeled domain examples |
| Too slow or expensive | smaller model, quantization, batching | behavior is good and runtime still fails constraints |

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
