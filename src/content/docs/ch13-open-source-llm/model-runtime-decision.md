---
title: "13.3 Model and Runtime Decision"
description: "Turn model choice into an engineering decision: license, size, context length, hardware, quantization, runtime, and fallback path."
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM model selection, runtime decision, Ollama, llama.cpp, vLLM, SGLang, quantization"
---
![Open-source LLM runtime decision map](/img/course/ch13-open-source-llm-runtime-decision-en.webp)

A good open-source LLM project starts before the first download. This lesson turns model selection into a written decision so you do not waste time on a model that your hardware, license, latency target, or product boundary cannot support.

## What This Page Solves

The beginner mistake is to ask, "Which model is best?" The engineering question is more specific: "Which model/runtime pair is enough for this project, on this hardware, with this license and rollback path?"

Use the smallest model and simplest runtime that proves the project behavior. Move upward only when evidence shows that quality, context length, throughput, privacy, or cost requires it.

## The Decision Ladder

1. **Task fit**
   Decide whether the project needs chat, extraction, code, multilingual support, long context, tool calling, or multimodal behavior.

2. **License fit**
   Read the model card and license before building around the model. Keep a note for commercial use, redistribution, and data-use restrictions.

3. **Hardware fit**
   Estimate VRAM/RAM and disk before download. If the model cannot run locally, choose a smaller model, quantization, a rented GPU, or a cloud API fallback.

4. **Runtime fit**
   Use Transformers for learning, Ollama/LM Studio for local handoff, llama.cpp for quantized CPU/edge tests, and vLLM/SGLang for server-style inference.

5. **Evidence fit**
   Do not call the decision complete until you have model version, command, first output, evaluation table, and stop procedure.

## Model Decision Table

Create `model_runtime_decision.md`:

```md
# Model Runtime Decision

project_goal: support-operations SOP assistant
must_have: private document handling, Chinese/English answers, stable JSON output
nice_to_have: low latency, long context, OpenAI-compatible endpoint

candidate_1: Qwen2.5-0.5B-Instruct
license_note: check model card before deployment
runtime: vLLM when GPU is available, Transformers for first local test
hardware_note: small enough for first experiment; still validate memory
risk: quality may be too weak for complex SOP reasoning

candidate_2: 7B instruct model
license_note: check commercial and redistribution terms
runtime: vLLM or SGLang on rented GPU
hardware_note: requires planned GPU budget and shutdown proof
risk: higher cost and slower iteration

fallback: cloud model API or RAG with current API model
why_now: prove the deployment loop before chasing larger models
rejected_for_now: full fine-tuning, because eval failures are not proven yet
```

The exact model names can change. The decision shape should not change.

## Runtime Selection Rules

**Start with Transformers when you need to inspect tokens, prompts, and Python behavior.** It is easy to debug and close to the model API, but it is not usually the final high-throughput server.

**Use Ollama or LM Studio when the goal is a laptop demo or non-engineer handoff.** They lower setup friction, but you give up some production control.

**Use llama.cpp when CPU, quantized, or edge constraints matter.** It is strong for small local experiments, but you still need a clear API and evaluation story.

**Use vLLM when the project needs OpenAI-compatible serving and throughput.** Do not start here until GPU, driver, memory, and security posture are clear.

**Use SGLang when structured generation or agentic serving patterns matter.** It is powerful, but it should still be justified by project requirements rather than novelty.

## Route-Specific Runtime Card

Use this card to keep the three compute routes honest. The route changes what the first proof can claim.

| Route | First model | Runtime | What to prove first | Upgrade signal |
|---|---|---|---|---|
| Local CPU | `sshleifer/tiny-gpt2` or a small quantized model | Transformers, llama.cpp, Ollama | environment, download, one prompt, eval script, local API skeleton | the loop is reproducible but quality is too weak |
| Free Colab | tiny model first, then a small instruct model if GPU appears | Transformers notebook | notebook can rerun, files can be copied back, GPU is optional | GPU is visible and eval cases justify a larger model |
| Rented GPU | small instruct model before 7B-class | vLLM or SGLang behind localhost / SSH tunnel | known VRAM, endpoint, eval table, latency note, shutdown proof | fixed eval cases pass and service behavior matters |

Do not compare routes as if they prove the same thing. Local CPU proves the workflow. Colab proves a portable notebook path. Rented GPU proves controlled serving and cost discipline.

## Write a Runnable Decision Helper

Create `choose_openllm_runtime.py`. It does not download a model. It forces the decision to depend on task, privacy, route, and available memory.

```python
import json
import os
from pathlib import Path


profile = {
    "task": os.environ.get("TASK", "course assistant"),
    "route": os.environ.get("ROUTE", "local_cpu"),
    "privacy": os.environ.get("PRIVACY", "private_docs"),
    "available_vram_gb": float(os.environ.get("VRAM_GB", "0")),
    "needs_service": os.environ.get("NEEDS_SERVICE", "no") == "yes",
}


def choose(profile):
    route = profile["route"]
    vram = profile["available_vram_gb"]

    if route == "local_cpu":
        return {
            "model": "sshleifer/tiny-gpt2 or a small quantized model",
            "runtime": "Transformers for code inspection; llama.cpp/Ollama for quantized local tests",
            "claim": "proves the workflow, not model quality",
        }

    if route == "free_colab":
        return {
            "model": "tiny model first; small instruct model only if GPU is visible",
            "runtime": "Transformers notebook",
            "claim": "proves a portable notebook run, not stable serving",
        }

    if route == "rented_gpu" and vram >= 16:
        runtime = "vLLM or SGLang" if profile["needs_service"] else "Transformers first, then vLLM"
        return {
            "model": "small instruct model before trying 7B-class",
            "runtime": runtime,
            "claim": "proves controlled serving, eval, latency, and shutdown",
        }

    return {
        "model": "smaller model or cloud API fallback",
        "runtime": "do not start GPU serving yet",
        "claim": "current hardware route is not ready",
    }


decision = {"profile": profile, "decision": choose(profile)}
Path("model_runtime_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
print(json.dumps(decision, indent=2))
```

Run one route at a time:

```bash
ROUTE=local_cpu python choose_openllm_runtime.py
ROUTE=free_colab python choose_openllm_runtime.py
ROUTE=rented_gpu VRAM_GB=24 NEEDS_SERVICE=yes python choose_openllm_runtime.py
```

The output is not a final architecture. It is a guardrail against choosing a model name before the route, memory, service need, and evidence claim are clear.

## Mini Exercise

Take a project from Chapter 8 or 9 and write one decision paragraph:

```text
For this project, I will start with _____ using _____ because _____.
I will not use a larger model yet because _____.
I will switch only if the fixed eval set shows _____.
```

<details>
<summary>Decision reasoning and explanation</summary>

A strong answer ties model size and runtime to evidence. For example: start with a small instruct model through Transformers or Ollama to prove prompts, RAG context, and output schema. Move to vLLM only after the same eval cases show acceptable quality and the project needs a service endpoint. Do not choose LoRA or a larger GPU before you know which failures remain after prompt, RAG, schema, and quantization choices.

</details>

## Evidence to Keep

```text
model_decision: selected model, license note, size, context length, and rejected alternatives
runtime_decision: chosen runtime, hardware reason, and fallback runtime
hardware_note: local CPU/GPU or rented GPU estimate, disk, and expected stop time
eval_gate: fixed cases that would justify changing model or runtime
expected_output: model_runtime_decision.md, model_runtime_decision.json, and one first-run command
```

## Pass Check

You pass this lesson when you can explain why your model/runtime pair is enough for the current project, what would make you upgrade, and what evidence prevents a random model demo from becoming an uncontrolled deployment.
