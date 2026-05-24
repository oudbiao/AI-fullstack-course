---
title: "13.2 Model and Runtime Decision"
description: "Turn model choice into an engineering decision: license, size, context length, hardware, quantization, runtime, and fallback path."
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM model selection, runtime decision, Ollama, llama.cpp, vLLM, SGLang, quantization"
---
![Open-source LLM runtime loop](/img/course/ch13-open-source-llm-runtime-loop-en.webp)

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
expected_output: model_runtime_decision.md and one first-run command
```

## Pass Check

You pass this lesson when you can explain why your model/runtime pair is enough for the current project, what would make you upgrade, and what evidence prevents a random model demo from becoming an uncontrolled deployment.
