---
title: "13.1 Compute Routes: Local CPU, Free Colab, Rented GPU"
description: "Choose where to run an open-source LLM experiment: local CPU, free Colab when available, or a rented GPU with budget, security, and shutdown evidence."
sidebar:
  order: 2
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM compute route, local CPU LLM, free Colab GPU, rented GPU LLM, vLLM GPU"
---
![Open-source LLM compute route selector](/img/course/ch13-open-source-llm-compute-routes-en.webp)

Before choosing a model name, choose the place where the experiment will run. A good compute route tells you what can be proven today, what should wait, what evidence to keep, and how to stop before cost or complexity runs away.

This page gives you three routes:

- **Local CPU**: safest first loop, no rental, proves code and evidence.
- **Free Colab**: useful when a free GPU is available, but not guaranteed.
- **Rented GPU**: best for vLLM-style serving or 7B-class models, but only with a written stop plan.

## Route Comparison

<div class="course-route-list">
  <section class="course-route-card">
    <h3>Local CPU</h3>
    <dl>
      <div>
        <dt>Use When</dt>
        <dd>You want the safest first run on your own machine.</dd>
      </div>
      <div>
        <dt>First Target</dt>
        <dd><code>sshleifer/tiny-gpt2</code>, quantized small model, evaluation script, local API skeleton.</dd>
      </div>
      <div>
        <dt>Not For</dt>
        <dd>Proving 7B quality, high throughput, or long-context serving.</dd>
      </div>
      <div>
        <dt>Evidence</dt>
        <dd><code>environment_report.txt</code>, <code>first_run.md</code>, <code>eval_results.csv</code>.</dd>
      </div>
    </dl>
  </section>

  <section class="course-route-card">
    <h3>Free Colab</h3>
    <dl>
      <div>
        <dt>Use When</dt>
        <dd>You need a temporary notebook and a GPU may be available.</dd>
      </div>
      <div>
        <dt>First Target</dt>
        <dd>Small instruct model, tokenizer checks, short evaluation, tiny LoRA dry run.</dd>
      </div>
      <div>
        <dt>Not For</dt>
        <dd>Private data, long jobs, public services, or guaranteed-GPU planning.</dd>
      </div>
      <div>
        <dt>Evidence</dt>
        <dd>Notebook copy, runtime type, <code>nvidia-smi</code> or CPU note, saved outputs.</dd>
      </div>
    </dl>
  </section>

  <section class="course-route-card">
    <h3>Rented GPU</h3>
    <dl>
      <div>
        <dt>Use When</dt>
        <dd>You need predictable VRAM, SSH, serving, or a 7B-class test.</dd>
      </div>
      <div>
        <dt>First Target</dt>
        <dd>vLLM/SGLang server, fixed eval set, latency and memory check.</dd>
      </div>
      <div>
        <dt>Not For</dt>
        <dd>Starting without budget, exposing a public port, or training before eval.</dd>
      </div>
      <div>
        <dt>Evidence</dt>
        <dd><code>gpu_plan.md</code>, <code>environment_report.txt</code>, request/response log, shutdown proof.</dd>
      </div>
    </dl>
  </section>
</div>

Colab is a good learning route, but treat it as opportunistic. Google's [Colab FAQ](https://research.google.com/colaboratory/intl/en-GB/faq.html) says free compute resources can include GPUs and TPUs, but resources are not guaranteed or unlimited and usage limits can fluctuate. Write your plan so the lab still works on CPU if the free GPU is unavailable.

## Start With The Smallest Proof

Choose the route by the question you are trying to answer:

| Question | Route |
|---|---|
| "Can my Python environment load a model and generate text?" | Local CPU |
| "Can I run the same notebook on a temporary hosted machine?" | Free Colab |
| "Can this model serve requests with known VRAM, latency, and shutdown?" | Rented GPU |
| "Should I fine-tune?" | None yet; run fixed eval cases first |

The first useful proof is not a clever answer. It is a reproducible trace: environment -> model -> prompt -> output -> evaluation -> stop.

## Write `compute_route.md`

Before running commands, write this file:

```md
# Compute Route

goal: prove the open-source LLM deployment loop for one small project
route: local_cpu / free_colab / rented_gpu
selected_model:
runtime:
expected_runtime_limit:
privacy_level:
budget_limit:
stop_time:
fallback_route:

## Why this route

## What this route can prove

## What this route cannot prove yet

## Evidence to copy back

## Stop or rollback step
```

If `stop_time`, `fallback_route`, or `evidence to copy back` is empty, do not rent a GPU yet.

## Route A: Local CPU

Use this route first. It is enough to complete most of [13.2 Hands-on: Run and Serve an Open-Source LLM](/ch13-open-source-llm/hands-on-open-llm-lab/) with the default tiny model.

```bash
mkdir openllm_lab
cd openllm_lab

python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install "torch" "transformers>=4.41" "accelerate" "safetensors" "sentencepiece" "fastapi" "uvicorn"
```

Then run the lab with the default smoke-test model:

```bash
python environment_report.py
python run_local_llm.py
python eval_openllm.py
uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000
```

Stop with `Ctrl+C`. Your pass condition is not quality; it is whether the environment, inference, evaluation, API, and stop path all work.

Use this route when you want to change code quickly. Leave model quality claims for a better model and a fixed evaluation set.

## Route B: Free Colab

Use this route when you want a hosted notebook and a GPU may be available. Do not assume a GPU will always be assigned.

In the notebook:

```bash
!python -V
!nvidia-smi || true
!python -m pip install -U pip
!python -m pip install "torch" "transformers>=4.41" "accelerate" "safetensors" "sentencepiece"
```

Then copy the local inference and evaluation code from the hands-on page into cells. Start with:

```bash
MODEL_ID="sshleifer/tiny-gpt2" python run_local_llm.py
python eval_openllm.py
```

If GPU is available and the notebook is stable, try a small instruct model:

```bash
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" python run_local_llm.py
```

Keep these Colab-specific notes:

```text
runtime_type:
gpu_visible: yes/no
notebook_url_or_copy:
install_cells:
first_run_output:
files_downloaded_back:
what_would_break_if_runtime_resets:
```

Do not put private documents, secrets, or long-running serving workloads into a free notebook. If you need stable serving, use the rented GPU route or a controlled local/server environment.

## Route C: Rented GPU

Rent only after the local CPU or Colab path has produced a working evidence bundle. A rented machine should answer one bounded question, such as:

- Can a 7B-class instruct model serve through vLLM?
- Does the fixed eval set pass on a larger model?
- What latency and memory do we observe for this route?

Write `gpu_plan.md` first:

```md
# GPU Plan

goal:
model:
runtime:
instance_vram:
disk:
region:
hourly_budget:
hard_stop_time:
ports_to_open:
access_method: SSH key
evidence_to_copy_back:
shutdown_proof:
fallback_if_oom:
```

On the remote machine:

```bash
python -V
nvidia-smi
df -h
python -m pip install -U pip
python -m pip install "vllm"
```

Bind to localhost first:

```bash
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

From your local machine, connect through an SSH tunnel:

```bash
ssh -L 8000:127.0.0.1:8000 user@your-gpu-host
```

Then test the OpenAI-compatible endpoint:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Give one deployment rule for a rented GPU."}]
  }'
```

After the test, copy back the evidence and stop or destroy the instance. A successful model demo that keeps billing silently is still a failed engineering run.

## Route Decision Drill

Fill this before continuing:

```text
I will use _____ because _____.
This route can prove _____.
This route cannot prove _____ yet.
I will stop or fall back when _____.
The evidence I must copy back is _____.
```

<details>
<summary>How to judge your answer</summary>

A strong answer names constraints instead of enthusiasm. For example: local CPU can prove the code path but not service throughput; Colab can test a notebook path but cannot guarantee GPU availability; rented GPU can test serving but needs budget, SSH, ports, and shutdown proof. If the answer only says "because it is faster," the route decision is not complete.

</details>

## Evidence to Keep

```text
compute_route: local_cpu / free_colab / rented_gpu and why
environment: Python, torch, CUDA/MPS/CPU, disk, runtime reset risk
budget_or_limit: free quota caveat or rental stop time
security: private data policy, secrets policy, exposed ports
first_run: model, command, prompt, output, latency or memory note
stop_proof: Ctrl+C, notebook saved, or rented instance stopped
```

## Pass Check

You pass this lesson when you can choose one compute route, explain what it can and cannot prove, run the environment check, and name the exact stop or fallback step before moving to the hands-on lab.
