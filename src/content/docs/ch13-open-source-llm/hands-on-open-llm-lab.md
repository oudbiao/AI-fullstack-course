---
title: "13.1 Hands-on: Run and Serve an Open-Source LLM"
description: "Run a reproducible open-source LLM loop: environment check, first Transformers inference, five-case evaluation, and a local OpenAI-style API."
sidebar:
  order: 2
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM lab, Transformers local inference, vLLM serving, OpenAI compatible API, LLM evaluation"
---
![Open-source LLM runtime deployment loop](/img/course/ch13-open-source-llm-runtime-loop-en.webp)

This page is the runnable lab. You will start with a tiny model and prove the whole loop: environment, inference, evaluation, API serving, and shutdown. The default model is not chosen for answer quality. It is chosen so the code path can run on an ordinary machine before you spend time or money on a larger model.

After the loop works, replace `MODEL_ID` with Qwen, Llama, InternLM, ChatGLM, or another model family. Self-LLM is useful because it has model-specific routes; this page gives you the shared engineering skeleton first.

## What You Will Produce

Your folder should end with:

```text
openllm_lab/
  environment_report.py
  environment_report.txt
  run_local_llm.py
  first_run.md
  eval_cases.csv
  eval_openllm.py
  eval_results.csv
  serve_openai_like.py
  README.md
```

The pass bar is not "the model sounds smart." The pass bar is:

- the environment report is reproducible;
- the local model loads and generates text;
- five fixed cases can be evaluated again;
- the API can be called with `curl`;
- you know how to stop the service and archive evidence.

## 0. Choose One Model Route

**Smoke test: `sshleifer/tiny-gpt2`**

Good for any ordinary computer. It proves the code path, not assistant quality.

**Small real model: `Qwen/Qwen2.5-0.5B-Instruct`**

Good for a machine with stable network and disk. It is closer to a real chat model, but it takes longer to download.

**GPU serving: 7B-class instruct model**

Good for a rented GPU or enough local VRAM. Run the small loop first, then upgrade to vLLM.

Start with the smoke test. Do not begin by downloading a large model.

## 1. Create the Project Environment

```bash
mkdir openllm_lab
cd openllm_lab

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install "torch" "transformers>=4.41" "accelerate" "safetensors" "sentencepiece" "fastapi" "uvicorn"
```

If `torch` fails to install, use the PyTorch website to choose the command for your operating system and accelerator. Do not skip this step; every later model load depends on it.

## 2. Write the Environment Check

Create `environment_report.py`:

```python
import platform
import shutil
import subprocess
from pathlib import Path


def run_optional(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"not available: {exc!r}"


lines = [
    f"python_platform: {platform.platform()}",
    f"python_version: {platform.python_version()}",
]

try:
    import torch

    lines.extend(
        [
            f"torch_version: {torch.__version__}",
            f"cuda_available: {torch.cuda.is_available()}",
            f"cuda_device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
            f"mps_available: {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}",
        ]
    )
except Exception as exc:
    lines.append(f"torch_check_failed: {exc!r}")

disk = shutil.disk_usage(".")
lines.append(f"disk_free_gb: {round(disk.free / 1024**3, 2)}")
lines.append("nvidia_smi:")
lines.append(run_optional(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]))

report = "\n".join(lines) + "\n"
Path("environment_report.txt").write_text(report, encoding="utf-8")
print(report)
```

Run it:

```bash
python environment_report.py
```

Read it this way:

- `cuda_available: True` means NVIDIA GPU is usable;
- `mps_available: True` means Apple Silicon may use MPS;
- neither one is fine for the default tiny model;
- `environment_report.txt` is required evidence.

## 3. Write Local Inference

Create `run_local_llm.py`:

```python
import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = pick_device()
    kwargs = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return tokenizer, model, device


def build_inputs(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        return {"input_ids": input_ids}, input_ids.shape[-1]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs, inputs["input_ids"].shape[-1]


def generate_once(tokenizer, model, device, prompt, max_new_tokens=80):
    inputs, input_length = build_inputs(tokenizer, prompt, device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    elapsed = time.time() - started
    new_tokens = output_ids[0][input_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text or "(empty output)", elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2"))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", "Explain what a local LLM runtime does in one sentence."))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    answer, elapsed = generate_once(tokenizer, model, device, args.prompt, args.max_new_tokens)

    report = f"""# First local LLM run

model: {args.model}
device: {device}
prompt: {args.prompt}
latency_seconds: {elapsed:.2f}

## Output

{answer}
"""
    Path("first_run.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

Run the default model:

```bash
python run_local_llm.py
```

The output may be odd. That is fine because `tiny-gpt2` is only a smoke test. Passing means the script downloaded a model, loaded weights, generated text, and wrote `first_run.md`.

Then try a small real model:

```bash
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
PROMPT="Explain in three sentences why a local LLM deployment should keep an environment report." \
python run_local_llm.py
```

If the download is slow, do not switch models yet. Run the evaluation and API service with the tiny model first.

## 4. Fix Five Evaluation Cases

Create `eval_cases.csv`:

```csv
id,prompt,expected_behavior
case_001,Explain why model license matters before deployment.,mentions license or usage limits
case_002,Give one reason to run a fixed eval set before LoRA.,mentions before after comparison
case_003,What should be saved after the first local model run?,mentions command prompt output or environment
case_004,Why should a rented GPU be stopped after the experiment?,mentions cost or billing
case_005,When should RAG be tried before fine-tuning?,mentions private knowledge or retrieval
```

Create `eval_openllm.py`:

```python
import csv
import json
import os
from pathlib import Path

from run_local_llm import generate_once, load_model


model_id = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")
tokenizer, model, device = load_model(model_id)

rows = []
with open("eval_cases.csv", newline="", encoding="utf-8") as file:
    for case in csv.DictReader(file):
        output, elapsed = generate_once(tokenizer, model, device, case["prompt"], max_new_tokens=80)
        passed = bool(output.strip()) and output != "(empty output)"
        rows.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "expected_behavior": case["expected_behavior"],
                "passed": passed,
                "latency_seconds": round(elapsed, 2),
                "output": output.replace("\n", " "),
            }
        )

with open("eval_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "model": model_id,
    "device": device,
    "total": len(rows),
    "passed_nonempty": sum(row["passed"] for row in rows),
}
Path("eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
```

Run it:

```bash
python eval_openllm.py
```

This first check only verifies non-empty output. In a real project, open `eval_results.csv`, review answers manually, and replace `passed` with real pass/fail notes. Fixed cases matter more than one good chat because they let you compare model, runtime, quantization, or LoRA changes.

## 5. Serve a Local OpenAI-Style API

Create `serve_openai_like.py`:

```python
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from run_local_llm import generate_once, load_model


MODEL_ID = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")
tokenizer, model, device = load_model(MODEL_ID)
app = FastAPI(title="Open LLM local lab")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 120


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    prompt = "\n".join(
        f"{message.role}: {message.content}"
        for message in request.messages
        if message.role != "system"
    )
    answer, elapsed = generate_once(tokenizer, model, device, prompt, request.max_tokens)
    return {
        "id": f"local-{int(time.time())}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"latency_seconds": round(elapsed, 2)},
    }
```

Start it:

```bash
uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000
```

Open another terminal:

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Give one deployment checklist item for a local LLM."}
    ],
    "max_tokens": 80
  }'
```

Stop the server:

```bash
Ctrl+C
```

Save the health response, request JSON, and response JSON. If a service cannot be stopped cleanly, it is not production-ready.

## 6. Upgrade to vLLM When You Have a GPU

The small FastAPI service is a teaching skeleton, not a high-throughput server. With an NVIDIA GPU, try vLLM:

```bash
python -m pip install "vllm"
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 0.0.0.0 --port 8000
```

Test it:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain vLLM in one sentence."}]
  }'
```

On a rented GPU, keep the security rule simple:

- do not expose the port to the public internet by default;
- use an SSH tunnel or platform-private network first;
- record the start and stop commands;
- stop the instance immediately after copying evidence back.

## 7. Decide Whether LoRA Is Needed

Do not fine-tune because one answer was bad. Bucket failures in `eval_results.csv` first:

**Missing private knowledge**

Try RAG, better documents, and better retrieval first. Consider LoRA only when retrieval is correct but behavior remains wrong.

**Unstable format**

Try schema constraints, few-shot examples, and a parser first. Consider LoRA only when the same format failure repeats in fixed cases.

**Unstable style**

Try the system prompt and examples first. Consider LoRA only when the style issue repeats across many examples.

**Slow inference**

Try a smaller model, quantization, and vLLM first. Move toward training only when behavior is good but runtime constraints still fail.

If LoRA is justified, prepare these before training:

```text
train.jsonl        # high-quality training samples
eval_cases.csv     # fixed eval cases, separate from training data
base_model_note.md # base model, license, version, and reason
```

Self-LLM's LoRA chapters fit after this point: you already have an environment report, base-model choice, fixed evaluation set, and first-run evidence.

## 8. Debug by Symptom

If `environment_report.py` fails, first check the Python version, whether the virtual environment is active, and whether `torch` was installed into that environment. Do not switch models yet.

If `run_local_llm.py` downloads slowly, keep using `sshleifer/tiny-gpt2` until the full loop works. A large-model download is a model-choice issue; it should not block environment, evaluation, and API practice.

If the output is empty, first simplify the prompt, check whether `pad_token_id` produced an error, and retry with fewer `max_new_tokens`.

If the API does not start, check whether port 8000 is already occupied, then confirm that `uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000` is being run inside `openllm_lab`.

If vLLM reports out-of-memory, try a smaller model or lower concurrency first. Do not jump straight to LoRA; memory pressure is usually not solved by training.

The final acceptance check is the evidence bundle: `environment_report.txt`, `first_run.md`, `eval_results.csv`, `eval_summary.json`, and the API health/request/response records.

## 9. Write the README

Create `README.md`:

````md
# Open LLM Lab

## Model

- Smoke test: sshleifer/tiny-gpt2
- Next model to try: Qwen/Qwen2.5-0.5B-Instruct

## Run

```bash
source .venv/bin/activate
python environment_report.py
python run_local_llm.py
python eval_openllm.py
uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000
```

## Evidence

- environment_report.txt
- first_run.md
- eval_cases.csv
- eval_results.csv
- eval_summary.json

## Stop

Use Ctrl+C to stop the local API. Stop rented GPU instances immediately after copying evidence back.
````

After this page, you have not merely read that open-source models can be deployed. You have run one reproducible path: environment -> model -> output -> evaluation -> API -> shutdown.
