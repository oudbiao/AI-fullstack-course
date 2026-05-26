---
title: "13.2 实操：跑通、训练并服务化一个开源大模型"
description: "从环境检查、Transformers 推理、五条样本评估、本地 OpenAI 风格 API 服务，到 mini GPT-2 GPU 训练证据包，手把手跑通一个可复现的开源大模型闭环。"
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "开源大模型实操, Transformers 本地推理, vLLM 服务, OpenAI compatible API, LLM 评估"
---
![开源大模型运行时部署闭环](/img/course/ch13-open-source-llm-runtime-loop.webp)

这一页补上真正的操作闭环。你会从一个很小的模型开始跑通环境、推理、评估和 API 服务，还会跑一次 mini GPT-2 训练演练，知道真实 GPU 训练证据长什么样。默认模型不是为了效果好，而是为了先证明你的机器、Python 环境、模型加载、生成接口和证据文件都能工作。

开始前，先在 [13.1 计算路线：本地 CPU、免费 Colab、租 GPU](/zh-cn/ch13-open-source-llm/compute-routes/) 里选好路线。本页默认第一轮可以在本地 CPU 跑通；Colab 和租 GPU 是证据链清楚后的升级路线。

跑通后，再把 `MODEL_ID` 换成 Qwen、Llama、InternLM、ChatGLM 等模型。Self-LLM 的价值在于给很多模型提供专项说明；本页先给你一条通用工程骨架。

## 你会得到什么

完成后，目录里应该有这些文件：

```text
openllm_lab/
  environment_report.py
  environment_report.txt
  requirements-freeze.txt
  model_decision.md
  run_local_llm.py
  first_run.md
  eval_cases.csv
  eval_openllm.py
  eval_results.csv
  eval_summary.json
  serve_openai_like.py
  gpu_plan.md
  mini_gpt2_train.py
  openllm_gpu_training_run/
    environment_report.json
    training_log.csv
    mini_gpt2_checkpoint.pt
    sample.txt
    README.md
  lora_decision.md
  README.md
```

最小通过标准不是“模型回答很聪明”，而是：

- 环境检查可复现；
- 本地模型能加载并生成输出；
- 五条固定样本能重复评估；
- API 服务能被 `curl` 调用；
- mini GPT-2 训练脚本能产生日志、checkpoint 和 sample；
- 实验结束知道如何停止服务和归档证据。

CPU 或 MPS 训练只算冒烟测试。最终训练证据应该包含一次 `environment_report.json` 里 `"device": "cuda"` 的运行，并且 `training_log.csv` 至少有三行 loss 日志。

## 0. 确认计算路线和模型路线

把上一节的 `compute_route.md` 放在实验目录里。第一轮可以很简单：

```md
# Compute Route

route: local_cpu
selected_model: sshleifer/tiny-gpt2
runtime: Transformers
what_this_proves: environment, model loading, generation, evaluation, local API
what_this_does_not_prove: answer quality, vLLM throughput, 7B-class serving
fallback_route: stay on tiny model until the loop is reproducible
stop_or_rollback: Ctrl+C for local API
```

然后再选择模型路线：

**烟雾测试：`sshleifer/tiny-gpt2`**

适合任何普通电脑。它只用来验证代码链路，不代表真实助手效果。

**小型真实模型：`Qwen/Qwen2.5-0.5B-Instruct`**

适合网络和磁盘较稳定的机器。它更接近真实对话模型，但下载时间更久。

**GPU 服务：7B 级 instruct 模型**

适合租 GPU 或本机有足够显存的场景。先跑小模型，再升级到 vLLM。

先用默认烟雾测试跑通。不要一上来就下载大模型。

先写 `model_decision.md`：

```md
# Model Decision

## Task

课程知识助手，先验证本地模型运行链路。

## Selected model

- Smoke test: sshleifer/tiny-gpt2
- Next model: Qwen/Qwen2.5-0.5B-Instruct

## License and source

- Source: Hugging Face model page
- License check: read model card before real deployment

## Runtime

- First run: Transformers
- GPU server candidate: vLLM

## Rejected for now

- 7B model: wait until the tiny and 0.5B loops have evidence
- Fine-tuning: wait until fixed eval cases show repeated failures
```

Self-LLM 可以在你换具体模型时作为参考，但这张决策卡要留在自己的项目里。

## 1. 创建项目和环境

```bash
mkdir openllm_lab
cd openllm_lab

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install "torch" "transformers>=4.41" "accelerate" "safetensors" "sentencepiece" "fastapi" "uvicorn"
python -m pip freeze > requirements-freeze.txt
```

如果 `torch` 安装失败，先去 PyTorch 官网选择适合你系统的安装命令。不要跳过这一步，因为后面所有模型加载都依赖它。

`requirements-freeze.txt` 不是为了让你背依赖版本，而是为了之后能解释“这次运行到底在什么包环境里发生”。

## 2. 写环境检查脚本

新建 `environment_report.py`：

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

运行：

```bash
python environment_report.py
```

看什么：

- `cuda_available: True` 表示能走 NVIDIA GPU；
- `mps_available: True` 表示 Apple Silicon 可以尝试 MPS；
- 两者都不是也没关系，默认 tiny 模型可以用 CPU 先验证链路；
- `environment_report.txt` 是必须保存的证据。

## 3. 写本地推理脚本

新建 `run_local_llm.py`：

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

先跑默认模型：

```bash
python run_local_llm.py
```

你可能会看到不聪明甚至有点奇怪的英文输出。这是正常的，因为 `tiny-gpt2` 只是用来验证链路。通过标准是：脚本能下载模型、加载权重、生成文本，并写出 `first_run.md`。

再切到更真实的小模型：

```bash
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
PROMPT="用三句话解释本地部署大模型时为什么要保留环境报告。" \
python run_local_llm.py
```

如果下载慢，先不要换模型。先把后面的评估和 API 服务用 tiny 模型跑通。

## 4. 固定五条评估样本

新建 `eval_cases.csv`：

```csv
id,prompt,expected_behavior,must_include_any
case_001,Explain why model license matters before deployment.,mentions license or usage limits,license|usage|restriction|permission
case_002,Give one reason to run a fixed eval set before LoRA.,mentions before after comparison,before|after|compare|evaluation
case_003,What should be saved after the first local model run?,mentions command prompt output or environment,command|prompt|output|environment
case_004,Why should a rented GPU be stopped after the experiment?,mentions cost or billing,cost|billing|money|charge
case_005,When should RAG be tried before fine-tuning?,mentions private knowledge or retrieval,private|retrieval|document|knowledge
```

新建 `eval_openllm.py`：

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
        output_lower = output.lower()
        keywords = [item.strip().lower() for item in case["must_include_any"].split("|") if item.strip()]
        matched_keywords = [keyword for keyword in keywords if keyword in output_lower]
        passed = bool(matched_keywords)
        rows.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "expected_behavior": case["expected_behavior"],
                "must_include_any": case["must_include_any"],
                "passed": passed,
                "matched_keywords": "|".join(matched_keywords),
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
    "passed_keyword_check": sum(row["passed"] for row in rows),
}
Path("eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
```

运行：

```bash
python eval_openllm.py
```

这里做的是很粗的关键词检查。`tiny-gpt2` 很可能不通过，这正好说明“能运行”和“能完成任务”不是一回事。真实项目里，你要人工看 `eval_results.csv`，把 `passed` 改成真正的通过/失败，并写明失败类型。

读评估表时只看三件事：

1. **是否可重复**
   同一组 prompt 能不能在换模型、换运行时、改参数后重复执行。

2. **失败是否可分桶**
   是缺知识、格式错、语言错、拒答错，还是延迟太高。

3. **下一步是否只改一个因素**
   先固定评估集，再换模型、Prompt、RAG、量化或 LoRA。不要一次改很多东西。

固定样本比一次聊天更重要，因为它让你能比较换模型、换运行时、量化或 LoRA 之后的变化。

## 5. 包成一个 OpenAI 风格本地 API

新建 `serve_openai_like.py`：

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

启动服务：

```bash
uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000
```

另开一个终端测试：

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

停止服务：

```bash
Ctrl+C
```

把健康检查输出、请求 JSON 和响应 JSON 保存进 `first_run.md` 或 README。API 服务如果不能停止，就不算部署完成。

## 6. 有 GPU 后切到 vLLM

上面的小服务是教学骨架，不是高吞吐生产服务。有 NVIDIA GPU 后，再试 vLLM：

先写 `gpu_plan.md`：

```md
# GPU 计划

- 目标：通过 OpenAI-compatible endpoint 服务一个小型 instruct model
- 最高预算：写下本次实验的上限
- 停止时间：写下明确计划停止实例的时间
- 实例规格：GPU 类型、VRAM、磁盘、区域
- 访问方式：SSH key，默认不公开模型 API
- 要复制回本地的证据：environment_report.txt、first_run.md、eval_results.csv、README.md
- 关机证明：截图或云厂商停止记录
```

在远程机器上优先绑定本机地址，然后用 SSH tunnel 测试：

```bash
python -m pip install "vllm"
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

本地电脑另开终端建立隧道：

```bash
ssh -L 8000:127.0.0.1:8000 user@your-gpu-host
```

再测试：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Explain vLLM in one sentence."}]
  }'
```

如果你在租 GPU 上运行，先确认安全规则：

- 不要把端口直接公开给全网；
- 用 SSH tunnel 或平台内网测试；
- 记录启动命令和停止命令；
- 实验结束立即停止实例。

## 7. 跑一次 mini GPT-2 GPU 训练演练

服务现成模型和训练模型是两种能力。在 LoRA 或全参微调前，先跑一次很小的 GPT 风格训练循环，并保存证据。这个脚本不会训练出有用助手；它证明的是真实训练路径：数据 -> tokenizer -> embedding -> causal attention -> loss -> optimizer -> checkpoint -> generation。

下载课程脚本：

```bash
curl -O https://airoads.org/examples/ch13-open-llm-lab/mini_gpt2_train.py
```

本地 CPU 或 Apple Silicon 冒烟测试：

```bash
python mini_gpt2_train.py --steps 20 --batch-size 8
```

在 Colab、Kaggle、RunPod、AutoDL 或其他租用 GPU 上完成课程验收运行：

```bash
python mini_gpt2_train.py \
  --output-dir openllm_gpu_training_run \
  --device cuda \
  --steps 500 \
  --batch-size 64 | tee gpu_train_log.txt
```

如果用 Notebook，按这个 cell 顺序执行：

| Cell | 执行内容 |
|---|---|
| 1 | `!nvidia-smi` |
| 2 | `!python -V && python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` |
| 3 | 上传或 `curl` `mini_gpt2_train.py` |
| 4 | `!python mini_gpt2_train.py --device cuda --steps 500 --batch-size 64 \| tee gpu_train_log.txt` |
| 5 | 列出 `openllm_gpu_training_run` 并下载证据文件 |
| 6 | 停止 GPU runtime 或租用实例 |

预期日志形状：

```text
device: cuda
cuda_name: Tesla T4
parameters: 108836
step 0001 | loss 3.7427 | elapsed 0.2s
step 0050 | loss 3.21xx | elapsed 1.8s
step 0500 | loss lower than the first logged loss
checkpoint: openllm_gpu_training_run/mini_gpt2_checkpoint.pt
training_log: openllm_gpu_training_run/training_log.csv
--- sample ---
Open-source language models...
```

不要用生成文本好不好看来判断这次训练。看证据：

| 证据 | 证明什么 |
|---|---|
| `environment_report.json` | Python、torch、CUDA、GPU 型号、参数量、训练设置 |
| `training_log.csv` | loss 在选定设备上被重复记录 |
| `mini_gpt2_checkpoint.pt` | 训练后的权重已经保存 |
| `sample.txt` | 训练后的模型能生成 token |
| `gpu_train_log.txt` | 终端 trace 包含 device、loss、checkpoint 和 sample |
| 关机截图或平台停止记录 | 租 GPU 的成本已经停止 |

如果 `device` 是 `cpu` 或 `mps`，把它标记为冒烟测试。如果 `device` 是 `cuda`，这次运行可以作为本章 GPU 训练证据。

## 8. 什么时候进入 LoRA

不要因为一次输出差就微调。先把 `eval_results.csv` 里的失败分桶：

**缺私有知识**

先试 RAG、补文档、补检索。只有当检索正确但表达或格式仍反复错时，才考虑 LoRA。

**格式不稳定**

先试 schema、few-shot 和解析器。只有同一格式错误在固定样本里反复出现时，才考虑 LoRA。

**风格不稳定**

先改 system prompt 和示例。只有风格问题跨很多样本重复出现时，才考虑 LoRA。

**推理太慢**

先试小模型、量化和 vLLM。只有行为正确但性能约束不达标时，才进入训练路线。

如果确实要 LoRA，先准备三件东西：

```text
train.jsonl       # 高质量训练样本
eval_cases.csv    # 固定评估样本，不能和训练集混
base_model_note.md # 基座模型、许可证、版本、选择理由
```

同时写 `lora_decision.md`：

```md
# LoRA Decision

## Repeated failure

固定评估集中反复失败的样本编号：

## Tried before LoRA

- Prompt/schema:
- RAG/retrieval:
- Smaller or larger model:
- Decoding settings:

## Training data

- Sample count:
- Data owner:
- Privacy check:
- Train/eval split:

## Decision

当前选择：no_lora / prepare_lora / full_finetune_not_allowed

理由：
```

Self-LLM 的 LoRA 教程适合接在这里：你已经有环境报告、基座模型选择、固定评估集、首跑证据和一份小型训练证据包，再去跟模型专项教程会稳得多。

## 9. 按现象排查

如果 `environment_report.py` 失败，先看 Python 版本、虚拟环境是否激活、`torch` 是否装进当前环境。不要急着换模型。

如果 `run_local_llm.py` 下载慢，先继续用 `sshleifer/tiny-gpt2` 跑完整流程。下载大模型只是模型选择问题，不应该阻塞环境、评估和 API 练习。

如果输出为空，先降低 `max_new_tokens` 的复杂度、确认 `pad_token_id` 没报错，再换一个短 prompt 复测。

如果 API 起不来，先检查 8000 端口是否被占用，再确认 `uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000` 是在 `openllm_lab` 目录里执行。

如果 vLLM 报显存不足，先换更小模型或降低并发。不要直接开始 LoRA；显存问题通常不是训练能解决的。

如果 mini GPT-2 训练脚本在 `--device cuda` 失败，先检查 `nvidia-smi`、CUDA 版 PyTorch 是否装对、Notebook runtime 是否真的打开了 GPU。不要在冒烟测试没跑通前租更贵的 GPU。

最终验收只看证据包是否完整：`environment_report.txt`、`first_run.md`、`eval_results.csv`、`eval_summary.json`、API 的 health/request/response 记录。

## 留下的证据

完成这一页后，把这些文件作为实验证据保留下来：

```text
compute_route.md
model_decision.md
environment_report.txt
requirements-freeze.txt
first_run.md
eval_cases.csv
eval_results.csv
eval_summary.json
serve_openai_like.py
gpu_plan.md
openllm_gpu_training_run/environment_report.json
openllm_gpu_training_run/training_log.csv
openllm_gpu_training_run/mini_gpt2_checkpoint.pt
openllm_gpu_training_run/sample.txt
gpu_train_log.txt
lora_decision.md
README.md
```

这组文件要能回答五个问题：在哪跑、跑了哪个模型、固定样本表现如何、真实训练循环是否跑过、服务或 GPU 实例怎样停止或回滚。

## 10. 写 README 交付

新建 `README.md`：

````md
# Open LLM Lab

## 模型

- 烟雾测试：sshleifer/tiny-gpt2
- 下一步尝试：Qwen/Qwen2.5-0.5B-Instruct

## 运行

```bash
source .venv/bin/activate
python environment_report.py
python run_local_llm.py
python eval_openllm.py
uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000
python mini_gpt2_train.py --device cuda --steps 500 --batch-size 64
```

## 证据

- environment_report.txt
- requirements-freeze.txt
- model_decision.md
- first_run.md
- eval_cases.csv
- eval_results.csv
- eval_summary.json
- gpu_plan.md
- openllm_gpu_training_run/environment_report.json
- openllm_gpu_training_run/training_log.csv
- openllm_gpu_training_run/mini_gpt2_checkpoint.pt
- openllm_gpu_training_run/sample.txt
- gpu_train_log.txt
- lora_decision.md

## 停止

用 Ctrl+C 停止本地 API。租用 GPU 时，复制完证据后立即停止实例，并保留关机截图或平台停止记录。
````

<details>
<summary>检查思路与讲解</summary>

合格的实验可以继续使用 `sshleifer/tiny-gpt2`，只要证据完整。小模型证明的是工程链路，不是回答质量。

检查证据包时按这个顺序看：

1. `compute_route.md` 说明本地 CPU、免费 Colab 或租 GPU，并写清这条路线不能证明什么。
2. `environment_report.txt` 和 `requirements-freeze.txt` 让运行可复现。
3. `first_run.md` 记录模型、prompt、设备、延迟和输出。
4. `eval_cases.csv`、`eval_results.csv`、`eval_summary.json` 区分“模型跑起来了”和“模型解决了任务”。
5. API 证据包含 `/health`、一次 `/v1/chat/completions` 请求、一次响应，以及停止方式。
6. `openllm_gpu_training_run/` 证明你见过真实训练循环，而不只是推理。
7. `gpu_plan.md` 和 `lora_decision.md` 避免过早升级：更大模型服务和微调都需要重复评测证据。

最常见错误是在链路跑通前就把烟雾测试换成大模型。先让同一批固定样本能稳定重跑，再升级。
</details>

学完这一页，你不只是“知道可以部署开源模型”，而是已经跑过一条可复现链路：环境 -> 模型 -> 输出 -> 评估 -> API -> GPU 训练证据 -> 停止。
