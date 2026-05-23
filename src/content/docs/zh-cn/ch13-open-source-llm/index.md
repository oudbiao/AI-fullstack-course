---
title: "13 开源大模型部署与微调"
description: "学习如何选择、运行、服务化、评估和轻量微调开源大模型，并留下可复现的环境、运行时和交付证据。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "开源大模型, 本地大模型部署, vLLM, SGLang, Transformers, LoRA, 模型服务"
---
![开源大模型运行时部署闭环](/img/course/ch13-open-source-llm-runtime-loop.webp)

第 13 章把开源大模型使用变成工程流程。目标不是收集模型名字，而是能选择一个模型，在明确环境里跑起来，用稳定接口暴露出来，评估行为，并留下别人可以复现的证据。

可以把 [Datawhale Self-LLM](https://github.com/datawhalechina/self-llm) 当成模型和案例参考库。本章提供课程自己的学习路径：选择更小，步骤更明确，检查标准更可执行。

## 这一章的位置

你已经能构建 LLM、RAG 和 Agent 工作流。这一章回答另一个问题：

> 当模型不再只是云 API，而是你自己下载、托管、量化、服务化或微调时，工程上会多出什么？

开源大模型工作本质上是系统工程：硬件、驱动、模型文件、运行时、API 契约、日志、评估集和回滚方案都要可控。

## 部署闭环

| 步骤 | 要做的决策 | 留下的证据 |
|---|---|---|
| 选型 | 模型家族、许可证、尺寸、上下文、语言、模态 | model card、许可证说明、选择理由 |
| 准备 | GPU/CPU、CUDA、PyTorch、磁盘、网络、密钥 | 环境报告、成本估算 |
| 运行 | Transformers、Ollama、llama.cpp、vLLM、SGLang 或平台运行时 | 精确命令、模型路径、首次回答 |
| 服务化 | OpenAI 兼容 API、内部 SDK 或批处理脚本 | 请求/响应样例、错误路径 |
| 评估 | 固定 Prompt、RAG 问题、安全问题、延迟/成本 | 评估表、失败备注 |
| 适配 | Prompt、RAG、量化、LoRA 或全参微调 | 决策说明、adapter 产物、before/after |
| 发布 | README、容器、运行手册、监控、关机方案 | 部署清单、回滚记录 |

## Learning Order And Task List

| 顺序 | 做什么 | 停在什么证据上 |
|---|---|---|
| 1 | 选一个模型和一个运行时 | 模型/运行时决策说明 |
| 2 | 验证环境 | Python、PyTorch、CUDA 或 CPU 状态 |
| 3 | 跑一次本地推理 | Prompt、输出、命令、模型版本 |
| 4 | 封装成 API 或脚本 | 可重复的请求/响应 |
| 5 | 跑一个小评估集 | 至少五个 Prompt 和通过/失败备注 |
| 6 | 判断是否需要微调 | 不微调、LoRA 或全参训练的理由 |
| 7 | 打包运行手册 | README、命令、成本、限制、关机 |

本阶段交付物是可运行 runbook、环境报告、五条样本评估表、模型/运行时决策说明，以及包含停止或回滚步骤的 README。

## 第一个可运行循环：生成模型运行手册

这个离线脚本不会下载模型。它训练的是租 GPU 或下载大模型前最重要的规划习惯。

新建 `ch13_open_llm_runbook.py`，用 Python 3.10 或更新版本运行。

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

预期输出：

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

这个输出就是你的部署 runbook 雏形。真正跑大模型前，先改显存、隐私要求、任务和是否微调。

## 逐段读懂 runbook

| 代码部分 | 它在做什么 | 先改哪里 |
|---|---|---|
| `project = {...}` | 项目约束卡。把“我想跑个模型”变成硬件、隐私、用户、延迟和微调需求。 | 先改 `task`、`privacy`、`available_vram_gb`、`needs_fine_tuning`。 |
| `choose_runtime(info)` | 运行时决策规则。避免你在确认显存前就租 GPU 或下载模型。 | 知道真实实例或本机配置后，再调整显存阈值。 |
| `choose_adaptation(info)` | 微调闸门。私有知识通常先试 RAG，不直接训练。 | 只有固定评估样本反复失败后，才把 `needs_fine_tuning` 设成 `True`。 |
| `plan = {...}` | 部署检查表。把模型选择、运行时选择和必须保留的证据连起来。 | 加上项目自己的鉴权、日志或回滚证据。 |
| `write_text(...)` 和 `print(...)` | 同一份计划既保存到文件，也打印到终端，方便之后复盘。 | 把 `open_llm_runbook.json` 和实验记录一起归档。 |

如果你能解释每一行，就真的读懂了这个脚本。如果某一行说不清楚，先改项目约束卡并重跑，不要急着碰 GPU。

## 最小环境检查

下载模型前先运行：

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

保存输出。如果环境不可见，模型效果就不可复现。

## 租 GPU 不跑偏的闭环

把租来的 GPU 当成短实验，不要当成永久电脑。免费或免租路径优先：本地量化模型、学校或公司 GPU、Notebook 平台的免费额度。如果这些不够用，再只租足够验证一个模型/运行时闭环的时间。

| 步骤 | 动作 | 保存什么证据 |
|---|---|---|
| 1. 定义本次运行 | 写清模型尺寸、运行时、第一条 Prompt、最长租用时间。 | `gpu_plan.md`，包含停止时间和预算上限 |
| 2. 选择实例 | 选择 Linux、足够显存、足够磁盘、可 SSH 的实例。 | 实例类型、显存、磁盘、小时价格备注 |
| 3. 锁住访问 | 使用 SSH key，默认不公开模型 API，只开放必要端口。 | 安全备注和开放端口 |
| 4. 准备环境 | 运行 `python -V`、torch/CUDA 检查、可用时运行 `nvidia-smi` 和磁盘检查。 | `environment_report.txt` |
| 5. 跑通一条模型路径 | 下载或挂载一个模型，跑一条 Prompt，保存命令、输出和失败备注。 | `first_run.md` |
| 6. 停止并归档 | 把 runbook、日志、评估样本和 README 拷回项目，然后停止或销毁实例。 | 关机截图或停止备注 |

最重要的命令往往是最后一个：停止机器。实验成功但一直默默计费，仍然是工程失败。

## 运行时选择

| 运行时 | 适合什么时候用 | 什么时候先别用 |
|---|---|---|
| Transformers | 学习、调试、自定义 Python 流水线 | 你马上需要高吞吐服务 |
| Ollama / LM Studio | 本地演示、笔记本试跑、交给非工程同学体验 | 你需要精细生产控制 |
| llama.cpp | CPU 或量化边缘实验 | 你需要标准 GPU 服务能力 |
| vLLM | OpenAI 兼容、高吞吐 API 服务 | GPU 或依赖环境还没准备好 |
| SGLang | 结构化生成、服务化、Agent 型负载 | 你只想做最简单的第一次运行 |
| 云模型 API | 低运维产品原型 | 隐私、成本或延迟要求本地控制 |

先用能证明产品行为的最简单运行时。只有当延迟、成本、隐私或吞吐真的要求时，再升级。

## 微调判断

不要因为一次回答不好就微调。

| 现象 | 先尝试 | 只有在什么情况下才微调 |
|---|---|---|
| 缺少私有知识 | RAG | 检索正确，但模型行为仍然错 |
| 输出格式不稳 | schema、解析器、示例 | 固定样本仍大量失败 |
| 语气或角色不对 | system prompt 和示例 | 很多样本反复出现同类风格问题 |
| 领域术语弱 | 术语表、RAG、few-shot | 已有足够高质量领域样本 |
| 太慢或太贵 | 小模型、量化、批处理 | 行为已满足要求但运行约束不达标 |

多数课程项目里，LoRA 是第一个严肃适配方法。全参微调是后续工程选择，不是默认选项。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
模型选择：模型、尺寸、许可证和选择理由
运行时选择：Transformers/Ollama/llama.cpp/vLLM/SGLang/API 以及原因
环境信息：Python、torch、CUDA/device、磁盘和成本估算
首次运行：精确命令、prompt、输出、延迟、显存备注
适配决策：Prompt/RAG/量化/LoRA/全参微调的选择
期望产出：runbook、评估表、README、回滚或关机说明
```

## 常见错误

- 没查磁盘、网络和显存就下载大模型。
- 把一次聊天成功当成部署证据。
- 忽略模型许可证或数据使用限制。
- 没有固定 before/after 评估集就微调。
- 暴露本地模型服务时没有鉴权、日志和关机规则。
- 实验结束后忘记停止租用 GPU。

## 通过标准

能为一个项目选择模型和运行时，跑环境检查，生成 `open_llm_runbook.json`，解释下一步应该用 Prompt、RAG、量化、LoRA 还是全参微调，并写出别人能跟着执行的 README 命令，就算通过。
