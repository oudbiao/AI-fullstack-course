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
![第13章开源大模型路线图](/img/course/ch13-open-source-llm-overview-route.webp)

第 13 章把开源大模型使用变成工程流程。目标不是收集模型名字，而是能选择计算路线、选择模型，在明确环境里跑起来，用稳定接口暴露出来，评估行为，并留下别人可以复现的证据。

可以把 [Datawhale Self-LLM](https://github.com/datawhalechina/self-llm) 当成模型和案例参考库。本章提供课程自己的学习路径：选择更小，步骤更明确，检查标准更可执行。

默认学习路线是：

1. [13.1 计算路线：本地 CPU、免费 Colab、租 GPU](/zh-cn/ch13-open-source-llm/compute-routes/)
2. [13.2 实操：跑通并服务化一个开源大模型](/zh-cn/ch13-open-source-llm/hands-on-open-llm-lab/)
3. [13.3 模型与运行时决策](/zh-cn/ch13-open-source-llm/model-runtime-decision/)
4. [13.4 服务化、评估与发布 Runbook](/zh-cn/ch13-open-source-llm/serving-evaluation-runbook/)

如果你想直接跟做，也先从计算路线页开始。它会先帮你判断应该用本地 CPU、可用时的免费 Colab，还是租 GPU，然后再复制命令。

## 这一章的位置

你已经能构建 LLM、RAG 和 Agent 工作流。这一章回答另一个问题：

> 当模型不再只是云 API，而是你自己下载、托管、量化、服务化或微调时，工程上会多出什么？

开源大模型工作本质上是系统工程：硬件、驱动、模型文件、运行时、API 契约、日志、评估集和回滚方案都要可控。

## 本章必须证明什么

学完本章，你应该能证明四件事：

| 证明 | 产物 | 为什么重要 |
|---|---|---|
| 计算路线证明 | `compute_route.md` 和环境报告 | 说明这次运行应该在本地 CPU、免费 Colab 还是租 GPU 上进行 |
| 运行时证明 | 首次模型命令、Prompt、输出和 API 请求 | 说明模型能通过可复现接口跑起来 |
| 评估证明 | 固定五条样本评估表 | 区分“一次回答成功”和“行为可比较” |
| 发布证明 | README、停止步骤、回滚备注 | 让另一个工程师也能接手 |

这也是为什么本章会比普通模型教程更严格。开源大模型只有在可重跑、可检查、可停止、可比较时，才真正有工程价值。

## 部署闭环

1. **选型**
   决定模型家族、许可证、尺寸、上下文、语言和模态。留下 model card、许可证说明和选择理由。

2. **准备**
   明确 GPU/CPU、CUDA、PyTorch、磁盘、网络和密钥。留下环境报告和成本估算。

3. **运行**
   选择 Transformers、Ollama、llama.cpp、vLLM、SGLang 或平台运行时。留下精确命令、模型路径和首次回答。

4. **服务化**
   封装 OpenAI 兼容 API、内部 SDK 或批处理脚本。留下请求/响应样例和错误路径。

5. **评估**
   固定 Prompt、RAG 问题、安全问题、延迟和成本指标。留下评估表和失败备注。

6. **适配**
   在 Prompt、RAG、量化、LoRA 或全参微调之间做决策。留下决策说明、adapter 产物和 before/after 记录。

7. **发布**
   整理 README、容器、运行手册、监控和关机方案。留下部署清单和回滚记录。

## 学习顺序与任务清单

1. 在 [13.1 计算路线](/zh-cn/ch13-open-source-llm/compute-routes/) 里选择运行位置，停在 `compute_route.md` 写清本地 CPU、免费 Colab 或租 GPU 以及理由。
2. 验证环境，停在 Python、PyTorch、CUDA/MPS/CPU、磁盘和重置/租用风险记录。
3. 在 [13.2 实操](/zh-cn/ch13-open-source-llm/hands-on-open-llm-lab/) 跑一次本地推理，停在 Prompt、输出、命令和模型版本。
4. 封装成 API 或脚本，停在可重复请求/响应和停止命令。
5. 跑一个小评估集，停在至少五个 Prompt 和通过/失败备注。
6. 进入 [13.3 模型与运行时决策](/zh-cn/ch13-open-source-llm/model-runtime-decision/) 比较模型和运行时，停在“为什么当前组合足够”的说明。
7. 进入 [13.4 服务化、评估与发布 Runbook](/zh-cn/ch13-open-source-llm/serving-evaluation-runbook/) 整理发布路径，停在 README、命令、成本、限制和关机步骤。
8. 判断是否需要微调，停在不微调、LoRA 或全参训练的理由。

本阶段交付物是 `compute_route.md`、可运行 runbook、环境报告、五条样本评估表、模型/运行时决策说明，以及包含停止或回滚步骤的 README。

## 怎样配合 Self-LLM 使用

Self-LLM 更像“模型专项手册”，本章更像“工程闭环模板”。建议这样用：

1. **先用本章建立通用证据包**
   先完成环境报告、模型选择、首次运行、评估表、API 请求/响应和停止步骤。

2. **再去 Self-LLM 找具体模型路线**
   当你要换成 Qwen、Llama、ChatGLM、InternLM 或 Baichuan 等模型时，再查对应模型的下载、推理、微调说明。

3. **最后把证据带回本章模板**
   不管参考了哪个模型教程，都要回到 `model_decision.md`、`eval_cases.csv`、`first_run.md`、`README.md` 和关机记录。

这样做可以避免一个常见问题：教程跑通了，但你说不清模型、环境、许可证、评估和停止流程。

## 模型选择卡

每次换模型前，先写一张选择卡。不要只写“这个模型热门”。

```text
候选模型：
模型来源：
许可证：
参数量和量化方式：
上下文长度：
语言/领域能力：
预计显存或内存：
预计磁盘：
运行时：
选择理由：
暂不选择的模型：
风险：许可证、隐私、下载、显存、速度、输出质量
```

如果这张卡填不完整，先不要租 GPU，也不要开始微调。

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

1. **`project = {...}`**
   这是项目约束卡。它把“我想跑个模型”变成硬件、隐私、用户、延迟和微调需求。先改 `task`、`privacy`、`available_vram_gb`、`needs_fine_tuning`。

2. **`choose_runtime(info)`**
   这是运行时决策规则。它避免你在确认显存前就租 GPU 或下载模型。知道真实实例或本机配置后，再调整显存阈值。

3. **`choose_adaptation(info)`**
   这是微调闸门。私有知识通常先试 RAG，不直接训练。只有固定评估样本反复失败后，才把 `needs_fine_tuning` 设成 `True`。

4. **`plan = {...}`**
   这是部署检查表。它把模型选择、运行时选择和必须保留的证据连起来。你可以加上项目自己的鉴权、日志或回滚证据。

5. **`write_text(...)` 和 `print(...)`**
   这两行把同一份计划既保存到文件，也打印到终端，方便之后复盘。把 `open_llm_runbook.json` 和实验记录一起归档。

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

1. **定义本次运行**
   写清模型尺寸、运行时、第一条 Prompt、最长租用时间。证据是 `gpu_plan.md`，包含停止时间和预算上限。

2. **选择实例**
   选择 Linux、足够显存、足够磁盘、可 SSH 的实例。证据是实例类型、显存、磁盘和小时价格备注。

3. **锁住访问**
   使用 SSH key，默认不公开模型 API，只开放必要端口。证据是安全备注和开放端口。

4. **准备环境**
   运行 `python -V`、torch/CUDA 检查、可用时运行 `nvidia-smi` 和磁盘检查。证据是 `environment_report.txt`。

5. **跑通一条模型路径**
   下载或挂载一个模型，跑一条 Prompt，保存命令、输出和失败备注。证据是 `first_run.md`。

6. **停止并归档**
   把 runbook、日志、评估样本和 README 拷回项目，然后停止或销毁实例。证据是关机截图或停止备注。

最重要的命令往往是最后一个：停止机器。实验成功但一直默默计费，仍然是工程失败。

## 运行时选择

**Transformers**

适合学习、调试和自定义 Python 流水线。如果你马上需要高吞吐服务，先别把它当最终方案。

**Ollama / LM Studio**

适合本地演示、笔记本试跑和交给非工程同学体验。如果你需要精细生产控制，先别用它做主服务。

**llama.cpp**

适合 CPU 或量化边缘实验。如果你需要标准 GPU 服务能力，优先看 vLLM 或 SGLang。

**vLLM**

适合 OpenAI 兼容、高吞吐 API 服务。如果 GPU 或依赖环境还没准备好，先从 Transformers 或小模型开始。

**SGLang**

适合结构化生成、服务化和 Agent 型负载。如果目标只是第一次简单运行，它不是最轻的入口。

**云模型 API**

适合低运维产品原型。如果隐私、成本或延迟要求本地控制，就要回到本地运行时。

先用能证明产品行为的最简单运行时。只有当延迟、成本、隐私或吞吐真的要求时，再升级。

## 微调判断

不要因为一次回答不好就微调。

**缺少私有知识**

先尝试 RAG。只有检索正确但模型行为仍然错时，才考虑微调。

**输出格式不稳**

先尝试 schema、解析器和示例。只有固定样本仍大量失败时，才考虑微调。

**语气或角色不对**

先改 system prompt 和示例。只有很多样本反复出现同类风格问题时，才考虑微调。

**领域术语弱**

先补术语表、RAG 和 few-shot。只有已经有足够高质量领域样本时，才考虑微调。

**太慢或太贵**

先试小模型、量化和批处理。只有行为已满足要求但运行约束不达标时，才进入训练路线。

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
