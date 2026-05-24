---
title: "13.2 模型与运行时决策"
description: "把模型选择变成工程决策：许可证、尺寸、上下文、硬件、量化、运行时和兜底路径。"
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "开源大模型选型, 运行时决策, Ollama, llama.cpp, vLLM, SGLang, 量化"
---
![开源大模型运行时闭环](/img/course/ch13-open-source-llm-runtime-loop.webp)

一个好的开源大模型项目，在第一次下载模型前就已经开始了。本节把模型选择变成书面决策，避免你把时间浪费在硬件、许可证、延迟目标或产品边界都撑不住的模型上。

## 本页解决什么

初学者常问：“哪个模型最好？”工程问题更具体：“哪个模型/运行时组合，足够支撑这个项目、这台机器、这个许可证约束和这条回滚路径？”

先用能证明项目行为的最小模型和最简单运行时。只有证据表明质量、上下文长度、吞吐、隐私或成本需要升级时，再往上走。

## 决策阶梯

1. **任务适配**
   判断项目需要聊天、抽取、代码、多语言、长上下文、工具调用，还是多模态行为。

2. **许可证适配**
   在围绕模型构建系统前，先阅读 model card 和许可证。保留商业使用、再分发和数据使用限制说明。

3. **硬件适配**
   下载前先估算显存/内存和磁盘。如果本地跑不动，就选择更小模型、量化、租 GPU 或云 API 兜底。

4. **运行时适配**
   学习和调试用 Transformers，本地交付用 Ollama/LM Studio，CPU/边缘量化实验用 llama.cpp，服务化推理用 vLLM/SGLang。

5. **证据适配**
   只有拿到模型版本、运行命令、首次输出、评估表和停止步骤，决策才算完成。

## 模型决策表

创建 `model_runtime_decision.md`：

```md
# 模型与运行时决策

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

具体模型名会变化，但决策结构不应该变化。

## 运行时选择规则

**需要看清 token、prompt 和 Python 行为时，从 Transformers 开始。** 它容易调试，也贴近模型接口，但通常不是最终高吞吐服务。

**目标是笔记本演示或非工程同伴交接时，用 Ollama 或 LM Studio。** 它们降低安装门槛，但生产控制力也更少。

**CPU、量化或边缘约束重要时，用 llama.cpp。** 它适合小型本地实验，但仍然需要清晰 API 和评估方案。

**项目需要 OpenAI 兼容服务和吞吐时，用 vLLM。** 只有 GPU、驱动、内存和安全边界清楚后再进入这一步。

**需要结构化生成或 agentic serving 模式时，用 SGLang。** 它很强，但仍然要由项目需求驱动，而不是因为新奇。

## 小练习

从第 8 章或第 9 章选一个项目，写一段决策：

```text
这个项目我会先用 _____ 和 _____，因为 _____。
我暂时不用更大模型，因为 _____。
只有当固定评估集显示 _____ 时，我才切换。
```

<details>
<summary>判断思路与讲解</summary>

好的答案会把模型尺寸和运行时绑定到证据。例如：先用小型 instruct model 加 Transformers 或 Ollama，证明 prompt、RAG 上下文和输出 schema；只有同一组评估样本质量可接受，并且项目确实需要服务接口时，再切到 vLLM。在知道 prompt、RAG、schema 和量化之后还剩哪些失败前，不要直接选择 LoRA 或更大的 GPU。

</details>

## 留下的证据

```text
model_decision: 选择的模型、许可证说明、尺寸、上下文长度和暂不选择项
runtime_decision: 选择的运行时、硬件理由和兜底运行时
hardware_note: 本地 CPU/GPU 或租用 GPU 估算、磁盘和预计停止时间
eval_gate: 支持切换模型或运行时的固定评估样本
expected_output: model_runtime_decision.md 和一条首次运行命令
```

## 通过标准

如果你能解释为什么当前模型/运行时组合足够、什么证据会触发升级，以及哪些证据能避免“随机模型演示”变成不可控部署，就通过本节。
