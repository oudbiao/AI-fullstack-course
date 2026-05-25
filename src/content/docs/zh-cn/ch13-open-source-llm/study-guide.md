---
title: "13.0 学习检查表：开源大模型部署"
description: "第 13 章检查表：模型选型、运行时选择、环境检查、服务化证据、评估和微调决策。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "开源大模型检查表, 本地模型部署, LoRA 检查表, vLLM 检查表"
---
这页当成可打印检查表使用。需要完整讲解时，回到 [第 13 章入口页](/zh-cn/ch13-open-source-llm/)。

![第13章学习检查表](/img/course/ch13-open-source-llm-study-checklist.webp)

如果你还没有亲手运行，请先完成 [13.1 计算路线：本地 CPU、免费 Colab、租 GPU](/zh-cn/ch13-open-source-llm/compute-routes/)，再完成 [13.2 实操：跑通并服务化一个开源大模型](/zh-cn/ch13-open-source-llm/hands-on-open-llm-lab/)。然后用 [13.3 模型与运行时决策](/zh-cn/ch13-open-source-llm/model-runtime-decision/) 和 [13.4 服务化、评估与发布 Runbook](/zh-cn/ch13-open-source-llm/serving-evaluation-runbook/) 补齐部署证据。

## 两小时快速通读

1. **20 分钟：选择计算路线**
   能说出“这次运行应该在本地 CPU、免费 Colab 或租 GPU 上做，并且我知道这条路线不能证明什么”就停。

2. **20 分钟：运行环境检查**
   能说出“我知道这台机器能不能用 CUDA，还是只能用 CPU”就停。

3. **25 分钟：运行 runbook 脚本**
   能说出“我能根据硬件和项目约束选择运行时”就停。

4. **25 分钟：建一个五条 Prompt 评估表**
   能说出“我能在改运行时或微调前比较模型行为”就停。

5. **30 分钟：写适配决策**
   能说出“我能解释为什么选 Prompt、RAG、量化、LoRA 或不微调”就停。

6. **30 分钟：写发布 runbook**
   能说出“另一个工程师可以启动、测试、停止和回滚这个服务”就停。

## 必须留下的证据

- `environment_report.txt`：Python、torch、CUDA/device、platform、磁盘或实例说明。
- `compute_route.md`：本地 CPU、免费 Colab 或租 GPU 选择，以及 fallback 和 stop rule。
- `model_decision.md`：模型、尺寸、许可证、来源、理由、被拒方案。
- `model_runtime_decision.json`：按本地 CPU、免费 Colab、租 GPU 路线输出的运行时建议。
- `open_llm_runbook.json`：运行时选择、适配选择、需要保留的证据。
- `api_smoke_test.json`：本地 OpenAI-compatible API 的健康检查和一次请求/响应证据。
- `first_run.md`：精确命令、Prompt、输出、延迟或显存备注。
- `eval_cases.csv`：至少五个 Prompt、期望行为、pass/fail、备注。
- `README.md`：设置、运行、评估、停止服务、回滚或关机。

## 质量闸门

- **可复现**：另一个工程师能识别模型版本、运行时、命令和环境。
- **安全**：对外共享前检查许可证、隐私、鉴权、日志和关机。
- **评估**：运行时或微调变化都用同一组评估样本比较。
- **成本控制**：记录免费 Notebook 限制或 GPU 租用时长、显存、延迟和停止流程。
- **适配决策**：微调来自重复证据，而不是一次不满意回答。

## 离章问题

- 你能解释为什么选择这个模型尺寸和许可证吗？
- 你能解释为什么这次运行适合本地 CPU、免费 Colab 或租 GPU 吗？
- 你能说明为什么这个运行时足够当前项目使用吗？
- 你能运行或复现环境检查吗？
- 改动后，你能用同一组五个 Prompt 比较输出吗？
- 你能为 Prompt、RAG、量化、LoRA 或全参微调的选择辩护吗？

如果答案都是可以，你就能把开源大模型当成工程选项，而不是随机模型 demo 集合。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
环境报告：Python、torch、CUDA/device、platform、硬件/成本备注
计算路线：local CPU / free Colab / rented GPU、fallback、stop rule
模型决策：选中模型、许可证、尺寸、来源、被拒方案
运行契约：命令或 endpoint、请求格式、响应格式、错误路径
评估记录：固定 Prompt、输出、pass/fail、延迟或显存备注
适配选择：Prompt/RAG/量化/LoRA/全参微调的选择和理由
```
