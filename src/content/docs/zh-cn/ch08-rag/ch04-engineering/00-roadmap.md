---
title: "8.4.1 工程路线图：异步、API、日志、部署"
description: "大模型工程化的简短实操路线：加入异步控制、API 合约、可观测性、Docker 部署和可追踪运维。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "大模型工程指南, 异步编程, API 设计, 日志监控, Docker"
---
工程化把一个能跑的大模型演示变成软件：Prompt、模型、文档和用户变化后，它仍然可以部署、调试、度量和维护。

## 先看 LLMOps 闭环

![大模型工程章节学习顺序图](/img/course/ch08-engineering-chapter-flow.webp)

![LLMOps 追踪 复盘闭环图](/img/course/ch08-llmops-trace-loop.webp)

![可观测性日志指标追踪图](/img/course/ch08-observability-logs-metrics-trace-map.webp)

第一个工程目标很简单：当答案错了，你能解释是哪一层导致的。

## 跑一个 追踪 完整性检查

每个接近生产的大模型功能，都需要足够的 trace 字段来排查一个错误答案。

```python
trace = {
    "request_id": "demo-001",
    "prompt_version": "rag-v2",
    "retrieval_hits": 2,
    "model_ms": 850,
    "format_ok": True,
    "cost_usd": 0.003,
}

required = ["request_id", "prompt_version", "retrieval_hits", "model_ms", "format_ok", "cost_usd"]

print("trace_ready:", all(field in trace for field in required))
print("debug_fields:", ", ".join(required))
```

预期输出：

```text
trace_ready: True
debug_fields: request_id, prompt_version, retrieval_hits, model_ms, format_ok, cost_usd
```

如果这些字段缺失，调试就会变成猜测。先补日志，再加功能。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 异步编程 | 加入超时、重试、并发限制和取消思维 |
| 2 | API 设计 | 定义请求/响应 结构约束 和错误码 |
| 3 | 日志与监控 | 记录 Prompt 版本、检索命中、延迟、成本和失败 |
| 4 | Docker 部署 | 用可复现运行说明打包应用 |
| 5 | 长上下文、RAG 与记忆选择 | 为边界清晰资料、可搜索语料或持久状态选择最简单的证据策略 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
服务契约：端点、输入模式、输出模式、错误模式
运行信号：延迟、吞吐量、日志、健康检查，或容器状态
可观测性：请求 ID、trace ID、结构化日志或指标
上下文策略：long context、RAG、memory 或 hybrid 的选择与原因
失败检查：超时、重试风暴、缺少日志或部署不匹配
运维动作：backoff、queue、alert、rollout 或 rollback
```

## 通过标准

如果你的最小应用有运行命令、API 合约、错误处理、日志和一次记录下来的失败排查，就通过了本章。

本章出口小项目是一份工程证据包：一条 trace 日志、一个常见错误、一次修复、一次回归检查和一条部署说明。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要能追踪 query、chunks、检索分数、引用证据、最终回答和兜底行为。
2. 证据应包含检索片段、source metadata、带引用的回答，以及至少一个空检索或误检索案例。
3. 自检时要能判断失败来自 chunking、检索、排序、prompt 拼装、资料缺失，还是无依据生成。

</details>
