---
title: "8.4.1 工程路线图：异步、API、日志、部署"
sidebar_position: 0
description: "大模型工程化的简短实操路线：加入异步控制、API 合约、可观测性、Docker 部署和可追踪运维。"
keywords: [大模型工程指南, 异步编程, API 设计, 日志监控, Docker]
---

# 8.4.1 工程路线图：异步、API、日志、部署

工程化把一个能跑的大模型 Demo 变成软件：Prompt、模型、文档和用户变化后，它仍然可以部署、调试、度量和维护。

## 8.4.1.1 先看 LLMOps 闭环

![大模型工程章节学习顺序图](/img/course/ch08-engineering-chapter-flow.png)

![LLMOps trace 复盘闭环图](/img/course/ch08-llmops-trace-loop.png)

![可观测性日志指标追踪图](/img/course/ch08-observability-logs-metrics-trace-map.png)

第一个工程目标很简单：当答案错了，你能解释是哪一层导致的。

## 8.4.1.2 跑一个 Trace 完整性检查

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

## 8.4.1.3 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 异步编程 | 加入超时、重试、并发限制和取消思维 |
| 2 | API 设计 | 定义请求/响应 schema 和错误码 |
| 3 | 日志与监控 | 记录 Prompt 版本、检索命中、延迟、成本和失败 |
| 4 | Docker 部署 | 用可复现运行说明打包应用 |

## 8.4.1.4 通过标准

如果你的最小应用有运行命令、API 合约、错误处理、日志和一次记录下来的失败排查，就通过了本章。

本章出口小项目是一份工程证据包：一条 trace 日志、一个常见错误、一次修复、一次回归检查和一条部署说明。
