---
title: "9.9.1 部署路线图：运行时、持久化、恢复"
sidebar_position: 0
description: "Agent 部署与运维的简短实操路线：暴露 API，持久化状态，记录 trace，控制成本，并从失败中恢复。"
keywords: [Agent 部署指南, Agent 运维, 成本优化, 运行时, 可观测性]
---

# 9.9.1 部署路线图：运行时、持久化、恢复

部署 Agent 不只是把代码放到服务器。你需要模型调用、工具服务、队列、状态存储、trace、权限、成本限制和回滚路径。

## 先看运行时闭环

![Agent 生产运行时架构图](/img/course/ch09-production-runtime-map.webp)

![Agent 部署与运维章节学习流程图](/img/course/ch09-deployment-chapter-flow.webp)

![Agent 部署可观测性与恢复闭环图](/img/course/ch09-deployment-observability-loop.webp)

生产问题不是“它是否成功过一次”，而是“它能不能持续工作、安全失败并恢复”。

## 跑一个部署就绪检查

这个检查会暴露缺失的生产基础能力。

```python
service = {
    "api_entry": True,
    "state_store": True,
    "trace_log": True,
    "cost_limit": True,
    "rollback": False,
}

missing = [name for name, ok in service.items() if not ok]

print("ready:", not missing)
print("missing:", missing)
```

预期输出：

```text
ready: False
missing: ['rollback']
```

如果系统不能回滚或恢复，就不要称为生产就绪。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 部署架构 | 画出前端、后端、模型服务、工具服务、存储 |
| 2 | 运行时管理 | 处理同步、异步、长任务、队列和中断 |
| 3 | 持久化与恢复 | 保存任务状态、记忆、trace 和中间结果 |
| 4 | 成本优化 | 追踪模型调用、工具调用、缓存、批处理、路由 |
| 5 | 生产实践 | 加入监控、告警、灰度发布、回滚和权限 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
runtime: queues, workers, state store, tool services, and model endpoint
persistence: checkpoints, event log, memory store, and recovery path
ops_signal: latency, cost, error rate, trace coverage, and saturation
failure_check: stuck run, duplicate action, partial failure, or runaway cost
recovery_action: resume, rollback, cancel, human handoff, or degrade gracefully
```

## 通过标准

如果一个本地 Agent 演示能变成小服务，包含 API 入口、状态持久化、trace 日志、错误响应、成本记录和部署说明，就通过了本章。
