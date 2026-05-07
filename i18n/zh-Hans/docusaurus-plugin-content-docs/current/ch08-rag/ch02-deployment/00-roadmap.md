---
title: "8.2.1 部署路线图：本地模型、服务、统一 API"
sidebar_position: 0
description: "模型部署的简短实操路线：判断模型在哪里运行，暴露成服务，并让应用始终调用稳定的 API 合约。"
keywords: [模型部署指南, 本地模型, 推理服务, 统一 API]
---

# 8.2.1 部署路线图：本地模型、服务、统一 API

部署把模型从 Notebook 实验变成可复用能力。即使模型、供应商、硬件或成本策略变化，应用也应该调用一个稳定接口。

## 先看服务决策

![模型部署章节学习流程图](/img/course/ch08-deployment-chapter-flow.png)

![模型服务选择决策图](/img/course/ch08-model-serving-decision-map.png)

![统一 API 供应商网关图](/img/course/ch08-unified-api-provider-gateway-map.png)

部署选择要平衡质量、延迟、成本、隐私和运维复杂度。最强模型不一定是最该调用的模型。

## 跑一个模型路线检查

设置真实服务工具前，先用它建立判断方式。它把部署变成明确的路由决策。

```python
request = {
    "privacy": "high",
    "latency_ms": 800,
    "quality_need": "medium",
    "budget": "low",
}

if request["privacy"] == "high":
    route = "local model or private endpoint"
elif request["quality_need"] == "high":
    route = "frontier cloud model"
else:
    route = "small hosted model"

print("route:", route)
print("contract:", "/v1/chat/completions")
print("watch:", "latency, cost, errors")
```

预期输出：

```text
route: local model or private endpoint
contract: /v1/chat/completions
watch: latency, cost, errors
```

路线可以变，但应用合约要稳定。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 本地模型 | 加载或调用一个本地/私有模型并记录限制 |
| 2 | 推理服务 | 通过服务端点暴露模型调用 |
| 3 | 统一 API | 为多个供应商保留一个应用接口 |

## 通过标准

如果你能解释模型在哪里运行、应用如何调用、哪些地方会失败，以及要观察哪些指标：延迟、成本、错误、限流和降级行为，就通过了本章。

本章出口小项目是一份模型网关注释或脚本：把一个请求路由到选定模型端点，并记录选择理由。
