---
title: "9.5.1 MCP 路线图：服务器、客户端、能力"
description: "MCP 的简短实操路线：理解协议层、Server/Client 职责、tools、resources、prompts 和安全生态集成。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "MCP 指南, Model Context Protocol, Agent 工具体系, MCP Server"
---

# 9.5.1 MCP 路线图：服务器、客户端、能力

MCP 是一种协议层，用更标准的方式把工具、资源和 Prompt 模板连接到模型应用。它不替代 Agent，也不替代工具本身，而是让能力更容易稳定暴露和使用。

## 先看 MCP 边界

![MCP Host Client Server 架构图](/img/course/mcp-host-client-server.webp)

![MCP 章节学习顺序图](/img/course/ch09-mcp-chapter-flow.webp)

![MCP 能力接入桥接图](/img/course/ch09-mcp-capability-bridge.webp)

Function Calling 更关注结构化调用；MCP 更关注外部能力如何通过协议被发现、描述、调用和治理。

## 跑一个能力注册检查

实现真实 MCP 服务器之前，先列出它暴露什么，以及客户端可以调用什么。

```python
server = {
    "tools": ["search_docs"],
    "resources": ["course://ch09-agent"],
    "prompts": ["study_plan"],
}

client_request = "search_docs"

print("server_ready:", all(server.values()))
print("can_call:", client_request in server["tools"])
print("boundary:", "server exposes, client calls")
```

预期输出：

```text
server_ready: True
can_call: True
boundary: server exposes, client calls
```

边界模糊，权限和调试也会模糊。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | MCP 概念 | 解释为什么协议层能减少集成混乱 |
| 2 | MCP 架构 | 区分 Host、Client、Server、工具、资源、提示 |
| 3 | 服务器开发 | 用清晰输入、输出和错误包装一个能力 |
| 4 | 客户端集成 | 安全发现并调用服务器能力 |
| 5 | 生态 | 把 MCP 和 IDE、数据库、浏览器、知识库、Agent 连接起来 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
能力：服务器暴露的资源、Prompt 或工具
契约：schema、传输、权限和错误形式
调用轨迹：发现、调用、响应和失败处理
失败检查：架构不兼容、缺少认证、不安全工具或服务器错误
集成动作: 在加入自主能力前先验证服务端契约
```

## 通过标准

如果你能画出 Host-Client-Server 关系，并解释服务器暴露什么、客户端调用什么、权限在哪里检查，就通过了本章。

本章出口小项目是一个课程资料 MCP 服务器设计：一个搜索工具、一个资源 URI 模式、一个 Prompt 模板和一条失败处理规则。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要描述 agent 循环：目标、计划、工具调用、观察结果、记忆或状态更新，以及停止条件。
2. 证据应包含另一个开发者可以检查的 trace，而不只是最终回答。
3. 自检时要能说出一个安全或可靠性控制，例如工具 schema、权限边界、重试、评估用例或人工复核点。

</details>
