---
title: "9.5.1 MCP 路线图：Server、Client、能力"
sidebar_position: 0
description: "MCP 的简短实操路线：理解协议层、Server/Client 职责、tools、resources、prompts 和安全生态集成。"
keywords: [MCP 指南, Model Context Protocol, Agent 工具体系, MCP Server]
---

# 9.5.1 MCP 路线图：Server、Client、能力

MCP 是一种协议层，用更标准的方式把工具、资源和 Prompt 模板连接到模型应用。它不替代 Agent，也不替代工具本身，而是让能力更容易稳定暴露和使用。

## 先看 MCP 边界

![MCP Host Client Server 架构图](/img/course/mcp-host-client-server.webp)

![MCP 章节学习顺序图](/img/course/ch09-mcp-chapter-flow.webp)

![MCP 能力接入桥接图](/img/course/ch09-mcp-capability-bridge.webp)

Function Calling 更关注结构化调用；MCP 更关注外部能力如何通过协议被发现、描述、调用和治理。

## 跑一个能力注册检查

实现真实 MCP Server 之前，先列出它暴露什么，以及 Client 可以调用什么。

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
| 2 | MCP 架构 | 区分 Host、Client、Server、tools、resources、prompts |
| 3 | Server 开发 | 用清晰输入、输出和错误包装一个能力 |
| 4 | Client 集成 | 安全发现并调用 Server 能力 |
| 5 | 生态 | 把 MCP 和 IDE、数据库、浏览器、知识库、Agent 连接起来 |

## 通过标准

如果你能画出 Host-Client-Server 关系，并解释 Server 暴露什么、Client 调用什么、权限在哪里检查，就通过了本章。

本章出口小项目是一个课程资料 MCP Server 设计：一个搜索工具、一个资源 URI 模式、一个 Prompt 模板和一条失败处理规则。
