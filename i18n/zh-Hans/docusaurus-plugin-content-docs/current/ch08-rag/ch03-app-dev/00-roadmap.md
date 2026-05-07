---
title: "8.3.1 应用开发路线图：API、工具、状态"
sidebar_position: 0
description: "大模型应用开发的简短实操路线：封装 API 调用，校验工具动作，管理对话状态，并构建产品闭环。"
keywords: [大模型应用开发指南, 对话系统, Function Calling, LangChain, 大模型应用]
---

# 8.3.1 应用开发路线图：API、工具、状态

大模型应用开发不是“一个输入框加一个模型 API”。真实功能需要校验输入、调用模型、使用工具、保存状态、解析输出、记录错误，并让用户有可恢复的体验。

## 先看应用闭环

![大模型应用开发章节关系图](/img/course/ch08-app-dev-chapter-flow.png)

![大模型应用开发学习顺序图](/img/course/ch08-app-dev-learning-order-map.png)

![大模型应用能力闭环图](/img/course/ch08-llm-app-capability-loop.png)

本章把一次模型调用升级成可维护的应用闭环：输入、Prompt/上下文、模型、可选工具、校验、输出、反馈。

## 跑一个工具分发检查

Function Calling 的意思是模型提出结构化动作参数，但应用必须负责校验和分发。

```python
model_output = {
    "tool": "search_docs",
    "arguments": {"query": "RAG citations"},
}

allowed_tools = {
    "search_docs": {"required": ["query"]},
    "create_ticket": {"required": ["title", "priority"]},
}

tool = model_output["tool"]
required = allowed_tools[tool]["required"]
validation_ok = all(name in model_output["arguments"] for name in required)

print("validation_ok:", validation_ok)
print("dispatch:", tool if validation_ok else "block")
```

预期输出：

```text
validation_ok: True
dispatch: search_docs
```

不要直接执行模型文本里的工具调用。要校验工具名、参数 schema、权限和失败路径。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | LLM API 实战 | 写一个带超时和错误处理的稳定调用封装 |
| 2 | 框架基础 | 拆分 Prompt、模型、工具、记忆、检索和解析器职责 |
| 3 | Function Calling | 在分发前校验结构化工具参数 |
| 4 | Hugging Face 生态 | 判断托管、本地或浏览器端模型适合哪里 |
| 5 | 对话系统 | 保存会话状态、槽位、记忆和用户反馈 |
| 6 | 文档与模板应用 | 把解析、抽取和生成拆成模块 |

## 通过标准

如果你能构建一个小助手闭环，包含一次 API 调用、一个可选工具调用、一个结构化输出和一条错误路径，就通过了本章。

本章出口小项目是课程问答与学习规划助手：分类用户请求，必要时检索知识，返回结构化建议，并记录反馈。
