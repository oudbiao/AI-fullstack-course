---
title: "9.3.1 工具路线图：Schema、权限、观察"
description: "Agent 工具的简短实操路线：设计 schema，校验参数，路由工具调用，记录观察，并保护边界。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "工具概览, Function Calling, Tool Use, Code Agent, Agent 工具"
---

# 9.3.1 工具路线图：结构约束、权限、观察

工具让 Agent 从语言走向行动。更多工具不会自动让 Agent 更强；不清楚的工具会带来错误调用、不安全动作、循环和成本泄漏。

## 先看行动边界

![Agent 工具行动层地图](/img/course/ch09-tools-action-layer-map.webp)

![Agent 工具章节学习顺序图](/img/course/ch09-tools-chapter-flow.webp)

![Agent 受控工具调用闭环图](/img/course/ch09-tool-control-loop.webp)

工具调用必须受控：选择工具、校验参数、检查权限、执行、观察，再决定下一步。

## 跑一个工具 结构约束 检查

执行任何工具调用前，先使用 schema。

```python
tool_call = {
    "name": "search_course_docs",
    "args": {"query": "RAG evaluation", "top_k": 3},
}

schema = {
    "name": "search_course_docs",
    "required": ["query", "top_k"],
    "max_top_k": 5,
}

name_ok = tool_call["name"] == schema["name"]
args_ok = all(field in tool_call["args"] for field in schema["required"])
limit_ok = tool_call["args"]["top_k"] <= schema["max_top_k"]

print("can_execute:", name_ok and args_ok and limit_ok)
print("observation_needed:", True)
```

预期输出：

```text
can_execute: True
observation_needed: True
```

工具运行后，Agent 必须观察并总结结果。不要让模型假装失败的工具已经成功。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 函数调用 | 把模型意图转成结构化行动 |
| 2 | 工具描述 | 写清用途、输入、限制、示例和失败模式 |
| 3 | 工具策略 | 选择工具顺序、fallback、timeout 和停止规则 |
| 4 | 工具安全 | 加入权限、沙箱、审计和人工确认 |
| 5 | 多工具实战 | 记录成功和失败调用 追踪 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
工具契约：名称、描述、输入 schema、输出 schema
权限：工具允许读取或修改的内容
调用轨迹：参数、结果、错误、重试或回退
失败检查：错误的工具、参数不当、不安全操作，或缺少观察结果
安全动作：验证、确认、沙箱、限流，或回滚
```

## 通过标准

如果你能阅读工具 trace，并判断失败发生在规划、参数、执行、观察还是权限控制，就通过了本章。

本章出口小项目是一个学习助手：包含 3 个工具 schema、5 个测试调用、1 条失败调用记录和一份可打印 trace。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要描述 agent 循环：目标、计划、工具调用、观察结果、记忆或状态更新，以及停止条件。
2. 证据应包含另一个开发者可以检查的 trace，而不只是最终回答。
3. 自检时要能说出一个安全或可靠性控制，例如工具 schema、权限边界、重试、评估用例或人工复核点。

</details>
