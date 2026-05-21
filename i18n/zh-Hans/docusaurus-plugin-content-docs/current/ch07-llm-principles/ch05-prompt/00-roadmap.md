---
title: "7.5.1 Prompt 工程路线图：任务简报、输出、评测"
sidebar_position: 0
description: "Prompt 工程的简短实操路线：把模糊需求变成可复用任务简报、结构化输出和可重复评测。"
keywords: [Prompt 指南, Prompt 工程, 结构化输出, Prompt 评测]
---

# 7.5.1 Prompt 工程路线图：任务简报、输出、评测

Prompt 工程是应用和模型之间的接口。目标不是写一句聪明的话，而是让一次模型调用变得可预测、可解析、可测试、可持续改进。

## 先看 Prompt 闭环

![Prompt 工程章节关系图](/img/course/ch07-prompt-chapter-flow.webp)

![Prompt 三层任务规格图](/img/course/ch07-prompt-spec-three-layer-map.webp)

![Prompt 迭代测试闭环图](/img/course/ch07-prompt-iteration-loop.webp)

当模型大致有能力，但结果含糊、不稳定、格式不对或难以评估时，就优先使用本章的方法。

## 跑一个 Prompt 合约检查

调用任何 LLM 之前，先把 Prompt 写成一个合约：任务、上下文、输出格式和约束。下面的小脚本检查这个合约是否完整到可以测试。

```python
prompt_contract = {
    "task": "Extract chapter metadata",
    "context": "One course markdown file",
    "output_format": ["chapter", "goals", "prerequisites", "risks"],
    "constraints": ["return JSON only", "mark missing facts as null"],
}

required = ["task", "context", "output_format", "constraints"]
missing = [field for field in required if not prompt_contract.get(field)]

print("ready:", not missing)
print("fields:", ", ".join(required))
print("test_case_count:", 3)
```

预期输出：

```text
ready: True
fields: task, context, output_format, constraints
test_case_count: 3
```

![Prompt 合约检查运行结果图](/img/course/ch07-prompt-contract-check-result-map.webp)

如果 `ready` 是 `False`，先补完整任务简报，再继续试更多样例。模糊的 Prompt 会带来模糊的调试。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | Prompt 基础 | 把一个模糊需求改写成任务、上下文、格式、约束 |
| 2 | 进阶 Prompt | 只在有帮助时加入示例、步骤、角色和边界说明 |
| 3 | 结构化输出 | 生成可被程序解析的 JSON、表格或 Markdown |
| 4 | Prompt 实战 | 在同一批固定输入上比较 Prompt 版本 |
| 5 | 评测实验室 | 记录通过率、失败类型和下一次修改 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
prompt_contract: task, context, constraints, output format
fixed_cases: same inputs used across prompt versions
schema_check: structured output validated by parser
failure_note: prompt failure grouped by cause
bridge: Chapter 8 adds retrieved context to this loop
```

## 通过标准

如果你能固定输入集，每次只改一个 Prompt 层，并用证据说明新版本为什么更好，而不是凭感觉判断，就通过了本章。

本章出口小项目是课程内容抽取 Prompt：输入一篇课程文档，输出章节主题、学习目标、前置知识、关键术语、练习建议和风险提醒，格式为 JSON 或 Markdown 表格。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要说明 token、上下文、attention、prompt 和生成行为如何组成一次请求到回答的路径。
2. 证据至少包含一个可复现 prompt 或结构化输出测试，并说明输出为什么通过或失败。
3. 自检时要区分 prompt、RAG、微调和对齐：优先使用能解决已观察问题的最轻方案。

</details>
