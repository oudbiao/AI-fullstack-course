---
sidebar_position: 10
title: "AI 学习助手仓库模板"
description: "为贯穿全课的 AI 学习助手项目提供精简仓库结构、README、评估和 trace 模板。"
keywords: [AI 学习助手, 项目模板, 作品集项目, RAG 项目模板, Agent 项目模板]
---

# AI 学习助手仓库模板

![AI 学习助手仓库证据柜](/img/course/intro-ai-assistant-repo-evidence-cabinet.png)

把仓库当成证据柜。每个文件夹都要证明一件事：项目能运行、能复查、能评估或能改进。

## 1. 先从这个结构开始

```text
ai-learning-assistant/
  README.md
  requirements.txt
  .env.example
  src/
  data/
  evals/
  logs/
  docs/
  tests/
```

第 1-3 章只需要 `README.md`、`src/`、`data/` 和 `docs/`。等学到 RAG、Agent、评估和日志，再补 `evals/`、`logs/` 等内容。

## 2. 每个文件夹证明什么

| 文件夹 | 证明 |
|---|---|
| `src/` | 项目有可运行代码 |
| `data/` | 输入和材料是明确的 |
| `evals/` | 结果可以再次评估 |
| `logs/` | 失败和 trace 可以复查 |
| `docs/` | 截图和决策过程可见 |
| `tests/` | 修复后能再次检查 |

## 3. 最小 README

````md
# AI 学习助手

## 目标

## 如何运行
```bash
pip install -r requirements.txt
python -m src.app
```

## 示例

## 评估

## 已知失败

## 下一步
````

## 4. 第一份评估和 trace 文件

```jsonl
{"id":"q001","question":"Why does RAG need citations?","expected_sources":["ch08-rag"]}
```

```json
{
  "run_id": "demo-001",
  "user_input": "Help me review RAG",
  "steps": [
    {"action": "retrieve", "sources": ["ch08-rag"]},
    {"action": "generate_plan", "status": "ok"}
  ],
  "failure": null
}
```

展示项目时，优先展示运行命令、样例输入、评估案例、trace、失败记录和截图。
