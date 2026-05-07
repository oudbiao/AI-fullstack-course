---
sidebar_position: 10
title: "AI 学习助手仓库模板"
description: "AI 学习助手贯穿项目的简短仓库结构、README、评估和 trace 模板。"
keywords: [AI学习助手, 项目模板, 作品集项目, RAG项目模板, Agent项目模板]
---

# AI 学习助手仓库模板

![AI 学习助手仓库证据柜](/img/course/intro-ai-assistant-repo-evidence-cabinet.png)

这个模板不是目录装饰，而是证据柜：代码、数据、日志、评估和截图共同说明项目是否能运行、能复查。

## 最小目录结构

```text
ai-learning-assistant/
  README.md
  requirements.txt
  .env.example
  src/
    app/
    rag/
    agent/
  data/
    raw/
    processed/
  evals/
    questions.jsonl
    results/
  logs/
    traces/
    failures/
  docs/
    screenshots/
    decisions.md
  tests/
```

先从小结构开始。第 1-3 章只需要 `README.md`、`src/`、`data/` 和 `docs/screenshots/`。课程进入对应能力后，再加入 `evals/`、`logs/`、`rag/` 和 `agent/`。

## 每个文件夹证明什么

| 文件夹 | 证明 |
| --- | --- |
| `src/` | 系统有可运行代码 |
| `data/` | 输入和材料是明确的 |
| `evals/` | 结果可以被判断 |
| `logs/` | 失败和 trace 可以复查 |
| `docs/` | 别人能理解项目 |
| `tests/` | 修复后还能再次检查 |

## 最小 README

````md
# AI 学习助手

## 目标
这个助手解决什么学习问题？

## 当前版本
v0.x：

## 如何运行
```bash
pip install -r requirements.txt
python -m src.app.cli
```

## 示例
输入：
输出：

## 评估
用了哪些固定问题、指标或人工检查？

## 失败样本
哪里失败了？下一步要改什么？
````

## 最小评估与 trace 示例

```jsonl
{"id":"q001","question":"为什么 RAG 需要引用？","expected_sources":["ch08-rag"],"ideal_points":["grounding","evaluation","failure cases"]}
```

```json
{
  "run_id": "demo-001",
  "user_input": "帮我复习 RAG",
  "steps": [
    {"action": "retrieve", "sources": ["ch08-rag"]},
    {"action": "generate_plan", "status": "ok"}
  ],
  "failure": null
}
```

展示项目时，把仓库当证据来讲：运行命令、示例数据、评估用例、trace 日志、失败笔记和截图。
