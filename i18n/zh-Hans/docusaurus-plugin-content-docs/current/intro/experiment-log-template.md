---
sidebar_position: 13
title: "实验日志与 README 模板"
description: "可直接复制的 README、实验日志和失败样本模板，把练习变成作品集证据。"
keywords: [实验日志模板, README模板, AI项目复盘, 作品集]
---

# 实验日志与 README 模板

![AI 产品实验指标闭环](/img/course/elective-ai-product-experiment-metrics-loop.png)

项目真的有命令、输出、指标或失败时，再使用这些模板。模板要短；没人愿意填写的模板只是噪音。

## 最小 README 模板

````md
# 项目名

## 目标
解决什么问题？给谁用？

## 如何运行
```bash
python main.py
```

## 示例输入输出
输入：

输出：

## 评估或检查
你怎么知道结果可接受？

## 失败样本
哪里失败了，为什么，之后如何验证修复？

## 下一步
下一版本要改什么？
````

## 实验日志模板

| 字段 | 填什么 |
| --- | --- |
| `experiment_id` | `rag_exp_003` |
| 目标 | 这次想验证什么 |
| baseline | 和什么对比 |
| 变更 | 这次只改的一个主要变量 |
| 配置 | 模型、Prompt、检索、Agent 或训练设置 |
| 指标 | Accuracy、Hit@k、citation_ok、延迟、成本或人工评分 |
| 结果 | 哪里变好，哪里变差 |
| 决策 | 保留、放弃，还是改后重试 |

## 失败样本模板

| 字段 | 填什么 |
| --- | --- |
| 输入 | 精确到那条失败输入 |
| 预期 | 原本应该发生什么 |
| 实际 | 实际发生什么 |
| 层级 | environment / data / model / prompt / RAG / Agent / deployment |
| 原因 | 当前最可能的解释 |
| 修复 | 你改了什么 |
| 回归检查 | 如何确认它不会再回来 |

## 推荐文件

```text
reports/
  failure_cases.md
  improvement_record.md
evals/
  eval_questions.csv
  citation_check.csv
logs/
  llm_calls.jsonl
  retrieval_logs.jsonl
  agent_traces.jsonl
```

记录不是文书工作。它证明你不只是会展示成功 Demo，也能评估、排障和改进系统。
