---
sidebar_position: 13
title: "实验记录与 README 模板"
description: "提供可直接复制的项目 README、实验记录、错误样本和复盘模板，帮助学习者把练习沉淀成作品。"
keywords: [实验记录模板, README模板, AI项目复盘, 作品集]
---

# 实验记录与 README 模板

做 AI 项目时，代码只是作品的一部分。真正能体现能力的是：你为什么这样做，怎么验证，失败过什么，下一步怎么改。下面的模板可以直接复制到每个阶段项目 README。

## 项目 README 模板

````md
# 项目名称

## 项目目标

这个项目解决什么问题？用户输入是什么，系统输出是什么？

## 运行方式

```bash
python main.py
```

## 示例输入输出

输入：

```text
这里放一个真实输入
```

输出：

```text
这里放系统输出
```

## 项目结构

```text
project/
  main.py
  data/
  README.md
```

## 方法说明

用了什么数据、模型、工具或 API？为什么这样选？

## 评估方式

用什么指标判断效果？baseline 是什么？有没有错误样本？

## 遇到的问题

记录至少一个环境、数据、模型、接口或工程问题，以及你怎么排查。

## 下一步计划

下一版准备改什么？为什么？
````

如果项目已经进入 Prompt、RAG、Agent 或毕业项目阶段，README 还应该展示工程闭环，而不只是运行方式。

````md
## 系统链路

用户输入 -> Prompt / RAG / Agent -> 工具或知识库 -> 模型输出 -> 校验与日志

## LLM 调用层

- 模型：
- Prompt 版本：
- 结构化输出 schema：
- 错误处理：timeout / retry / fallback
- 成本记录：tokens / latency

## RAG 层

- 原始资料：
- chunk 策略：
- metadata 字段：
- 检索策略：关键词 / 向量 / 混合 / rerank
- 引用检查方式：

## Agent / 工具层

- 工具清单：
- 工具 schema：
- 最大步数：
- 人工确认边界：
- trace 示例：

## 评估结果

| 实验 | 配置 | 指标 | 失败样本 | 结论 |
|---|---|---|---|---|
| baseline |  |  |  |  |
| exp-1 |  |  |  |  |

## 已知限制

- 数据范围：
- 模型限制：
- 成本/延迟限制：
- 安全边界：
````

这个增强模板适合放在阶段 8b、阶段 9 和毕业项目中。它会让别人看到你不仅会调用模型，还会设计可观测、可评估、可复盘的 AI 应用。

## 实验记录模板

| 字段 | 内容 |
|---|---|
| 实验日期 | 例如 2026-04-26 |
| 实验目标 | 这次想验证什么 |
| 数据/输入 | 使用什么数据或样例 |
| 方法/配置 | 模型、Prompt、参数、工具版本 |
| 结果 | 指标、截图、输出样例 |
| 失败样本 | 哪些样本表现不好 |
| 结论 | 这次学到了什么 |
| 下一步 | 下一轮怎么改 |

## AI 应用实验记录模板

当项目涉及 LLM、Prompt、RAG 或 Agent 时，建议用更细的实验表。

| 字段 | 示例 | 说明 |
|---|---|---|
| `experiment_id` | `rag_exp_003` | 每次实验的唯一编号 |
| `goal` | 提升同义问法检索命中率 | 这次实验要解决的问题 |
| `baseline` | 关键词检索 top-k=3 | 对照组是什么 |
| `change` | 加 query rewrite | 本次只改一个主要变量 |
| `prompt_version` | `qa_v2` | 如果改了 Prompt，要记录版本 |
| `retrieval_config` | hybrid, top-k=5, rerank=true | RAG 配置 |
| `agent_config` | max_steps=4, tools=search/read | Agent 配置 |
| `metrics` | Hit@3=0.82, citation_ok=0.76 | 指标变化 |
| `latency_cost` | avg_latency=1.2s, avg_tokens=900 | 成本和延迟 |
| `fixed_cases` | 修复 6 条退款同义问法 | 哪些失败样本变好 |
| `new_failures` | 2 条证书问题被误改写 | 新增副作用 |
| `decision` | 保留，但限制 rewrite 规则 | 是否采用这次改动 |

## 错误样本记录模板

| 样本 | 预期结果 | 实际结果 | 可能原因 | 改进方向 |
|---|---|---|---|---|
| 示例 1 | 应该答 A | 实际答 B | 检索没命中 | 改 chunk 或 query rewrite |

## AI 应用失败样本模板

| 字段 | 示例 |
|---|---|
| 用户输入 | “我这个情况还能退吗？” |
| 预期结果 | 命中退款政策，说明 7 天和学习进度条件 |
| 实际结果 | 只回答“可以申请退款”，漏掉条件 |
| 失败层级 | generation / citation |
| 相关日志 | request_id=`req_001`，source=`refund_policy` |
| 可能原因 | prompt 没要求保留限制条件 |
| 修复动作 | 输出格式增加 `conditions` 字段 |
| 回归测试 | 加入 eval_questions.csv，后续每次都测 |

## 最小项目日志文件建议

进入 AI 应用阶段后，建议至少保留这些文件。不是每个项目都必须全有，但越接近作品集，越应该能看到这些证据。

```text
logs/
├── llm_calls.jsonl
├── retrieval_logs.jsonl
├── agent_traces.jsonl
├── tool_calls.jsonl
└── safety_audit.jsonl

reports/
├── baseline.md
├── failure_cases.md
└── improvement_record.md
```

## 为什么要坚持记录

如果你只保留最终代码，别人很难看出你的成长过程。记录实验、失败和改进，可以证明你不是只会跑 demo，而是能像工程师一样迭代系统。
