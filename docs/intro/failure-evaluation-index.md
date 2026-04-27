---
sidebar_position: 16
title: "失败案例与评估模板索引"
description: "按课程阶段和项目类型整理失败样本、评估集、测试样例和复盘模板，帮助学习者把项目做成可验证作品。"
keywords: [失败案例索引, 评估模板, 测试样例, AI项目评估, 作品集]
---

# 失败案例与评估模板索引

AI 项目不能只展示成功样例。越接近作品集，越要能说明：系统在哪些输入下失败，失败属于哪一层，如何复现，修复后如何回归测试。这页把全课分散的失败案例和评估模板收束成一个索引，方便做项目时快速查阅。

## 按失败层级查

| 失败层级 | 常见现象 | 优先回看 | 应保留的证据 |
|---|---|---|---|
| 环境与工具 | 命令找不到、依赖冲突、Git 提交失败 | 1 开发者工具基础、环境准备、排障索引 | 命令记录、环境版本、错误截图 |
| Python 程序 | 文件路径错误、JSON 解析失败、函数逻辑混乱 | 2 Python 编程基础 | 输入输出样例、异常日志、修复记录 |
| 数据分析 | 缺失、重复、异常值导致结论不可信 | 3 数据分析与可视化 | 数据字典、清洗日志、图表解释 |
| 数学与指标 | 相似度、概率、loss 或指标解释不清 | 4 AI 数学基础、5 机器学习 | 小实验、指标说明、复盘笔记 |
| 机器学习 | 数据泄漏、过拟合、baseline 不清 | 5 机器学习 | train/test 划分、baseline、错误样本 |
| 深度学习 | shape mismatch、loss 不降、显存不足 | 6 深度学习与 Transformer | 训练日志、曲线、配置文件 |
| Prompt | JSON 解析失败、字段缺失、标签漂移 | 7 大模型原理与 Prompt | Prompt 版本、固定测试样本、schema |
| RAG | 检索不到、引用不支持、答案幻觉 | 8 LLM 应用与 RAG | chunks、retrieval logs、eval questions |
| Agent | 工具选错、循环、越权、trace 缺失 | 9 AI Agent | tool schema、agent trace、安全边界 |
| 多模态 | OCR 错、生成不可控、版权或肖像风险 | 10 计算机视觉、11 自然语言处理、12 AIGC 与多模态 | 素材来源、人工审核、导出限制 |
| 部署运行 | 本地能跑线上失败、成本不可控 | 工程化、毕业项目指南 | `.env.example`、日志、监控和限流说明 |

## 按项目类型准备评估材料

| 项目类型 | 最小评估材料 | 作品集级评估材料 |
|---|---|---|
| Python 小工具 | 3 个命令输入输出样例 | 正常、异常、空输入、文件损坏等测试样例 |
| 数据分析项目 | 一份数据质量检查表 | 数据字典、清洗前后对比、结论局限性 |
| 机器学习项目 | train/test 指标和 baseline | 交叉验证、错误样本、特征泄漏检查 |
| 深度学习项目 | loss 曲线和验证指标 | 配置记录、训练日志、混淆矩阵、失败图片或文本 |
| Prompt 项目 | 固定输入和输出对比 | Prompt 版本表、schema 校验、回归测试集 |
| RAG 项目 | 10 个固定问题和来源检查 | gold_doc、gold_answer、citation_ok、失败类型统计 |
| Agent 项目 | 3～5 个固定任务 | 完成率、平均步数、工具错误率、越权测试和成本 |
| 多模态项目 | 1 个完整素材到输出案例 | 成功、失败、边界、人工编辑和审核记录 |
| 毕业项目 | 20～50 个固定测试问题或任务 | baseline、优化记录、失败归因和演示脚本 |

## 推荐文件命名

建议每个项目至少保留一个 `reports/` 或 `evals/` 目录。文件名可以保持统一，后续整理作品集会更轻松。

```text
reports/
├── baseline.md
├── failure_cases.md
├── improvement_record.md
└── demo_notes.md

evals/
├── eval_questions.csv
├── prompt_eval_cases.csv
├── agent_tasks.jsonl
└── citation_check.csv

logs/
├── llm_calls.jsonl
├── retrieval_logs.jsonl
├── agent_traces.jsonl
└── tool_calls.jsonl
```

## 一个失败样本的最低格式

```md
## 失败样本标题

- 输入：触发失败的真实输入
- 预期：本来应该发生什么
- 实际：系统实际输出或行动
- 层级：环境 / 数据 / 模型 / Prompt / RAG / Agent / 部署
- 证据：日志、截图、trace、检索片段或代码位置
- 原因：当前最可能的解释
- 修复：准备改什么
- 回归：以后用哪个测试样例防止复发
```

失败样本不需要写得很长，但必须能复现。一个不能复现的失败，只能算印象；一个能复现、能定位、能回归测试的失败，才是作品集里的工程证据。

## 和其他模板页的关系

如果你需要完整 README 和实验记录格式，看 [实验记录与 README 模板](/intro/experiment-log-template)。如果你正在做 LLM、Prompt、RAG 或 Agent 项目，看 [AI 应用失败样本库](/intro/ai-failure-samples)。如果你要准备毕业项目，看 [毕业项目设计指南](/intro/graduation-project-guide)。
