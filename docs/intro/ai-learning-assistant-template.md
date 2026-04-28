---
sidebar_position: 10
title: "贯穿项目仓库模板：AI 学习助手"
description: "为贯穿全课的 AI 学习助手项目提供目录结构、版本迭代、README、评估、日志和作品集沉淀模板。"
keywords: [AI学习助手, 项目模板, 作品集项目, RAG项目模板, Agent项目模板]
---

# 贯穿项目仓库模板：AI 学习助手

## 本节定位

这一页给“AI 学习助手”贯穿项目一个可直接照着搭建的仓库模板。它不是要求你一开始就把所有目录都写满，而是让你从第 1 站开始就按真实项目方式保存代码、数据、实验、日志、评估和文档。

好的作品集项目不只是功能截图，还应该让别人看得懂你怎么迭代、怎么评估、怎么定位失败、怎么做取舍。

## 推荐目录结构

```text
ai-learning-assistant/
  README.md
  requirements.txt
  .env.example
  data/
    raw/
    processed/
    samples/
  src/
    app/
    rag/
    agent/
    multimodal/
    utils/
  notebooks/
  evals/
    questions.jsonl
    expected_sources.jsonl
    results/
  logs/
    traces/
    failures/
  docs/
    screenshots/
    decisions.md
    changelog.md
  tests/
```

这个结构可以从很小开始。第 1～3 站只需要 `README.md`、`src/`、`data/` 和 `docs/screenshots/`；第 5～6 站开始加入 `notebooks/`、`evals/`；第 8～9 站再加入 `rag/`、`agent/`、`logs/traces/`；第 12 站再加入 `multimodal/`。

## 每个目录放什么

| 目录 | 用途 | 常见内容 |
|---|---|---|
| `data/raw/` | 原始数据 | 学习记录、课程文档、示例文本 |
| `data/processed/` | 清洗后的数据 | 切分后的文档、特征表、索引输入 |
| `src/app/` | 应用入口 | CLI、API、简单 Web 页面 |
| `src/rag/` | RAG 能力 | 文档解析、切分、检索、引用、评估 |
| `src/agent/` | Agent 能力 | 工具定义、任务规划、执行轨迹、权限控制 |
| `src/multimodal/` | 多模态能力 | OCR、截图解析、PDF 页面处理、图文输出 |
| `evals/` | 评估集 | 固定问题、期望来源、评估结果 |
| `logs/` | 复盘材料 | Trace、失败样本、成本和耗时记录 |
| `docs/` | 作品集材料 | 截图、架构图、技术决策、版本记录 |
| `tests/` | 自动化检查 | 数据处理、检索、工具调用和格式测试 |

## 按 1～12 站逐步升级

| 学习站 | 项目版本 | 新增能力 | 应该留下的证据 |
|---|---|---|---|
| 1 | v0.1 项目骨架 | Git、README、目录结构 | 仓库截图、运行说明 |
| 2 | v0.2 命令行助手 | 添加任务、查看任务、保存 JSON | CLI 示例输入输出 |
| 3 | v0.3 学习数据分析 | 完成率、学习时长、主题统计 | 图表和结论 |
| 4 | v0.4 数学直觉卡 | 向量、概率、梯度解释卡 | 概念图和小实验 |
| 5 | v0.5 预测模型 | 学习任务分类或延期预测 | baseline、指标、错误样本 |
| 6 | v0.6 深度学习实验 | 文本或图像分类训练 | loss 曲线、测试结果 |
| 7 | v0.7 Prompt 助手 | 学习计划、笔记摘要、复盘卡 | Prompt 版本和失败样本 |
| 8 | v0.8 RAG 问答助手 | 文档检索、引用、评估集 | 检索片段、来源引用、评估结果 |
| 9 | v0.9 Agent 规划助手 | 工具调用、任务拆解、Trace | 执行轨迹、权限边界、失败恢复 |
| 10～11 | v1.0 方向扩展 | CV 或 NLP 子能力 | 独立方向实验报告 |
| 12 | v1.1 多模态助手 | 截图、PDF、图文复盘卡 | 多模态输入输出、审核清单 |

## README 最小模板

````md
# AI 学习助手

## 项目目标

这个项目帮助学习者记录学习任务、分析学习状态，并逐步升级为能回答课程问题、规划学习任务、理解截图和课件的 AI 助手。

## 当前版本

当前版本：v0.8 RAG 课程问答助手

本版本新增：课程文档读取、文本切分、检索、带来源回答、固定评估问题集。

## 如何运行

```bash
pip install -r requirements.txt
python -m src.app.cli
```

## 示例输入输出

输入：RAG 项目为什么需要评估集？

输出：系统回答、引用来源、检索片段和日志文件路径。

## 评估方式

使用 `evals/questions.jsonl` 中的固定问题，检查是否命中期望来源、答案是否忠实、是否在无答案时拒绝编造。

## 失败样本

记录至少 3 个失败样本：检索不到、引用不准、回答过度概括，并说明下一步怎么修复。

## 下一步计划

加入 Reranking、Query Rewrite、Agent 学习规划和多模态 PDF 理解。
````

## 评估文件示例

```json
{"id":"q001","question":"RAG 项目为什么需要评估集？","expected_sources":["ai-engineering-checklist.md"],"ideal_points":["比较优化效果","避免凭感觉判断","记录失败样本"]}
{"id":"q002","question":"Agent 高风险动作为什么要人工确认？","expected_sources":["ai-engineering-checklist.md","ch09-agent/index.md"],"ideal_points":["权限边界","审计日志","避免自动执行危险操作"]}
```

## Trace 日志示例

```json
{
  "run_id": "2026-04-25-rag-001",
  "user_input": "帮我准备 RAG 阶段复习",
  "steps": [
    {"action": "rewrite_query", "output": "RAGOps 评估 日志 检索质量"},
    {"action": "retrieve", "sources": ["modern-ai-stack.md", "ai-engineering-checklist.md"]},
    {"action": "generate_plan", "cost_estimate": "low"}
  ],
  "final_output": "生成 RAG 复习计划",
  "failure": null
}
```

## 作品集展示建议

展示这个项目时，不要只放最终截图。更好的展示顺序是：先说明学习者问题，再展示产品如何从命令行工具一步步升级，接着展示 RAG 检索片段、Agent 执行轨迹、多模态输入输出，最后展示评估结果、失败样本和下一步计划。

如果面试官问“这个项目难点是什么”，可以回答：难点不是调用模型，而是让系统可复现、可评估、可追踪、可控制成本，并且在回答错、检索错或工具调错时能定位原因。
