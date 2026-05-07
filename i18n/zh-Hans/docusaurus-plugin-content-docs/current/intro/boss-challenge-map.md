---
sidebar_position: 5
title: "阶段验收卡"
description: "用每阶段一张验收卡替代冗长挑战地图：成果、证据和复盘。"
keywords: [阶段验收, AI 学习证据, 项目验收, 作品集证据]
---

# 阶段验收卡

![阶段验收卡](/img/course/intro-stage-checkpoint-cards.png)

这页替代旧的“Boss 战”说法，只保留一个简单规则：

> 每个阶段都要留下一个可运行成果、一份证据和一句复盘。

## 验收卡格式

| 部分 | 含义 | 示例 |
| --- | --- | --- |
| Artifact | 你做出了什么 | script、notebook、API、RAG demo、Agent trace |
| Evidence | 别人怎么知道它有效 | 截图、CSV、日志、指标表、引用检查 |
| Reflection | 你学到或修复了什么 | 局限、失败样本、下一步 |

## 阶段卡片

| 阶段 | 最小成果 | 证据 |
| --- | --- | --- |
| 1 工具 | 能运行的项目文件夹 | README 和 Git commit |
| 2 Python | 小 CLI 或 API | 样例输入输出和异常处理 |
| 3 数据 | 清洗后的数据报告 | 图表、质量说明、局限 |
| 4 数学 | 可运行小实验 | 公式到代码的解释 |
| 5 机器学习 | baseline 模型 | 指标和错误样本 |
| 6 深度学习 | 一次训练运行 | loss 曲线和 checkpoint |
| 7 LLM/Prompt | 稳定 prompt 测试 | 固定输入和输出检查 |
| 8 RAG | 带引用的问答 demo | 检索日志和引用表 |
| 9 Agent | 可控任务运行 | 工具 trace 和停止条件 |
| 10-12 拓展方向 | 一个方向项目 | 前后对比和评估 |
| 毕业项目 | 完整 AI 产品 | Demo、评估报告、部署说明 |

## 复盘要短

每阶段结束后用这个模板：

```md
## Stage Checkpoint

Artifact:
Evidence:
One failure:
Fix or next step:
```

如果卡片完整，就继续下一阶段。如果卡片是空的，少读一点，先做出最小成果。
