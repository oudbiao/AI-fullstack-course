---
sidebar_position: 1
title: "AI 全栈能力地图"
description: "用一张紧凑地图理解 AI 全栈学习的七层能力。"
keywords: [AI 全栈, 能力地图, AI 学习路线, LLM 应用, RAG, AI Agent]
---

# AI 全栈能力地图

![AI 全栈能力总地图](/img/course/intro-ai-fullstack-capability-map.png)

把这页当地图，不要当背诵清单。课程主线只有一句话：

```text
tools -> data -> models -> LLMs -> applications -> Agents -> engineering
```

## 七层能力

| 层级 | 为什么重要 | 应该留下什么成果 |
| --- | --- | --- |
| Tools | 你需要稳定地写、跑、保存项目 | 可运行文件夹、README、Git commit |
| Data | AI 项目从可检查的数据开始 | 数据报告、图表、清洗后文件 |
| Models | 你需要理解模型如何学习、如何失败 | baseline、指标、错误样本 |
| LLMs | Prompt、Embedding、Transformer 和上下文不再神秘 | Prompt 测试、解释笔记 |
| Applications | 模型能力变成用户可用功能 | 聊天、文档工具、知识库问答 |
| Agents | AI 能分步计划、调用工具、保留轨迹 | 工具日志、任务 trace、权限规则 |
| Engineering | 真实项目需要部署、评估、成本和安全 | Demo、监控说明、评估报告 |

## 怎么读这张图

从上到下看一遍，然后停下。第一次不需要理解每个分支。

迷路时只问一个问题：

> 我当前项目卡在哪一层？

如果代码跑不起来，回到 Tools。答案没有证据，回到 Data 或 RAG。Agent 行为不可控，回到 trace、权限和评估。
