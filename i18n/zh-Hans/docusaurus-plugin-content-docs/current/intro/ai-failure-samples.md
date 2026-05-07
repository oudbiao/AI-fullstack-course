---
sidebar_position: 15
title: "AI 应用失败样本库"
description: "一份简短失败样本索引，用于记录 LLM API、Prompt、RAG、Agent、安全和部署问题。"
keywords: [AI失败样本, RAG排障, Agent排障, Prompt排障, LLM应用]
---

# AI 应用失败样本库

![AI 项目速查排障索引图](/img/course/appendix-quick-ref-debug-index-map.png)

失败样本记录的是一次真实输入下系统没有按预期工作的情况。它能帮助排障，也能防止同样问题回来。

## 失败层级

| 层级 | 常见现象 | 先查什么 |
| --- | --- | --- |
| LLM API | timeout、rate limit、空响应、成本高 | request_id、原始响应、tokens、延迟 |
| Prompt/schema | JSON 无效、字段缺失、标签漂移 | schema、示例、解析器、固定测试 |
| RAG | 来源错误、引用弱、没检索到文档 | 检索片段、metadata、citation_ok |
| Agent/tool | 选错工具、参数错、循环、缺 trace | 工具 schema、最大步数、action/observation |
| 安全 | 越权、敏感日志、不安全动作 | allowlist、人工确认、审计日志 |
| 部署 | 只在本机能跑、密钥问题、运行不稳 | `.env.example`、依赖版本、启动日志 |

## 失败样本模板

```md
## 失败样本

用户输入：
预期：
实际：
层级：
相关日志：
可能原因：
修复：
回归测试：
是否解决：
```

每个作品集项目至少保留 3 个失败样本。好的项目不是隐藏失败，而是展示如何定位和修复失败。
