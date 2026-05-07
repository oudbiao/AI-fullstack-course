---
sidebar_position: 16
title: "失败与评估索引"
description: "用紧凑索引判断不同 AI 项目应该保留哪些失败样本、评估集、日志或复盘文件。"
keywords: [失败案例索引, 评估模板, 测试用例, AI 项目评估, 作品集]
---

# 失败与评估索引

![AI 项目排障索引图](/img/course/appendix-quick-ref-debug-index-map.png)

成功截图不够。好的 AI 项目要保留可复现失败，也要保留能反复运行的评估案例。

## 1. 先定位失败层

| 现象 | 可能层级 | 保留证据 |
|---|---|---|
| 命令、import 或路径错误 | 环境或 Python | 命令、完整错误、版本信息 |
| 图表或结论不对 | 数据 | 数据样例、清洗记录、前后对比图 |
| 模型分数异常高 | 机器学习评估 | 划分规则、baseline、泄漏检查 |
| Loss 不下降 | 深度学习 | 配置、曲线、张量形状记录 |
| JSON 字段漂移 | Prompt | Prompt 版本、固定测试输入、输出 diff |
| RAG 引用错误来源 | 检索或引用 | chunks、top-k 日志、引用对比 |
| Agent 选错工具 | 工具 schema 或规划 | trace、工具输入输出、停止条件 |
| 本地能跑线上失败 | 部署 | 环境变量、日志、启动命令 |

## 2. 最小文件

```text
reports/
├── failure_cases.md
├── improvement_record.md
└── demo_notes.md

evals/
├── eval_questions.csv
├── prompt_cases.csv
└── agent_tasks.jsonl

logs/
├── llm_calls.jsonl
├── retrieval_logs.jsonl
└── agent_traces.jsonl
```

## 3. 失败样本格式

```md
## 失败标题

- 输入：
- 期望：
- 实际：
- 层级：
- 证据：
- 可能原因：
- 修复：
- 回归测试：
```

失败记录要短，但必须可复现。能回放的失败是工程证据，不是丢脸记录。
