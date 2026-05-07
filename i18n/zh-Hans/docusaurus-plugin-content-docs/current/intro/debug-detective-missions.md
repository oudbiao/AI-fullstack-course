---
sidebar_position: 6
title: "Debug 侦探任务集"
description: "一组简短的图解排障卡，用来处理环境、数据、模型、RAG、Agent 和部署中的常见失败。"
keywords: [Debug, 排障, AI工程错误, 新手练习]
---

# Debug 侦探任务集

![Debug 侦探任务集](/img/course/debug-detective-missions.png)

Debug 不是失败证明，而是工程能力开始生长的地方。

遇到问题时，不要随机改。每次都用同一个调查闭环：

```mermaid
flowchart LR
  A["保存现象"] --> B["猜 2 个原因"]
  B --> C["跑最小测试"]
  C --> D["定位层级"]
  D --> E["写修复笔记"]
```

## 案件卡

| 案件 | 现象 | 先查什么 | 保留什么证据 |
| --- | --- | --- | --- |
| 命令失踪 | 找不到 `python`、`pip`、`npm` 或 `docusaurus` | 当前目录、PATH、安装版本、激活环境 | 终端输出和版本记录 |
| JSON 损坏 | `JSONDecodeError` | 空文件、缺括号、多逗号、JSON/JSONL 混用 | 损坏样本和恢复逻辑 |
| DataFrame 列缺失 | `KeyError: 'minutes'` | `df.columns.tolist()`、分隔符、表头行、空格 | 清洗后的列名和数据字典 |
| 模型分数过高 | accuracy 异常漂亮 | 数据泄漏、重复行、答案列进特征、baseline | baseline 指标和泄漏检查 |
| LLM JSON 漂移 | 输出有时不符合 schema | Prompt 示例、解析校验、重试逻辑 | Prompt 版本表和失败输出 |
| RAG 找不到证据 | 答案没有有效来源 | 先关闭生成，只打印检索结果 | 检索日志和失败问题 |
| 引用不支撑答案 | 有引用，但引用内容不支持回答 | 逐句检查证据是否支撑 | 引用检查表 |
| Agent 原地打转 | 重复同一计划或工具 | 最大步数、停止条件、trace 字段 | `agent_traces.jsonl` |
| 工具越权 | Agent 随意写入、发送或删除 | 工具风险等级、allowlist、人工确认 | 权限表和阻断测试 |
| 只在本机能跑 | 别人运行失败 | README、依赖、`.env.example`、示例数据 | 干净环境运行日志 |

## 最小修复笔记

可以写进 `failure_cases.md`、`debug_notes.md` 或 README：

```md
## 案件

现象：
我原本期待：
实际发生：
疑似层级：
最小测试：
根因：
修复：
回归检查：
```

目标不是收集错误，而是让下一次类似错误更容易定位。
