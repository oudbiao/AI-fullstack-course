---
title: "9.10.1 プロジェクトロードマップ：Traceable Agent を作る"
sidebar_position: 0
description: "第 9 章 projects の短い実践ロードマップ：goals、plans、tools、memory、traces、safety、evaluation、deployment evidence を持つ Agent portfolio project を作る。"
keywords: [Agent Project Guide, research assistant, data analysis Agent, multi-Agent project, Agent portfolio]
---

# 9.10.1 プロジェクトロードマップ：Traceable Agent を作る

Agent project portfolio は、1 つの final model answer ではなく、traceable execution loop を見せるべきです。

## まず project loop を見る

![Agent comprehensive project roadmap](/img/course/ch09-projects-route-map-ja.webp)

![Agent project learning order diagram](/img/course/ch09-project-learning-order-map-ja.webp)

![Agent project delivery loop diagram](/img/course/ch09-project-delivery-loop-ja.webp)

loop は、goal、plan、tool call、observation、state update、failure handling、stop decision、final output、evaluation です。

## Agent evidence check を動かす

portfolio-ready と呼ぶ前に、このチェックを使います。

```python
project = {
    "goal_defined": True,
    "trace_saved": True,
    "tool_logs": True,
    "failure_case": True,
    "eval_tasks": 10,
}

ready = (
    project["goal_defined"]
    and project["trace_saved"]
    and project["tool_logs"]
    and project["failure_case"]
    and project["eval_tasks"] >= 5
)

print("portfolio_ready:", ready)
print("evidence:", "goal, trace, tools, failure, eval")
```

出力：

```text
portfolio_ready: True
evidence: goal, trace, tools, failure, eval
```

ここが `False` なら、Agent roles を増やす前に evidence を改善します。

## この順番で学ぶ

| 手順 | プロジェクト | 本当に鍛える力 |
|---|---|---|
| 1 | Research assistant | retrieval、citation、summarization、trustworthy output |
| 2 | Data analysis Agent | Python tool calls、table analysis、charts、interpretation |
| 3 | Multi-Agent development team | role division、handoff、review loop、merge ownership |
| 4 | Hands-on workshop | 最小 traceable single-Agent baseline |

project を広げる前に、[9.10.5 実践：Traceable Single-Agent Assistant を作る](./04-stage-hands-on-workshop.md) を実行します。

## プロジェクト成果物基準

| 成果物 | 最低要件 | 強いポートフォリオ版 |
|---|---|---|
| README | goal、run command、dependencies、examples | architecture、trade-offs、cost、safety、retrospective を追加 |
| Architecture | model、tools、memory、state、evaluation、safety | deployment boundary と human handoff を追加 |
| Tool list | callable tools、input/output schema、failures | permission rules と sandbox notes を追加 |
| Execution trace | plan、action、observation、replan、stop | replayable JSONL logs を追加 |
| Failure case | 1 件以上の real failure | 3 件の cause、fix、regression check を追加 |
| Evaluation set | fixed tasks と pass/fail rules | baseline、metrics、comparison experiments を追加 |
| Deployment note | local run 方法 | API entry、environment variables、monitoring、rollback を追加 |

## 合格ライン

別の開発者が Agent run を replay し、各 tool call と observation を inspect し、なぜ stop したか理解し、少なくとも 1 件の failure analysis を見られれば、この章は合格です。

basic version は single-Agent project で十分です。memory、MCP、multi-Agent collaboration、deployment は、trace と evaluation loop が固まってから追加します。
