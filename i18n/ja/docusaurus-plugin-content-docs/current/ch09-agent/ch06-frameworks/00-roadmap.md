---
title: "9.6.1 Frameworks ロードマップ：必要なときだけ選ぶ"
sidebar_position: 0
description: "Agent frameworks の短い実践ロードマップ：LangGraph、LlamaIndex、CrewAI、AutoGen を比較し、state、data、roles、risk に基づいて選ぶ。"
keywords: [Agent Framework Guide, LangGraph, LlamaIndex, CrewAI, AutoGen]
---

# 9.6.1 Frameworks ロードマップ：必要なときだけ選ぶ

Framework は Agent を自動的に賢くしません。task が十分に複雑になったとき、state、tools、workflows、memory、logs、collaboration を整理するための abstraction です。

## まず selection map を見る

![Agent framework position map](/img/course/ch09-frameworks-position-map-ja.webp)

![Agent framework selection map](/img/course/ch09-framework-selection-map-ja.webp)

![Agent framework selection decision map](/img/course/ch09-framework-selection-decision-map-ja.webp)

task が 3 つの固定 steps だけなら、plain Python functions のほうが良いことがあります。state、branching、recovery、data connection、role collaboration が管理しづらくなったら framework を入れます。

## Framework route check を動かす

人気だからという理由で framework を選ぶ前に、このチェックを使います。

```python
task = {
    "needs_state": True,
    "needs_rag": False,
    "needs_roles": False,
    "needs_resume": True,
}

if task["needs_state"] or task["needs_resume"]:
    route = "LangGraph-style state graph"
elif task["needs_rag"]:
    route = "LlamaIndex-style data app"
elif task["needs_roles"]:
    route = "CrewAI or AutoGen-style collaboration"
else:
    route = "plain functions first"

print("route:", route)
print("reason:", "choose the smallest abstraction that exposes state")
```

期待される出力：

```text
route: LangGraph-style state graph
reason: choose the smallest abstraction that exposes state
```

Framework choice は README に trade-off として書きます。dependencies の中に隠さないでください。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Framework overview | framework が何を抽象化するか説明する |
| 2 | LangChain / LangGraph | state、nodes、edges、branches、recovery を model 化する |
| 3 | LlamaIndex | documents、indexes、retrieval、evaluation を接続する |
| 4 | CrewAI / AutoGen | role collaboration と multi-Agent conversation を比較する |
| 5 | Framework selection | decision table と no-framework baseline を書く |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
problem_shape: workflow graph, retrieval app, role team, or experiment
framework_choice: what abstraction it adds and what control it hides
trace: state, node, tool call, message, or run id
failure_check: framework magic hides state, retries, or permissions
decision: choose framework only after the single-agent loop is clear
```

## 合格ライン

同じ小さな task を plain functions と 1 つの framework で実装し、どちらが debug しやすいか、なぜかを説明できれば、この章は合格です。
