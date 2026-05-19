---
title: "9.1.1 Agent 基礎ロードマップ：目標、状態、行動"
sidebar_position: 0
description: "Agent 基礎の短い実践ロードマップ：Agent、chatbot、workflow を区別し、最小の goal-state-action loop を作る。"
keywords: [Agent guide, intelligent agent guide, Agent system architecture, tool calling, Agent loop]
---

# 9.1.1 Agent 基礎ロードマップ：目標、状態、行動

Agent はモデル名ではありません。目標に向かって、モデル、ツール、状態、記憶、フィードバックをまとめるシステムパターンです。

## まず single-Agent loop を見る

![Agent 基礎位置づけブリッジ図](/img/course/ch09-basics-position-bridge-ja.webp)

![Agent 基礎章の学習順序図](/img/course/ch09-basics-chapter-flow-ja.webp)

![Single-Agent 実行ループ図](/img/course/ch09-basics-execution-loop-ja.webp)

普通の chatbot は 1 回答えます。workflow は固定手順を進みます。Agent は plan、act、observe、state update を行い、goal が終わっていなければ続けます。

## 小さな Agent state loop を動かす

このスクリプトはまだモデルを呼びません。Agent を debug できるようにする最小 state を示します。

```python
goal = "summarize RAG citation rules"
state = {"steps": [], "done": False}

for action in ["plan", "search_docs", "summarize"]:
    state["steps"].append(action)

state["done"] = True

print("goal:", goal)
print("steps:", " -> ".join(state["steps"]))
print("done:", state["done"])
```

期待される出力：

```text
goal: summarize RAG citation rules
steps: plan -> search_docs -> summarize
done: True
```

デモが目標、状態、行動、観察、停止条件を示せないなら、まず LLM アプリと呼び、Agent とは呼ばないほうが正確です。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Agent とは何か | chatbot、workflow、RAG app、Agent を比較する |
| 2 | 発展史 | なぜ LLM が Agent systems を再び注目させたか理解する |
| 3 | 能力レベル | 回答、検索、ツール利用、計画、記憶、協調を同じ段階表に置く |
| 4 | システムアーキテクチャ | 目標、状態、プランナー、ツール、記憶、観察、実行器を描く |
| 5 | RL から Agent への突破 | 行動、報酬、フィードバック、計画をつなげる |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
agent_boundary: how this differs from chatbot or fixed workflow
goal_state_action: goal, current state, next action, observation
architecture_parts: planner, tools, memory, guardrails, evaluator
failure_check: over-autonomy, vague goal, missing state, or no trace
next_action: build the smallest traceable single-agent loop
```

## 合格ライン

single-Agent loop を描き、multi-Agent collaboration の前に single-Agent stability が必要な理由を説明できれば、この章は合格です。

出口ミニプロジェクトは research assistant Agent trace です：1 つの goal、1 つの plan、少なくとも 1 つの tool decision、1 つの observation、1 つの stop condition、1 つの final answer を残します。
