---
title: "9.1.1 Agent 基礎ロードマップ：Goal、State、Action"
sidebar_position: 0
description: "Agent 基礎の短い実践ロードマップ：Agent、chatbot、workflow を区別し、最小の goal-state-action loop を作る。"
keywords: [Agent guide, intelligent agent guide, Agent system architecture, tool calling, Agent loop]
---

# 9.1.1 Agent 基礎ロードマップ：Goal、State、Action

Agent はモデル名ではありません。Goal に向かって、model、tools、state、memory、feedback をまとめる system pattern です。

## まず single-Agent loop を見る

![Agent 基礎位置づけブリッジ図](/img/course/ch09-basics-position-bridge-ja.png)

![Agent 基礎章の学習順序図](/img/course/ch09-basics-chapter-flow-ja.png)

![Single-Agent 実行ループ図](/img/course/ch09-basics-execution-loop-ja.png)

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

出力：

```text
goal: summarize RAG citation rules
steps: plan -> search_docs -> summarize
done: True
```

demo が goal、state、action、observation、stop condition を示せないなら、まず LLM app と呼び、Agent とは呼ばないほうが正確です。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Agent とは何か | chatbot、workflow、RAG app、Agent を比較する |
| 2 | 発展史 | なぜ LLM が Agent systems を再び注目させたか理解する |
| 3 | 能力レベル | answer、retrieve、tool use、plan、memory、collaboration を同じ ladder に置く |
| 4 | System architecture | goal、state、planner、tools、memory、observation、executor を描く |
| 5 | RL から Agent への突破 | action、reward、feedback、planning をつなげる |

## 合格ライン

single-Agent loop を描き、multi-Agent collaboration の前に single-Agent stability が必要な理由を説明できれば、この章は合格です。

出口ミニプロジェクトは research assistant Agent trace です：1 つの goal、1 つの plan、少なくとも 1 つの tool decision、1 つの observation、1 つの stop condition、1 つの final answer を残します。
