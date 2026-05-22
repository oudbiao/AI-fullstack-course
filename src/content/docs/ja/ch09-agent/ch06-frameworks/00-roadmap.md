---
title: "9.6.1 フレームワークロードマップ：必要なときだけ選ぶ"
description: "Agent フレームワークの短い実践ロードマップ：LangGraph、LlamaIndex、CrewAI、AutoGen を比較し、状態、データ、役割、リスクに基づいて選ぶ。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Agent フレームワークガイド, LangGraph, LlamaIndex, CrewAI, AutoGen"
---

# 9.6.1 フレームワークロードマップ：必要なときだけ選ぶ

フレームワークは Agent を自動的に賢くしません。タスクが十分に複雑になったとき、状態、ツール、ワークフロー、メモリ、ログ、協調を整理するための抽象化です。

## まず選択マップを見る

![Agent フレームワーク位置図](/img/course/ch09-frameworks-position-map-ja.webp)

![Agent フレームワーク選択図](/img/course/ch09-framework-selection-map-ja.webp)

![Agent フレームワーク選択判断図](/img/course/ch09-framework-selection-decision-map-ja.webp)

タスクが 3 つの固定ステップだけなら、普通の Python 関数のほうが良いことがあります。状態、分岐、復旧、データ接続、役割協調が管理しづらくなったらフレームワークを入れます。

## フレームワークルートチェックを動かす

人気だからという理由でフレームワークを選ぶ前に、このチェックを使います。

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

フレームワーク選択は README にトレードオフとして書きます。依存関係の中に隠さないでください。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | フレームワーク概観 | フレームワークが何を抽象化するか説明する |
| 2 | LangChain / LangGraph | 状態、ノード、エッジ、分岐、復旧をモデル化する |
| 3 | LlamaIndex | 文書、インデックス、検索、評価を接続する |
| 4 | CrewAI / AutoGen | 役割協調とマルチ Agent 会話を比較する |
| 5 | フレームワーク選定 | 判断表とフレームワークなしの基準実装を書く |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
問題の形：ワークフローグラフ、検索アプリ、役割チーム、または実験
フレームワーク選択：どの抽象化を追加し、何を隠すか
追跡記録：state、node、tool call、message、または run id
失敗確認：フレームワークの魔法が状態、再試行、または権限を隠す
判断: シングルエージェントのループが明確になってからフレームワークを選ぶ
```

## 合格ライン

同じ小さなタスクを普通の関数と 1 つのフレームワークで実装し、どちらがデバッグしやすいか、なぜかを説明できれば、この章は合格です。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、agent loop を goal、plan、tool call、observation、memory/state update、stop condition として説明します。
2. 証拠には、最終回答だけでなく、別の開発者が確認できる trace を残します。
3. tool schema、permission boundary、retry、evaluation case、人間レビューなど、安全性または信頼性の制御を1つ説明できれば十分です。

</details>
