---
title: "9.7.1 マルチ Agent ロードマップ：役割、メッセージ、責任者"
sidebar_position: 0
description: "マルチ Agent システムの短い実践ロードマップ：必要なときだけ役割を分け、メッセージ契約を定義し、調整コストを抑え、最終責任者を残す。"
keywords: [Multi-Agent ガイド, 協調システム, Agent 通信, Agent 調整, multi-agent]
---

# 9.7.1 マルチ Agent ロードマップ：役割、メッセージ、責任者

マルチ Agent は役割分担の仕組みであり、複数のチャットボットを並べることではありません。役割分離、並列作業、相互チェック、専門家協調の利益が調整コストを上回るときだけ使います。

## まず協調コストを見る

![マルチ Agent 協調メッセージフロー図](/img/course/multi-agent-message-flow-ja.webp)

![マルチ Agent 章の学習順序図](/img/course/ch09-multi-agent-chapter-flow-ja.webp)

![マルチ Agent 協調と調整の図](/img/course/ch09-multi-agent-coordination-map-ja.webp)

重要な問いは、分業の利益がメッセージ、重複コンテキスト、衝突、最終統合のコストを上回るかです。

## 役割境界チェックを動かす

各役割には 1 つの責務と 1 つの出力が必要です。最終判断の責任者を 1 人残します。

```python
agents = {
    "researcher": "collect evidence",
    "editor": "rewrite content",
    "reviewer": "check beginner clarity",
}

final_owner = "reviewer"

print("agent_count:", len(agents))
for name, job in agents.items():
    print(f"{name}: {job}")
print("final_owner:", final_owner)
```

期待される出力：

```text
agent_count: 3
researcher: collect evidence
editor: rewrite content
reviewer: check beginner clarity
final_owner: reviewer
```

2 つの役割が同じ出力を出すなら統合します。最終責任者がいないとシステムは迷走します。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | マルチ Agent を使う時 | 単一 Agent のほうがよい時を書く |
| 2 | よくあるパターン | supervisor-executor、pipeline、debate、expert committee を比較する |
| 3 | コミュニケーション | メッセージ形式、共有状態、交接ルールを定義する |
| 4 | 調整 | 責任者、キュー、衝突ルール、集約を追跡する |
| 5 | 実践とリスク | コスト、ループ、重複作業、役割越権を測る |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

## 合格ライン

2〜3 体の Agent のデモが追跡可能な入力、出力、交接、最終責任を持ち、単一 Agent より良い理由を説明できれば、この章は合格です。

<details>
<summary>参考解答と解説</summary>

1. 合格レベルの答えでは、agent loop を goal、plan、tool call、observation、memory/state update、stop condition として説明します。
2. 証拠には、最終回答だけでなく、別の開発者が確認できる trace を残します。
3. tool schema、permission boundary、retry、evaluation case、人間レビューなど、安全性または信頼性の制御を1つ説明できれば十分です。

</details>
