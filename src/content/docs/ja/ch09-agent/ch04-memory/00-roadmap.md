---
title: "9.4.1 メモリロードマップ：書き込み、検索、忘却"
description: "Agent メモリの短い実践ロードマップ：何を覚えるか決め、適切な文脈を取り出し、古い事実を更新し、メモリ汚染を避ける。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "memory systems overview, Agent memory, short-term memory, long-term memory, episodic memory"
---
メモリは Agent を人間らしく見せるためではありません。タスクを助けるためにあります：同じ質問を減らし、有用な文脈を保ち、経験を再利用し、古い情報やプライバシー漏えいを避けます。

## まずメモリループを見る

![Agent メモリシステムの階層図](/img/course/agent-memory-system-ja.webp)

![Agent メモリシステム章の学習順序図](/img/course/ch09-memory-chapter-flow-ja.webp)

![Agent メモリ書き込みと検索の閉ループ図](/img/course/ch09-memory-write-retrieve-loop-ja.webp)

重要なのは「全部保存」ではありません。何を保存し、いつ検索し、いつ更新し、いつ忘れるかです。

## メモリ書き込みフィルタを動かす

長期メモリにするべきなのは、安定した好みと再利用できる事実です。

```python
events = [
    {"type": "preference", "text": "prefers short examples"},
    {"type": "temporary", "text": "debugging one local error"},
    {"type": "fact", "text": "project uses Python"},
]

memory = []
for event in events:
    if event["type"] in {"preference", "fact"}:
        memory.append(event["text"])

print("saved:", memory)
print("count:", len(memory))
```

期待される出力：

```text
saved: ['prefers short examples', 'project uses Python']
count: 2
```

メモリが有用で、最新で、許可され、検索可能でなければ、Agent を助けるより邪魔することがあります。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | メモリ概要 | コンテキストウィンドウ、短期メモリ、長期メモリを区別する |
| 2 | 短期メモリ | 複数ターンの現在タスク状態を追跡する |
| 3 | 長期メモリ | 安定した好み、事実、プロジェクト背景を保存する |
| 4 | エピソード記憶と手続き記憶 | 何が起きたかと次にどうするかを分ける |
| 5 | メモリエンジニアリング | 書き込み、検索、更新、期限切れ、削除ルールを設計する |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
メモリ種別：短期、長期、エピソード記憶、または手続き記憶
書き込みルール：メモリが作成または更新されるとき
取得ルール：クエリ、関連性、鮮度、権限チェック
失敗確認: 古い記憶、プライバシー漏えい、矛盾、または過剰検索
クリーンアップ操作：要約、統合、期限切れ、削除、または確認を求める
```

## 合格ライン

「たくさん覚える」ことが「良い性能」と同じではない理由を説明できれば、この章は合格です。

出口ミニプロジェクトは learning-planning assistant memory rules です：何を保存し、何を確認し、何を temporary にし、何を delete するかを決めます。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、agent loop を goal、plan、tool call、observation、memory/state update、stop condition として説明します。
2. 証拠には、最終回答だけでなく、別の開発者が確認できる trace を残します。
3. tool schema、permission boundary、retry、evaluation case、人間レビューなど、安全性または信頼性の制御を1つ説明できれば十分です。

</details>
