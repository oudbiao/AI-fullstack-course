---
title: "9.4.1 Memory ロードマップ：Write、Retrieve、Forget"
sidebar_position: 0
description: "Agent memory の短い実践ロードマップ：何を覚えるか決め、適切な context を取り出し、古い facts を更新し、memory pollution を避ける。"
keywords: [memory systems overview, Agent memory, short-term memory, long-term memory, episodic memory]
---

# 9.4.1 Memory ロードマップ：Write、Retrieve、Forget

Memory は Agent を人間らしく見せるためではありません。task を助けるためにあります：同じ質問を減らし、有用 context を保ち、経験を再利用し、古い情報や privacy leak を避けます。

## まず memory loop を見る

![Agent memory system の階層図](/img/course/agent-memory-system-ja.webp)

![Agent memory systems 章の学習順序図](/img/course/ch09-memory-chapter-flow-ja.webp)

![Agent memory writing and retrieval の閉ループ図](/img/course/ch09-memory-write-retrieve-loop-ja.webp)

重要なのは「全部保存」ではありません。何を保存し、いつ retrieve し、いつ update し、いつ forget するかです。

## Memory write filter を動かす

長期 memory にするべきなのは、安定した preferences と reusable facts です。

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

出力：

```text
saved: ['prefers short examples', 'project uses Python']
count: 2
```

memory が useful、current、permitted、retrievable でないなら、Agent を助けるより邪魔することがあります。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Memory overview | context window、short-term memory、long-term memory を区別する |
| 2 | Short-term memory | 複数ターンの current task state を追跡する |
| 3 | Long-term memory | 安定した preferences、facts、project background を保存する |
| 4 | Episodic and procedural memory | what happened と how to do it next time を分ける |
| 5 | Memory engineering | write、retrieve、update、expire、delete rules を設計する |

## 合格ライン

「たくさん覚える」ことが「良い性能」と同じではない理由を説明できれば、この章は合格です。

出口ミニプロジェクトは learning-planning assistant memory rules です：何を保存し、何を確認し、何を temporary にし、何を delete するかを決めます。
