---
title: "9.7.1 Multi-Agent ロードマップ：Roles、Messages、Owner"
sidebar_position: 0
description: "Multi-Agent systems の短い実践ロードマップ：必要なときだけ roles を分け、message contract を定義し、coordination cost を抑え、final owner を残す。"
keywords: [Multi-Agent guide, collaborative systems, Agent communication, Agent coordination, multi-agent]
---

# 9.7.1 Multi-Agent ロードマップ：Roles、Messages、Owner

Multi-Agent は役割分担の仕組みであり、複数の chatbot を並べることではありません。role separation、parallel work、cross-checking、specialist collaboration の利益が coordination cost を上回るときだけ使います。

## まず collaboration cost を見る

![Multi-Agent collaboration message flow diagram](/img/course/multi-agent-message-flow-ja.png)

![Multi-Agent 章の学習順序図](/img/course/ch09-multi-agent-chapter-flow-ja.png)

![Multi-Agent collaboration and coordination map](/img/course/ch09-multi-agent-coordination-map-ja.png)

重要な問いは、分業の利益が messages、repeated context、conflicts、final merge のコストを上回るかです。

## Role boundary check を動かす

各 role には 1 つの責務と 1 つの output が必要です。final decision の owner を 1 人残します。

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

出力：

```text
agent_count: 3
researcher: collect evidence
editor: rewrite content
reviewer: check beginner clarity
final_owner: reviewer
```

2 つの roles が同じ output を出すなら merge します。final owner がいないと system は drift します。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Multi-Agent を使う時 | single Agent のほうがよい時を書く |
| 2 | Common patterns | supervisor-executor、pipeline、debate、expert committee を比較する |
| 3 | Communication | message format、shared state、handoff rule を定義する |
| 4 | Coordination | owner、queue、conflict rule、aggregation を追跡する |
| 5 | Practice and risks | cost、loops、duplicated work、role overreach を測る |

## 合格ライン

2〜3 Agents の demo が traceable inputs、outputs、handoffs、final ownership を持ち、single Agent より良い理由を説明できれば、この章は合格です。
