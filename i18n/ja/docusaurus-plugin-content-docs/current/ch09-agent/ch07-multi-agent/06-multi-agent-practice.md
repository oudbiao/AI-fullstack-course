---
title: "9.7.7 実践：マルチ Agent 協調システム"
sidebar_position: 43
description: "タスク入力、役割分担、状態の流れから結果の集約まで、最小構成のマルチ Agent 協調システムを一通り作ります。"
keywords: [multi-agent project, planner, retriever, writer, reviewer, workflow, collaboration]
---

# 9.7.7 実践：マルチ Agent 協調システム

![マルチ Agent 協調実践の実行図](/img/course/ch09-multi-agent-collaboration-run-map-ja.webp)

:::tip この節の位置づけ
この節は本章のまとめとなるプロジェクトです。  
ここまでで、すでに次の内容を学んできました。

- アーキテクチャパターン
- 通信
- タスク分配
- 協調パターン
- 課題と解決方法

ここからは、これらを実際につなぎ合わせて、最小だけれど完成したマルチ Agent システムを作ります。
:::

## 学習目標

- 最小構成のマルチ Agent 協調ループを作る
- planner、retriever、writer、reviewer に役割を分けて動かす
- タスク状態が複数の役割の間でどう流れるかを理解する
- 単一 Agent システムと比べて、マルチ Agent では何が本当に増えるのかを理解する

---

## まずプロジェクトの目的を定義する

最小構成の研究型マルチ Agent システムを作ります。

ユーザー入力：

> 「返金ポリシーの重要な条件をまとめてください。」

システム内部の役割：

- Planner：タスクを分解する
- Retriever：資料を探す
- Writer：要約を書く
- Reviewer：結果を確認する

このタスクが向いている理由は、自然に分担しやすく、しかも各役割の責任がはっきりしているからです。

---

## まず資料ベースを用意する

```python
knowledge_base = {
    "返金ポリシー": "コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。",
    "証明書ポリシー": "すべての必修項目を完了し、テストに合格すると修了証を取得できます。",
    "学習順序": "まず Python、データ分析、機械学習を学び、その後に深層学習と大規模モデルの段階に進むのがおすすめです。"
}

print(knowledge_base)
```

想定出力：

```text
{'返金ポリシー': 'コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。', '証明書ポリシー': 'すべての必修項目を完了し、テストに合格すると修了証を取得できます。', '学習順序': 'まず Python、データ分析、機械学習を学び、その後に深層学習と大規模モデルの段階に進むのがおすすめです。'}
```

これが、このシステムが扱う最小限の知識ソースです。

---

## 4つの Agent を定義する

### Planner

```python
def planner_agent(user_query):
    if "返金" in user_query:
        return ["返金ポリシーを検索する", "重要条件を整理する", "要約を作成する", "出力をレビューする"]
    return ["関連資料を検索する", "要約を作成する", "出力をレビューする"]
```

### Retriever

```python
def retriever_agent(task):
    if "返金ポリシー" in task:
        return knowledge_base["返金ポリシー"]
    return "資料が見つかりませんでした"
```

### Writer

```python
def writer_agent(evidence):
    return f"要約：{evidence}"
```

### Reviewer

```python
def reviewer_agent(draft):
    if "7 日以内" in draft and "20%" in draft:
        return {"approved": True, "comment": "重要情報がそろっています"}
    return {"approved": False, "comment": "重要な条件が不足しています"}
```

---

## それらをつなぎ合わせる

### 最小構成のマルチ Agent 協調フロー

上の知識ベースと 4 つの Agent 関数を定義したあと、同じ Python ファイルまたは同じインタプリタセッションで続けて実行してください。

```python
def multi_agent_system(user_query):
    state = {
        "query": user_query,
        "plan": [],
        "evidence": None,
        "draft": None,
        "review": None
    }

    # 1. 企画
    state["plan"] = planner_agent(user_query)

    # 2. 検索
    state["evidence"] = retriever_agent(state["plan"][0])

    # 3. 執筆
    state["draft"] = writer_agent(state["evidence"])

    # 4. レビュー
    state["review"] = reviewer_agent(state["draft"])

    return state

result = multi_agent_system("返金ポリシーの重要な条件をまとめてください。")
for k, v in result.items():
    print(k, "->", v)
```

想定出力：

```text
query -> 返金ポリシーの重要な条件をまとめてください。
plan -> ['返金ポリシーを検索する', '重要条件を整理する', '要約を作成する', '出力をレビューする']
evidence -> コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。
draft -> 要約：コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。
review -> {'approved': True, 'comment': '重要情報がそろっています'}
```

### このコードから分かることは？

このコードから分かるのは、次の点です。

- マルチ Agent は、ただ関数が複数あるだけではない
- 重要なのは状態の流れ
- 各役割は、自分の担当部分だけを処理する

これが、本当に最小構成のマルチ Agent システムです。

---

## システムを実際のワークフローに近づける

### reviewer が通らなかったらどうする？

実際のシステムでは、review に通らなかったら、そのまま終了することはあまりありません。  
より自然なやり方は次のとおりです。

- comment を writer に戻す
- もう一度修正する

### 修正版を作る小さな例

同じファイルまたは同じセッションで続けて実行し、`multi_agent_system` と前の Agent 関数が定義済みであることを確認してください。

```python
def reviser_agent(draft, review):
    if review["approved"]:
        return draft
    return draft + " 補足: 返金には学習進捗が 20% 未満であることも必要です。"

state = multi_agent_system("返金ポリシーの重要な条件をまとめてください。")
final_output = reviser_agent(state["draft"], state["review"])

print("draft :", state["draft"])
print("review:", state["review"])
print("final :", final_output)
```

想定出力：

```text
draft : 要約：コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。
review: {'approved': True, 'comment': '重要情報がそろっています'}
final : 要約：コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。
```

このステップが大切なのは、次のことを表しているからです。

> マルチ Agent システムの価値は、分担だけでなく、役割同士で反復できる閉ループを作れることにあります。 

---

## より明確なタスクログを加える

### どうしてプロジェクトに trace が必要なのか？

もしシステムが間違えたら、少なくとも次のことが分かる必要があります。

- planner がどう分解したか
- retriever が何を見つけたか
- writer が何を書いたか
- reviewer がなぜ止めなかったか

### 最小構成の trace 版

同じファイルまたは同じセッションで続けて実行し、4 つの Agent 関数が定義済みであることを確認してください。

```python
def traced_multi_agent_system(user_query):
    trace = []

    plan = planner_agent(user_query)
    trace.append({"agent": "planner", "output": plan})

    evidence = retriever_agent(plan[0])
    trace.append({"agent": "retriever", "output": evidence})

    draft = writer_agent(evidence)
    trace.append({"agent": "writer", "output": draft})

    review = reviewer_agent(draft)
    trace.append({"agent": "reviewer", "output": review})

    return trace

for step in traced_multi_agent_system("返金ポリシーの重要な条件をまとめてください。"):
    print(step)
```

想定出力：

```text
{'agent': 'planner', 'output': ['返金ポリシーを検索する', '重要条件を整理する', '要約を作成する', '出力をレビューする']}
{'agent': 'retriever', 'output': 'コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。'}
{'agent': 'writer', 'output': '要約：コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請が可能です。'}
{'agent': 'reviewer', 'output': {'approved': True, 'comment': '重要情報がそろっています'}}
```

この trace は、後でデバッグしたり評価したりするときの重要な土台になります。

---

## なぜこのシステムは単一 Agent より学ぶ価値があるのか？

### 問題を分けて考えられるから

単一 Agent だと、たいてい一気に次のことをやります。

- タスクを理解する
- 検索する
- 要約する
- 自分で確認する

一方、マルチ Agent ではこれらを分けるので、次のことがしやすくなります。

- 各段階を観察する
- 途中の1段階だけ差し替える
- どの段階で失敗したかを見つける

### ただし、コストも複雑さも増える

だから、本当に大事なエンジニアリング判断は次です。

> マルチ Agent は、常により高度というわけではない

ではなく、

> このタスクは、「分解しやすい」「制御しやすい」という利点のために、追加の複雑さを払う価値があるか

です。 

---

## このプロジェクトをどう発展させるか？

さらに次の要素を追加できます。

1. より実際的な retriever
2. 複数タスクのルーティング
3. 非同期通信
4. 競合解決の仕組み
5. 失敗時の再試行

ここまで拡張すると、実際のマルチ Agent 製品システムにかなり近づきます。

---

## 初学者がよくやるミス

### すべての役割をほぼ同じにしてしまう

これでは最後に「名前が違うだけの同じ Agent が複数ある」だけになります。

### 共有状態や trace がない

一度エラーが起きると、原因を追いにくくなります。

### 見た目はにぎやかでも、実際には役割分担ができていない

これは多くのマルチ Agent デモでよくある問題です。

---

## まとめ

この節で最も大事なのは、4つの関数を書くことではなく、次を理解することです。

> **マルチ Agent プロジェクトの核心は、各役割が状態の流れに沿って異なる責任を持ち、最終的に説明可能で、反復可能なワークフローへ収束することです。**

これこそが、マルチ Agent が単一 Agent より本当に価値を持つ点です。

---

## 演習

1. このシステムに `fact_checker_agent` を追加して、数値条件を専用に確認してください。
2. `planner_agent` が「証明書ポリシー」に対しても異なる計画を出せるようにしてください。
3. 考えてみましょう：reviewer がずっと通らない場合、システムは修正の回数をどう制限すべきでしょうか？
4. 自分の言葉で説明してください。なぜ「マルチ Agent プロジェクトで本当に重要なのは役割数ではなく、状態の流れだ」と言えるのでしょうか？
