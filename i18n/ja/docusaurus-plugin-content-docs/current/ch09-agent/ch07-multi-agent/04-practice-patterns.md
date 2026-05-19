---
title: "9.7.5 マルチ Agent 実践パターン"
sidebar_position: 41
description: "研究型、執筆型、開発型からレビュー型までの協力を通して、実際のタスクでよく使われるマルチ Agent の組み合わせ方を理解します。"
keywords: [multi-agent patterns, research team, writer-reviewer, dev team, agent collaboration]
---

# 9.7.5 マルチ Agent 実践パターン

:::tip この節の位置づけ
ここまでで学んだ内容は次のとおりです。

- マルチ Agent のアーキテクチャパターン
- Agent 間の通信
- タスクの分担と調整

この節では、これらをもう少し「実際のプロジェクト」に近い場面に当てはめていきます。

> **マルチ Agent は実際のタスクで、どう組み合わせるのが一般的なのか？**
:::

## 学習目標

- よく使われるマルチ Agent の実践パターンを理解する
- タスクの目的に合わせて、より適切な協力方法を選べるようになる
- 小さなマルチ Agent ワークフローの例を読めるようになる
- 「パターン」が「Agent をただ増やすこと」より重要な理由を理解する

---

## なぜ「実践パターン」を学ぶのか？

### 実際のシステムは、たいてい純粋な理論アーキテクチャではないから

多くのプロジェクトでは、次のような言い方になります。

- 「peer-to-peer のマルチ Agent システムがほしい」
  
それよりも、実際には次のように言うことが多いです。

- 「リサーチアシスタントのチームがほしい」
- 「執筆 + レビューのワークフローがほしい」
- 「コード開発チームがほしい」

つまり、実際のプロジェクトは、抽象的なアーキテクチャ名というより、「タスクの組織形態」に近いのです。

### では、実践パターンを学ぶ意味は何でしょうか？

それは、次のように移るのを助けてくれます。

- 抽象的な構造の理解

から

- 具体的なプロダクトへの実装

へ

---

## パターン1: 研究型協力

### 典型的な分担

- プランナー（Planner）：問題を分解する
- 調査担当（Researcher）：資料を調べる
- 統合担当（Synthesizer）：結果を統合する

### どんなタスクに向いているか？

- 背景調査をする
- 資料を集める
- 構造化されたレポートを出す

### 最小例

```python
def planner(query):
    return ["返金ポリシーを集める", "時間条件を整理する", "結論をまとめる"]

def researcher(task):
    docs = {
        "返金ポリシーを集める": "コース購入後 7 日以内かつ学習進捗が 20% 未満なら返金可能。",
        "時間条件を整理する": "重要な条件には、期間と学習進捗が含まれる。"
    }
    return docs.get(task, "資料が見つかりませんでした")

def synthesizer(items):
    return "結論：" + " ".join(items)

plan = planner("返金ポリシーは何ですか")
materials = [researcher(task) for task in plan[:-1]]
answer = synthesizer(materials)

print(plan)
print(materials)
print(answer)
```

想定出力：

```text
['返金ポリシーを集める', '時間条件を整理する', '結論をまとめる']
['コース購入後 7 日以内かつ学習進捗が 20% 未満なら返金可能。', '重要な条件には、期間と学習進捗が含まれる。']
結論：コース購入後 7 日以内かつ学習進捗が 20% 未満なら返金可能。 重要な条件には、期間と学習進捗が含まれる。
```

このパターンのポイントは次のとおりです。

> まず広く集めて、そのあとでまとめて絞り込む。

---

## パターン2: 執筆 + レビュー

### 最も定番で、実用的なパターンの一つ

分担は通常、次のようになります。

- 執筆担当（Writer）：まず下書きを書く
- レビュー担当（Reviewer）：問題点を確認する
- 修正担当（Reviser）：指摘に従って修正する

### なぜこのパターンは特によく使われるのか？

多くのタスクは、自然に次の流れに向いているからです。

- 生成
- 確認
- 再修正

たとえば次のようなものです。

- レポート作成
- 回答生成
- コードドキュメント

### 最小例

```python
def writer(topic):
    return f"下書き：{topic} の核心は 7 日以内なら返金可能という点です。"

def reviewer(draft):
    if "7 日以内" in draft:
        return "学習進捗の条件も補足するとよいです。"
    return "時間条件が抜けています。"

def reviser(draft, review):
    return draft + " " + review

draft = writer("返金ポリシー")
review = reviewer(draft)
final = reviser(draft, review)

print(draft)
print(review)
print(final)
```

想定出力：

```text
下書き：返金ポリシー の核心は 7 日以内なら返金可能という点です。
学習進捗の条件も補足するとよいです。
下書き：返金ポリシー の核心は 7 日以内なら返金可能という点です。 学習進捗の条件も補足するとよいです。
```

このパターンの最大の利点は次のとおりです。

> 「生成する力」と「誤りを直す力」を分けられることです。

---

## パターン3: 開発チームパターン

### よくある AI 開発チームの抽象化

たとえば、次のような役割です。

- PM / プランナー（PM / Planner）：要件を定義する
- 実装担当（Coder）：実装を書く
- レビュー担当（Reviewer）：コードをチェックする
- テスト担当（Tester）：結果を検証する

### なぜ AI coding の場面でよく使われるのか？

ソフトウェア開発には、もともとこうした役割分担があるからです。  
マルチ Agent は、それをプログラム化・自動化したものだと考えられます。

### 最小例

```python
workflow = [
    {"agent": "planner", "task": "実装する機能を定義する"},
    {"agent": "coder", "task": "実装コードを書く"},
    {"agent": "reviewer", "task": "ロジック上の問題を確認する"},
    {"agent": "tester", "task": "出力が期待どおりか検証する"}
]

for step in workflow:
    print(step["agent"], "->", step["task"])
```

想定出力：

```text
planner -> 実装する機能を定義する
coder -> 実装コードを書く
reviewer -> ロジック上の問題を確認する
tester -> 出力が期待どおりか検証する
```

このパターンで大切なのは、「役割名がかっこいいこと」ではなく、

> 各段階で、違う種類の問題を見つけられることです。

---

## パターン4: 二重検証 / 高リスクレビュー

### いつ必要になるのか？

タスクのリスクが高い場合です。たとえば、

- 法律に関する助言
- 医療の補助
- 金融判断

このような場合、1つの Agent だけで結論を出すのは危険なことがあります。

### よくある方法

- 1つの Agent が答えを生成する
- 別の Agent が事実確認をする
- さらに別の Agent がリスクとコンプライアンスを確認する

このパターンは少し遅くなりますが、より安定します。

---

## 小さなマルチ Agent ワークフローの例

```python
def planner(query):
    return ["retrieve", "write", "review"]

def retriever(query):
    return "検索結果：返金は時間と進捗の条件を満たす必要があります。"

def writer(material):
    return f"回答の下書き：{material}"

def reviewer(draft):
    if "進捗の条件" in draft:
        return {"approved": True, "comment": "情報はかなり揃っています"}
    return {"approved": False, "comment": "重要な条件が抜けています"}

query = "返金ポリシーは何ですか？"
steps = planner(query)
material = retriever(query)
draft = writer(material)
review = reviewer(draft)

print("steps  :", steps)
print("draft  :", draft)
print("review :", review)
```

想定出力：

```text
steps  : ['retrieve', 'write', 'review']
draft  : 回答の下書き：検索結果：返金は時間と進捗の条件を満たす必要があります。
review : {'approved': True, 'comment': '情報はかなり揃っています'}
```

![マルチ Agent workflow trace の結果図](/img/course/ch09-multi-agent-practice-trace-result-map-ja.webp)

:::tip 役割名だけでなく、引き継ぎを見る
この trace の価値は、それぞれの行が別の問いに答えている点です。planner が選んだ手順、retriever が集めた材料、reviewer がなぜ下書きを承認したかを順に読めます。
:::

このコードは小さいですが、実践パターンの本質をよく表しています。

- まず計画する
- 次に実行する
- そのあとレビューする

---

## どうやって適切な実践パターンを選ぶのか？

### タスクの重点が資料収集なら

優先するのは、

- 研究型協力

です。

### タスクの重点が内容の質なら

優先するのは、

- 執筆 + レビュー

です。

### タスクの重点がエンジニアリングの実装なら

優先するのは、

- 開発チームパターン

です。

### タスクのリスクが高いなら

優先するのは、

- 二重検証 / 高リスクレビュー

です。

本当に大事なのは、「どのパターンが一番かっこいいか」ではなく、

> 「今のタスクの失敗リスクと目的の構造に、どのパターンが一番合っているか？」

という点です。

---

## 初学者がよくつまずくポイント

### パターンを役割数と結びつけてしまう

「Agent が 3 個なら、必ずこのパターン」とは限りません。  
大事なのは数ではなく、責任の関係です。

### 見た目を複雑にするためにパターンを増やす

多くのタスクは、単一 Agent か 2 Agent で十分です。

### 評価基準が明確でない

「なぜこのパターンが別のパターンより良いのか」が分からないと、システム改善を進めにくくなります。

---

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

## まとめ

この節で最も大切なのは、「研究型」「開発型」といったラベルを覚えることではなく、次を理解することです。

> **マルチ Agent 実践パターンの価値は、抽象的な協力構造を実際のタスク目標に対応させることにある。**

タスクの形と協力パターンを結びつけられるようになると、マルチ Agent はやっと概念からプロダクトへと進みます。

---

## 練習

1. 自分がよく知っているタスクを 1 つ選び、それが研究型、執筆レビュー型、開発チーム型のどれに近いか考えてみましょう。
2. この節の小さなワークフローに `reviser` Agent を追加し、review をもとに draft を修正させてみましょう。
3. 高リスクなタスクでは、なぜ「生成 + 検証 + リスクレビュー」の組み合わせがより必要になるのか考えてみましょう。
4. 自分の言葉で説明してみましょう。なぜマルチ Agent では、役割の数より協力構造のほうが重要だと言えるのでしょうか。
