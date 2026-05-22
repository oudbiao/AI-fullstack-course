---
title: "7.1.4 事前学習済み言語モデルの概観"
description: "共有 foundation の実験を通して、pretraining、transfer learning、task head、Prompt/RAG 適応、fine-tuning の選択を理解します。"
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "pretrained models, transfer learning, BERT, GPT, T5, foundation models, fine-tuning"
---
![転移学習の適応マップ](/img/course/ch06-transfer-learning-freeze-finetune-map-ja.webp)

:::tip[実務での判断]
事前学習済みモデルは、あなたの業務を自動で理解する魔法ではありません。再利用できる言語 foundation です。大事なのは、その foundation を最小コストで信頼できる形に適応させることです。
:::
## まずメンタルモデルを作る

Pretraining が一般化する前は、NLP タスクごとに別のモデルとデータ処理が必要になることが多くありました。現代の NLP は違う出発点から始まります。

```text
large general corpus -> pretrained foundation -> task adaptation -> product behavior
```

Foundation model は、すでに有用な言語パターンを学んでいます。多くのタスクでは次のどれかで適応します。

- prompt をより安定させる。
- RAG で不足知識を補う。
- 小さな task head を訓練する。
- LoRA や full update で fine-tuning する。
- 評価と guardrail で挙動を制御する。

## Pretraining が与えるもの

Pretraining は主に 3 つの実務的な資産を与えます。

| 資産 | 意味 | 例 |
|---|---|---|
| 再利用できる表現 | テキストが有用な hidden states に写る | classification、ranking、retrieval |
| 再利用できる生成能力 | 文章を続ける、変換する | chat、writing、code generation |
| 再利用できる言語 prior | 文法、よくあるパターン、高頻度事実 | 下流データが少なくて済む |

ただし、知識が最新であること、業務ポリシーが正しいこと、安全な挙動は保証されません。そこにはデータ、検索、評価、デプロイ時の制御が必要です。

## 実験：共有 foundation + 2 つの task head

この玩具例は本物の LLM を訓練しません。ただし構造は示します。1 つの共有 encoder と、2 つの異なる head です。

```python
from math import exp

word_vectors = {
    "refund": [0.9, 0.8, 0.1],
    "order": [0.8, 0.7, 0.2],
    "password": [0.1, 0.2, 0.9],
    "reset": [0.1, 0.1, 0.95],
    "great": [0.7, 0.2, 0.1],
    "bad": [0.2, 0.8, 0.1],
}


def encode(text):
    tokens = text.lower().split()
    valid = [word_vectors[token] for token in tokens if token in word_vectors]
    dim = len(valid[0])
    return [sum(vec[i] for vec in valid) / len(valid) for i in range(dim)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    exps = [exp(score) for score in scores]
    total = sum(exps)
    return [value / total for value in exps]


intent_head = {
    "refund_intent": [1.0, 0.9, 0.1],
    "password_intent": [0.1, 0.2, 1.0],
}

sentiment_head = {
    "positive": [1.0, 0.2, 0.0],
    "negative": [0.1, 1.0, 0.0],
}


def classify(vector, head):
    labels = list(head.keys())
    scores = [dot(vector, head[label]) for label in labels]
    probs = softmax(scores)
    best = max(zip(labels, probs), key=lambda item: item[1])
    rounded = dict(zip(labels, [round(prob, 3) for prob in probs]))
    return best, rounded


for text in ["refund order", "reset password"]:
    vector = encode(text)
    best, probs = classify(vector, intent_head)
    print("intent:", text, "->", best, probs)

for text in ["great refund", "bad refund"]:
    vector = encode(text)
    best, probs = classify(vector, sentiment_head)
    print("sentiment:", text, "->", best, probs)
```

期待される出力：

```text
intent: refund order -> ('refund_intent', 0.7604230019887309) {'refund_intent': 0.76, 'password_intent': 0.24}
intent: reset password -> ('password_intent', 0.654188113761243) {'refund_intent': 0.346, 'password_intent': 0.654}
sentiment: great refund -> ('positive', 0.5793242521487495) {'positive': 0.579, 'negative': 0.421}
sentiment: bad refund -> ('negative', 0.5361866202317948) {'positive': 0.464, 'negative': 0.536}
```

![共有 foundation と task head の結果図](/img/course/ch07-pretrained-shared-foundation-heads-result-map-ja.webp)

読み方：

- `encode()` が共有 foundation。
- `intent_head` と `sentiment_head` が task-specific adapter。
- foundation は再利用され、最後の判断層だけが変わる。
- 実際のモデルでは、手書き vector ではなく、百万から数千億規模の学習済み parameter が使われる。

## 主なモデルファミリー

| ファミリー | 典型的な情報の流れ | 得意 | 例 |
|---|---|---|---|
| エンコーダーのみ | 入力を双方向に読む | 分類、抽出、照合、embedding | BERT 系 |
| デコーダーのみ | 因果順序で次の token を予測 | チャット、補完、コード、ツール利用 | GPT/LLaMA/Qwen 系 |
| エンコーダー-デコーダー | 入力を読んでから出力を生成 | 翻訳、要約、構造化生成 | T5/BART 系 |

これは最初のフィルタであって、絶対ルールではありません。現代のシステムは retrieval、tools、serving constraints と組み合わせて設計されます。

## 適応方法を選ぶ

| 状況 | まず試すこと | 理由 |
|---|---|---|
| モデルがタスク形式をほぼ理解している | prompt 改善 | 反復が最速 |
| 答えが private / fresh knowledge に依存する | RAG | 重みを変えずに知識を更新できる |
| 安定した label や score が必要 | task head / classifier | 安く、評価しやすい |
| style や domain behavior を変えたい | LoRA / PEFT | 管理しやすいコストで挙動を変える |
| 高度に特殊で、データも強い | full fine-tuning | 最大の柔軟性。ただし高リスク高コスト |

これは信念ではなくエンジニアリング判断です。評価を通る最小の変更を選びます。

## よくある失敗モード

- **事前学習データのずれ：** 一般言語は学んでいても、あなたの業務ポリシーを知っているとは限らない。
- **知識の古さ：** 最近の事実を知らないことがある。
- **データ汚染：** 評価データやテストデータに似たものが訓練コーパスに入っていることがある。
- **過適応：** fine-tuning で一部は良くなり、別の能力が落ちることがある。
- **評価の抜け：** デモ prompt は良くても、境界ケースで失敗することがある。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
基盤：事前学習済みモデルがすでに知っていること
タスク見出し：タスク固有の部分がどこか
適応パス: プロンプト、特徴利用、ファインチューニング、またはPEFT
評価ケース：転移が成功した、または失敗したことを示す 1 例
リスク：事前学習済みの振る舞いは広範だが、自動的にタスクに整合するわけではない
```

## 練習

1. 実験に `topic_head` を追加し、`account_topic` と `commerce_topic` を分類する。
2. `bad` の vector を変えると sentiment confidence はどう変わるか。
3. private policy を持つ support bot なら、prompt、RAG、task head、fine-tuning のどれから始めるか。理由も書く。
4. 本番で pretrained model を信頼する前に、どの 2 つのチェックを行うか。
5. 「大きいモデル」と「今のタスクに合うモデル」が同じではない理由を説明する。

<details>
<summary>参考実装と解説</summary>

1. `topic_head` は pretrained representation を再利用し、`account_topic` や `commerce_topic` などの task label に写像します。head は task 固有で、foundation は再利用されます。
2. `bad` の vector を変えると、sentiment head が見る negative evidence の強さが変わります。移動先によって confidence は下がる、反転する、不安定になる可能性があります。
3. private policy では、多くの場合 RAG から始めます。知識が private で変化し、source citation も必要だからです。fine-tuning は後で安定した振る舞いや形式に使います。
4. 最低限、代表データでの task evaluation と sensitive case の failure review を行います。privacy、latency、cost、bias も本番では確認対象です。
5. 大きい model は広い capability を持つかもしれませんが、task fit は data、instruction、retrieval、evaluation、運用制約に左右されます。size は要素の 1 つです。

</details>

## まとめ

Pretraining はワークフローを変えます。

```text
do not relearn language every time -> reuse a foundation -> adapt with evidence
```

このパターンが見えると、Prompt、RAG、fine-tuning、alignment、Agent はすべて、同じ再利用可能な foundation を別の方法で制御する技術として理解できます。
