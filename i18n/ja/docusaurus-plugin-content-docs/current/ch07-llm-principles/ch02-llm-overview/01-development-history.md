---
title: "7.2.2 大規模モデルの発展史"
sidebar_position: 5
description: "AI の 15 段階タイムラインと実行できる bigram 実験で、言語モデルが rules、statistics から Transformer、instruction alignment、RAG、Agent へ進んだ流れを理解します。"
keywords: [LLM history, language model, n-gram, RNN, Transformer, GPT, RLHF, Agent]
---

# 7.2.2 大規模モデルの発展史

![AI 15 段階発展マップ](/img/course/appendix-ai-15-stage-history-map-ja.webp)

:::tip 年表暗記ではなく地図として読む
日付を暗記する必要はありません。次の 1 本の流れを押さえます。

```text
rules -> statistics -> neural representations -> attention -> scale -> alignment -> tools
```

大規模言語モデルは突然生まれたものではなく、この長い変化の結果です。
:::

## 15 段階の全体像

| 段階 | 何が変わったか | LLM との関係 |
|---|---|---|
| 1. Turing question | 機械知能が具体的な問いになった | 言語が知能の重要なテストになった |
| 2. Dartmouth AI | AI が研究分野になった | 初期は symbolic reasoning が中心 |
| 3. Perceptron | 学習できる neural model が登場 | trainable model の第一波 |
| 4. Expert systems | rules が狭い領域で広がった | 価値と保守コストを同時に示した |
| 5. Backpropagation | 多層 neural nets が訓練可能になった | deep learning の基礎 |
| 6. LeNet | neural nets が実タスクで機能した | representation learning の実例 |
| 7. Statistical ML | data-driven methods が多くの手書き rules を超えた | NLP が corpus evidence へ移った |
| 8. ImageNet / AlexNet | deep learning が scale で勝った | data、compute、architecture が重要になった |
| 9. ResNet | 非常に深い networks が訓練しやすくなった | scale がより安定した |
| 10. RNN / LSTM | sequences が neural に扱われた | language modeling が n-gram を超えた |
| 11. Attention | 関連位置に focus できるようになった | long コンテキスト bottleneck を緩和 |
| 12. Transformer | attention が主アーキテクチャになった | parallel training と scaling が進んだ |
| 13. BERT / GPT | pretraining が shared foundation になった | 1 つの model を多タスクへ転用 |
| 14. RLHF / ChatGPT | behavior が instruction に合わせられた | capability が product behavior になった |
| 15. RAG / Agent | models が knowledge と tools を使い始めた | LLM が application system になった |

ここから言語モデルの主線に絞ります。

## 5 つの言語モデル時代

| 時代 | 中心アイデア | 主な限界 |
|---|---|---|
| Rule-based systems | 人間が language rules を書く | 壊れやすく、保守が高コスト |
| Statistical language models | 頻度から次の単語を予測 | sparse data、short コンテキスト |
| Neural sequence models | vector と recurrent state を学ぶ | 長距離依存が難しく、訓練が遅い |
| Transformers | token 同士が直接 attention する | compute と data cost が高い |
| LLM + alignment | 大規模 pretraining 後に behavior を調整 | hallucination、安全、コスト、評価 |

主線は context です。各世代は、手書き仮定を減らしながら、より多くの context を使おうとしてきました。

## 実験：Bigram 言語モデルを作る

この小さな `n-gram` model は、現在の単語から次の単語を予測します。強くはありませんが、neural LM 以前の統計的な発想が見えます。

```python
from collections import Counter, defaultdict

corpus = [
    "I like learning AI",
    "I like learning Python",
    "You like learning NLP",
    "I like doing projects",
]

next_word_counter = defaultdict(Counter)

for sentence in corpus:
    tokens = sentence.split()
    for current_word, next_word in zip(tokens[:-1], tokens[1:]):
        next_word_counter[current_word][next_word] += 1


def suggest_next(word):
    candidates = next_word_counter[word]
    return candidates.most_common() if candidates else []


print("Common words after I       :", suggest_next("I"))
print("Common words after like    :", suggest_next("like"))
print("Common words after learning:", suggest_next("learning"))
```

期待される出力：

```text
Common words after I       : [('like', 3)]
Common words after like    : [('learning', 3), ('doing', 1)]
Common words after learning: [('AI', 1), ('Python', 1), ('NLP', 1)]
```

![Bigram 補完の実行結果図](/img/course/ch07-bigram-autocomplete-result-map-ja.webp)

これは autocomplete に少し似ています。しかし限界も明確です。

- 1 つ前の単語しか見ない。
- rare な組み合わせは統計が弱い。
- 文の意味表現を持たない。

## なぜニューラルモデル（Neural Models）が重要だったのか

Neural language models は、単なるカウントを learned representations に置き換えました。

```text
word id -> vector -> context state -> prediction
```

Word2Vec、GloVe、RNN、LSTM、GRU により、language modeling は柔軟になりました。similarity と長めの context を学べるようになりましたが、逐次的に読むため訓練は遅く、長距離記憶も不安定でした。

## なぜ Transformer が転換点だったのか

RNN は主に一歩ずつ読みます。Transformer では、token が attention を通して他の token と直接比較できます。

```text
current token -> attends to relevant tokens -> updated representation
```

これにより 3 つの変化が起きました。

- 訓練をより並列化しやすい。
- 長距離関係を扱いやすい。
- parameters、data、compute の scale が効きやすい。

だから BERT、GPT、T5、後の LLM は Transformer family tree に属します。

## なぜ Scale だけでは足りないのか

大規模 pretraining は広い capability を作りましたが、product behavior には別の層が必要でした。

| 必要なこと | 技術 |
|---|---|
| 指示に従う | instruction tuning |
| 役に立つ回答を好む | preference learning / RLHF |
| 最新または private knowledge を使う | RAG |
| 行動する | tool calling / Agent loop |
| unsafe behavior を減らす | safety evaluation and guardrails |

現代 LLM で重要な区別はこれです。

```text
model capability != model behavior
```

モデルは強力でも、ポリシーに従わない、証拠を引用しない、安全に行動しないことがあります。

## 何を覚えるか

大規模モデルは NLP の歴史に属しますが、今は狭い NLP を超えています。同じ architecture と training idea が、text、image、speech、code、video、multimodal QA、RAG、Agent に広がっています。

実務的な要点は次の通りです。

- rules は control を与えたが coverage が弱い。
- statistics は data evidence を与えたが コンテキスト が短い。
- neural representations は semantic space を作った。
- Transformer は scale を実用化した。
- alignment、RAG、tool calling が model を system に変えた。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
timeline: n-gram -> neural LM -> Transformer -> scaling -> instruction/alignment
turning_point: what Transformer changed about context mixing
scale_note: data and compute changed capability but not reliability alone
bigram_lab: one output sample and its limitation
memory_hook: history is a sequence of solved bottlenecks
```

## 練習

1. bigram corpus に 2 文を追加し、suggestion がどう変わるか見る。
2. bigram model が長い instruction に弱い理由を説明する。
3. Transformer が RNN より並列訓練しやすい理由を説明する。
4. model に capability はあるが、alignment または RAG が必要な例を 1 つ挙げる。
5. 15 段階のどれか 1 つを選び、それが今日の LLM application にどう残っているか説明する。
