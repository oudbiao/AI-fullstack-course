---
title: "7.1.3 単語埋め込みと意味表現"
sidebar_position: 2
description: "小さな実験で token を dense vector に変換し、cosine similarity、ミニ意味検索、文脈表現を理解します。"
keywords: [embedding, semantic representation, cosine similarity, sentence embedding, contextual embedding, retrieval]
---

# 7.1.3 単語埋め込みと意味表現

![Embedding 意味空間図](/img/course/embedding-semantic-space-ja.webp)

:::tip 一文でいうと
Tokenizer はモデルに離散 ID を渡します。Embedding はその ID を vector に変え、モデルが意味を比較し、組み合わせ、層の間で運べるようにします。
:::

## まずメンタルモデルを作る

One-hot ID は単語を区別できますが、どの単語が関連しているかは表せません。Dense embedding は token を vector space に置きます。

```text
token id -> embedding table lookup -> dense vector
```

この空間では次のことができます。

- 近い vector は、関連した使われ方をしていることが多い。
- cosine similarity は方向の近さを測る。
- 文 vector は、token vector を pooling して作ることが多い。
- 文脈モデルでは、同じ token でも周囲の単語によって位置が変わる。

## ワンホット表現（One-Hot）から密ベクトル（Dense Vector）へ

![one-hot から dense embedding への意味空間図](/img/course/ch07-embedding-onehot-dense-map-ja.webp)

One-hot vector では、異なる単語はすべて同じように「違う」だけです。

```text
refund   -> [1, 0, 0, 0]
return   -> [0, 1, 0, 0]
password -> [0, 0, 1, 0]
banana   -> [0, 0, 0, 1]
```

Dense vector では、より役に立つ幾何が表せます。

```text
refund  and return   -> close
password and reset   -> close
refund  and password -> far
```

この幾何は手作業のルールではなく、データから学習されます。似た文脈に出る単語は、近い vector になりやすいです。

## 実験 1：単語の類似度を比べる

小さな embedding table を動かします。数字は学習用に手で作っていますが、操作は実際の vector retrieval と同じです。

```python
from math import sqrt

embeddings = {
    "refund": [0.90, 0.80, 0.10],
    "return": [0.88, 0.78, 0.12],
    "password": [0.10, 0.20, 0.95],
    "reset": [0.12, 0.18, 0.92],
    "order": [0.75, 0.70, 0.15],
    "banana": [0.05, 0.95, 0.10],
    "policy": [0.82, 0.74, 0.18],
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


print("refund vs return  :", round(cosine(embeddings["refund"], embeddings["return"]), 3))
print("refund vs password:", round(cosine(embeddings["refund"], embeddings["password"]), 3))
print("password vs reset :", round(cosine(embeddings["password"], embeddings["reset"]), 3))
```

期待される出力：

```text
refund vs return  : 1.0
refund vs password: 0.293
password vs reset : 1.0
```

読み方：

- cosine が高いとは、方向が近いという意味で、完全に同じ意味ではない。
- `refund` と `return` は、この玩具表ではカスタマーサポートの返金領域にある。
- `password` と `reset` はアカウント問題の領域にある。
- `refund` と `password` は意図が違うので遠い。

## 実験 2：ミニ意味検索器を作る

token vector を平均して文 vector を作り、query に対して 3 つの文書を順位付けします。

```python
def mean_embedding(tokens):
    vectors = [embeddings[token] for token in tokens if token in embeddings]
    dim = len(vectors[0])
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(dim)]


query = mean_embedding(["reset", "password"])
documents = {
    "A refund policy": ["refund", "policy"],
    "B password reset": ["password", "reset"],
    "C banana return": ["banana", "return"],
}

ranked = sorted(
    (
        (name, cosine(query, mean_embedding(tokens)))
        for name, tokens in documents.items()
    ),
    key=lambda item: item[1],
    reverse=True,
)

for name, score in ranked:
    print(f"{name}: {score:.3f}")
```

期待される出力：

```text
B password reset: 1.000
C banana return: 0.335
A refund policy: 0.333
```

これが vector retrieval の基本です。

```text
query text -> query vector -> compare with document vectors -> top-k results
```

実際の RAG ではより強い embedding model と vector database を使いますが、考え方は similarity ranking です。

## 平均 vector は便利だが限界がある

Mean pooling は分かりやすい一方で、重要な情報を落とします。

- 語順
- 否定
- 強調
- 長距離依存
- どの token が重要か

たとえば玩具検索器では、`reset password` と `password reset` は同じになります。直感を作るには便利ですが、推論が必要なタスクには不十分です。

## 文脈表現

![文脈表現が多義性を解く図](/img/course/ch07-contextual-embedding-sense-map-ja.webp)

Static embedding は通常 1 単語に 1 vector です。文脈モデルでは、周囲の単語によって vector が変わります。

```text
bank account -> bank は金融意味へ近づく
river bank   -> bank は地理意味へ近づく
```

小さなシミュレーションを動かします。

```python
base_bank = [0.50, 0.50, 0.50]
finance_context = [0.30, -0.10, 0.20]
river_context = [-0.20, 0.25, -0.10]

bank_in_finance = [a + b for a, b in zip(base_bank, finance_context)]
bank_in_river = [a + b for a, b in zip(base_bank, river_context)]

print("bank in finance:", [round(x, 2) for x in bank_in_finance])
print("bank in river  :", [round(x, 2) for x in bank_in_river])
```

期待される出力：

```text
bank in finance: [0.8, 0.4, 0.7]
bank in river  : [0.3, 0.75, 0.4]
```

![Embedding 実験結果図](/img/course/ch07-embedding-cosine-retrieval-context-result-map-ja.webp)

これは本物の Transformer ではありません。覚えるための小さな模型です。同じ token でも、文脈が混ざると別の表現になり得ます。

## プロジェクトでの使い道

| 用途 | embedding が提供するもの | 注意点 |
|---|---|---|
| RAG retrieval | 意味的に関連する chunk を探す | chunk や metadata が悪いと答えも悪くなる |
| FAQ clustering | 似た質問をまとめる | 近いことは重複と同じではない |
| Deduplication | 近似重複を探す | テンプレ文や言い換えでスコアが乱れる |
| Classification | テキストを特徴量にする | ラベル品質と calibration は別途必要 |
| Recommendation | user、item、クエリ を比較する | 人気バイアスが similarity を支配することがある |

## デバッグチェックリスト

- ライブラリが自動でしない場合、cosine 前に vector を normalize する。
- top-1 だけでなく top-k スコアを出す。差が小さいなら retrieval は不確実。
- false positive を見る。関連語が必ず正解とは限らない。
- 同じデータで static、sentence、文脈依存 embedding を比較する。
- 多言語プロジェクトでは、言語間ペアを実測してからモデルを信頼する。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
vectors: at least three text embeddings or toy vectors
similarity_check: closest pair and score
retrieval_result: top match for one query
limitation: averaging or similarity misses context/negation/order
next_use: this becomes retrieval evidence in Chapter 8
```

## 練習

1. `banana` を `password` に近づけると、検索はどう壊れるか。
2. 文書 `D recover account` を追加し、`recover` と `account` の vector を設計する。
3. クエリ `refund order` を作る。どの文書が 1 位になるべきか。
4. `doctor` と `hospital` が近くても同義語ではない理由を説明する。
5. RAG プロジェクトで embedding model が十分よいと示すには、どんな証拠を集めるか。

<details>
<summary>参考解答と解説</summary>

1. `banana` を `password` に近づけると、account recovery の query に果物関連文書が返るかもしれません。失敗は偶然ではなく、vector 空間の配置ミスです。
2. `recover` と `account` は、password や account support に近づけるべきです。commerce や fruit とは離します。追加文書は account recovery query に合うはずです。
3. embedding 空間が refund intent と order の意味を捉えていれば、`refund order` は返金/注文関連文書を 1 位にすべきです。
4. `doctor` と `hospital` は同じ domain によく現れるため近くなります。similarity は厳密な同義ではなく、topic 関連を表すことがあります。
5. 固定 query set、期待 top-k、retrieval score、既知の失敗例、latency、cost、言い換えに対する安定性を evidence として集めます。

</details>

## まとめ

Embedding は離散 token ID を幾何に変えます。

```text
identity -> vector -> distance -> retrieval / clustering / model input
```

本当に大事なのは式そのものではありません。意味が、比較でき、順位付けでき、ニューラルネットワークへ渡せる形になることです。
