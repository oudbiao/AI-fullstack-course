---
title: "7.1.1 NLP 速習ロードマップ：テキストから token、ベクトルへ"
description: "短い NLP 速習ロードマップです。トークナイズ、埋め込み、事前学習済みモデル、Hugging Face、token ラボを扱います。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "NLP 速習, tokenizer, embedding, pretrained model, Hugging Face"
---
LLM を理解しやすくするには、まずテキストがモデルの処理できる形へ変わる流れを見ます。text -> tokens -> IDs -> vectors -> model output です。

## まず流れを見る

![NLP 速習章フローチャート](/img/course/ch07-nlp-crash-chapter-flow-ja.webp)

| 用語 | 最初の意味 |
|---|---|
| token | モデルが使うテキストの一部 |
| tokenizer | テキストを分け、ID に対応させる道具 |
| embedding | token やテキストの密なベクトル |
| pretrained model | 広いテキストで先に学習されたモデル |
| Hugging Face | モデル、データセット、ツールのエコシステム |

## 小さな token ラボを動かす

```python
text = "RAG retrieves evidence before answering"
tokens = text.lower().split()
vocab = {token: index for index, token in enumerate(sorted(set(tokens)))}
ids = [vocab[token] for token in tokens]

print("tokens:", tokens)
print("ids:", ids)
print("unique_tokens:", len(vocab))
```

期待される出力：

```text
tokens: ['rag', 'retrieves', 'evidence', 'before', 'answering']
ids: [3, 4, 2, 1, 0]
unique_tokens: 5
```

本物の tokenizer はもっと賢いですが、主な考え方は同じです。テキストは安定した部品と ID になってから、ベクトルやモデルへ進みます。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [7.1.2 Tokenizer](/ja/ch07-llm-principles/ch01-nlp-crash/01-tokenizer/) | text -> tokens -> IDs |
| 2 | [7.1.3 Embeddings](/ja/ch07-llm-principles/ch01-nlp-crash/02-embeddings/) | token/text -> vectors |
| 3 | [7.1.4 事前学習済みモデル](/ja/ch07-llm-principles/ch01-nlp-crash/03-pretrained-models/) | モデル能力をロードして再利用する |
| 4 | [7.1.5 Hugging Face クイックスタート](/ja/ch07-llm-principles/ch01-nlp-crash/04-huggingface-quickstart/) | pipeline、model card、ローカル実行 |
| 5 | [7.1.6 Tokenizer と Embedding ラボ](/ja/ch07-llm-principles/ch01-nlp-crash/05-tokenizer-embedding-lab/) | token とベクトルを確認する |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
テキストの流れ：raw text -> tokens -> ids -> embeddings
トークンリスク：長い入力は context またはコスト上限に達する可能性がある
埋め込みの用途：類似度は検索を支援できるが、推論ではない
モデルの橋渡し：事前学習モデル = 共通の基盤 + タスクの振る舞い
次の行動：Prompt 作業の前に tokenizer と embedding の演習を実行する
```

## 合格ライン

生テキストに tokenization が必要な理由、embedding がベクトルである理由、事前学習済みモデルをゼロからではなく再利用する理由を説明できれば合格です。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、token、context、attention、prompt、生成挙動が1回の request-response path でどうつながるかを説明します。
2. 証拠には、再現できる prompt または structured-output test を1つ残し、出力が通った理由または失敗した理由を書きます。
3. prompt 設計、RAG、fine-tuning、alignment を切り分け、観察した問題を直す最も軽い方法を選べれば十分です。

</details>
