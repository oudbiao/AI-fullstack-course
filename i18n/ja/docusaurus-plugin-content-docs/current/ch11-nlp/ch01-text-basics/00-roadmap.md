---
title: "11.1.1 テキスト基礎ロードマップ：Token、整形、表現"
sidebar_position: 0
description: "テキスト基礎章を短く実践的に進めるための地図です。生テキストを token に分け、数値化し、後続の NLP タスクへ渡す流れを先に見ます。"
keywords: [テキスト基礎ガイド, NLPガイド, テキスト表現]
---

# 11.1.1 テキスト基礎ロードマップ：Token、整形、表現

この章の目的は、NLP の入り口を作ることです。モデルは文字列をそのまま理解できません。まずテキストを小さな単位に分け、必要な形に整え、数値として扱えるようにします。

## 11.1.1.1 先に全体像を見る

![テキスト基礎章の進め方](/img/course/ch11-text-basics-chapter-flow-ja.png)

読む順番はシンプルです。

| 順番 | 学ぶこと | できるようになること |
|---|---|---|
| 1 | NLP タスクの種類 | 分類、抽出、生成の出力の違いを見分ける |
| 2 | 前処理 | 生テキストを安定した入力にする |
| 3 | 表現 | token を ID や特徴量に変える |

## 11.1.1.2 Token から始める

![テキストからタスクまでのパイプライン](/img/course/ch11-text-to-task-pipeline-ja.svg)

`token` は、モデルに渡すために切り出した最小単位です。英語なら単語に近いこともありますが、実際の tokenizer では単語の一部になることもあります。最初は「文章を小さな部品に分ける」と理解すれば十分です。

次のコードで、最小の token 化と ID 化を動かします。

```python
text = "RAG answers need citations"
tokens = text.lower().split()
vocab = {token: index for index, token in enumerate(sorted(set(tokens)))}
ids = [vocab[token] for token in tokens]

print("tokens:", tokens)
print("ids:", ids)
print("vocab_size:", len(vocab))
```

期待される出力：

```text
tokens: ['rag', 'answers', 'need', 'citations']
ids: [3, 0, 2, 1]
vocab_size: 4
```

操作のコツ：`lower()` は大文字小文字をそろえます。`split()` は空白で分けるだけなので、本格的な日本語、中国語、英語混在テキストでは専用 tokenizer を使います。

## 11.1.1.3 出力でタスクを見分ける

![NLP タスクと出力の対応](/img/course/ch11-nlp-task-output-map-ja.svg)

初心者が迷ったときは、モデル名より先に出力を見ます。

| タスク | 入力 | 出力 |
|---|---|---|
| 分類 | 文章 | 1 つのラベル |
| 系列ラベリング | token の列 | token ごとのラベル |
| 要約・翻訳 | 文章 | 新しい文章 |
| 抽出 | 文章 | 構造化フィールド |

## 11.1.1.4 通過条件

この章を終える前に、次を自分の言葉で説明できれば十分です。

| チェック | 合格ライン |
|---|---|
| token とは何か | テキストをモデルに渡すための小さな単位だと説明できる |
| 前処理の目的 | 掃除ではなく、入力を安定させる工程だと説明できる |
| 表現の目的 | テキストを計算できる形に変えることだと説明できる |
| 次章とのつながり | ID や特徴量から embedding へ進む流れを説明できる |
