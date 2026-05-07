---
title: "11.4.1 系列ラベリングロードマップ：Token ごとにラベルを付ける"
sidebar_position: 0
description: "系列ラベリング章を短く実践的に進めるための地図です。BIO ラベル、HMM/CRF、BiLSTM-CRF、NER プロジェクトの関係を先に見ます。"
keywords: [シーケンスラベリングガイド, NER, BiLSTM-CRF]
---

# 11.4.1 系列ラベリングロードマップ：Token ごとにラベルを付ける

分類は文全体に 1 つのラベルを付けます。系列ラベリングは、文の中の各 token にラベルを付けます。NER（Named Entity Recognition、固有表現認識）は代表例です。

## 先に全体像を見る

![系列ラベリング章の進め方](/img/course/ch11-sequence-labeling-chapter-flow-ja.png)

| 順番 | 学ぶこと | 目的 |
|---|---|---|
| 1 | BIO ラベル | エンティティの開始・継続・外側を表す |
| 2 | HMM / CRF | ラベル列の自然なつながりを扱う |
| 3 | BiLSTM-CRF | 文脈表現とラベル列制約を組み合わせる |

## BIO ラベルを手で見る

![HMM と CRF から系列ラベリングを見る](/img/course/ch11-hmm-crf-sequence-history-map-ja.png)

`B-PER` は人物名の開始、`I-PER` は人物名の続き、`O` は対象外を表します。まずは小さな文で、token とラベルの対応を確認します。

```python
tokens = ["Ada", "Lovelace", "wrote", "notes"]
tags = ["B-PER", "I-PER", "O", "O"]

for token, tag in zip(tokens, tags):
    print(token, tag)
```

期待される出力：

```text
Ada B-PER
Lovelace I-PER
wrote O
notes O
```

操作のコツ：系列ラベリングでは token 数と tag 数が一致している必要があります。ここがずれると、学習データとして使えません。

## BiLSTM-CRF の直感

![BiLSTM-CRF のラベル経路](/img/course/ch11-bilstm-crf-label-path-map-ja.png)

`BiLSTM` は前後の文脈を読み、`CRF` はラベル列として自然な並びを選びます。たとえば `I-PER` が文頭に突然出るより、`B-PER` の後に出るほうが自然です。

## 通過条件

| チェック | 合格ライン |
|---|---|
| 分類との違い | 文全体ではなく token ごとに予測すると説明できる |
| BIO | B、I、O の意味を言える |
| CRF | 1 token ずつではなくラベル列として整える役割を説明できる |
| プロジェクト化 | token、tag、評価指標をセットで準備できる |
