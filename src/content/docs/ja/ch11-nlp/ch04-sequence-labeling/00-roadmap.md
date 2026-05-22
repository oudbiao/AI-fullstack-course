---
title: "11.4.1 系列ラベリングロードマップ：Token ごとにラベルを付ける"
description: "系列ラベリング章を短く実践的に進めるための地図です。BIO ラベル、HMM/CRF、BiLSTM-CRF、NER プロジェクトの関係を先に見ます。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "シーケンスラベリングガイド, NER, BiLSTM-CRF"
---
分類は文全体に 1 つのラベルを付けます。系列ラベリングは、文の中の各 token にラベルを付けます。NER（Named Entity Recognition、固有表現認識）は代表例です。

## 先に全体像を見る

![系列ラベリング章の進め方](/img/course/ch11-sequence-labeling-chapter-flow-ja.webp)

![HMM と CRF から系列ラベリングを見る](/img/course/ch11-hmm-crf-sequence-history-map-ja.webp)

![BiLSTM-CRF のラベル経路](/img/course/ch11-bilstm-crf-label-path-map-ja.webp)

重要な出力は文全体のラベルではなく、`B-PER`、`I-PER`、`O` のように token ごとにそろったタグ列です。

## BIO ラベルを手で見る

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

## 学ぶ順番

| ステップ | 読む内容 | 実践で作るもの |
|---|---|---|
| 1 | NER と BIO | token 単位ラベルとエンティティ span を作る |
| 2 | HMM / CRF の歴史 | 系列制約とラベル遷移を理解する |
| 3 | BiLSTM-CRF | 文脈特徴と正しいラベル経路をつなげる |
| 4 | プロジェクト実践 | precision、recall、F1、境界エラーを評価する |

## 通過条件

token/tag の対応を確認でき、境界ミスまたは不正なタグ遷移を1つ説明できれば、この章は通過です。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
スキーマ: エンティティ型、BIO タグ、またはシーケンスラベル規則
予測：トークン単位のラベルと抽出スパン
指標：エンティティの precision/recall/F1 と境界ケース
失敗確認: span 境界、入れ子のエンティティ、未知語、または不一致なアノテーション
期待される成果：少なくとも1つの miss がある、gold と predicted の span 表
```
