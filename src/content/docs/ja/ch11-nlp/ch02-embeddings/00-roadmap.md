---
title: "11.2.1 表現学習ロードマップ：意味をベクトルで扱う"
description: "表現学習章を短く実践的に進めるための地図です。単語 ID から embedding、文脈依存表現、言語モデルへ進む流れを先に見ます。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "表現学習ガイド, 単語埋め込み, 文脈依存表現, 言語モデル"
---
前章では、テキストを token と ID にしました。この章では一歩進めて、「意味が近い言葉は、ベクトル空間でも近くなる」という考え方を扱います。

## 先に全体像を見る

![表現学習章の進め方](/img/course/ch11-embeddings-chapter-flow-ja.webp)

| 順番 | 学ぶこと | 役割 |
|---|---|---|
| 1 | 単語 embedding | 固定の意味ベクトルを作る |
| 2 | 文脈依存 embedding | 文によって意味が変わる単語を扱う |
| 3 | 言語モデル | 文脈全体から表現を学ぶ |

## ベクトルの近さを手で確かめる

![意味空間のイメージ](/img/course/embedding-semantic-space-ja.webp)

`embedding` は、token を数字の並びに変えたものです。数字そのものを暗記する必要はありません。大事なのは、似た意味の token が近い場所に置かれることです。

```python
vectors = {
    "cat": [1.0, 0.8],
    "dog": [0.9, 0.7],
    "car": [0.1, 0.2],
}

def dot(a, b):
    return sum(x * y for x, y in zip(a, b))

print("cat_dog:", round(dot(vectors["cat"], vectors["dog"]), 2))
print("cat_car:", round(dot(vectors["cat"], vectors["car"]), 2))
```

期待される出力：

```text
cat_dog: 1.46
cat_car: 0.26
```

ここでは簡単に内積を使っています。値が大きいほど、今回の小さな例では近い関係だと見なします。本格的な検索では cosine similarity なども使います。

## 固定表現と文脈依存表現を比べる

![固定 embedding と文脈依存 embedding の違い](/img/course/contextual-embedding-comparison-ja.webp)

同じ単語でも、文によって意味が変わります。たとえば `bank` は銀行にも川岸にもなります。固定 embedding は単語ごとに 1 つの表現を持ち、文脈依存 embedding は周囲の文から表現を変えます。

## 通過条件

| チェック | 合格ライン |
|---|---|
| embedding とは何か | token を意味を含むベクトルに変えるものだと説明できる |
| ID との違い | ID は区別、embedding は近さを扱えると説明できる |
| 文脈依存表現 | 同じ単語でも文によって表現が変わる理由を言える |
| 次章とのつながり | embedding を分類器の入力にできると説明できる |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
表現: BoW、TF-IDF、静的 embedding、文脈的 embedding、または言語モデルのスコア
比較：最も近いテキスト、類似度スコア、または次トークン/ログ確率形式の出力
解釈：この表現が何を捉え、何を捉え損ねるか
失敗確認: 多義性、ドメイン不一致、短文、トークン化、または意味のずれ
期待される成果: 少なくとも1つの意外な結果を含む小さな比較表
```
