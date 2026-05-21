---
title: "4.1.1 線形代数ロードマップ：データはベクトル、バッチは行列"
sidebar_position: 0
description: "AI 向けの短い線形代数ロードマップです。ベクトル、行列、内積、固有値、変換を扱います。"
keywords: [線形代数ガイド, AI 数学ガイド, ベクトル, 行列, 固有値, PCA]
---

# 4.1.1 線形代数ロードマップ：データはベクトル、バッチは行列

線形代数は、AI がデータを表し、変換するための言語です。証明の暗記から始めず、まず各オブジェクトがコードで何をするかを見ます。

## まずマップを見る

![線形代数学習マップ](/img/course/ch04-linear-algebra-roadmap-vertical-ja.webp)

この小章の流れです。

![線形代数章フロー](/img/course/ch04-linear-algebra-chapter-flow-ja.webp)

| 概念 | AI での最初の意味 |
|---|---|
| ベクトル | 1つの対象を数値列で表す |
| 行列 | 複数のベクトルを積む、または変換を表す |
| 内積 | 対応する位置を掛けて合計する |
| 行列積 | 多くの内積を一度に行う |
| 固有値/固有ベクトル | 重要な方向。PCA の直感に使う |

## 最小ループを動かす

`linear_algebra_first_loop.py` を作り、`numpy` をインストールしてから実行します。

```python
import numpy as np

student = np.array([90, 85, 92])
students = np.array(
    [
        [90, 85, 92],
        [70, 88, 75],
        [95, 91, 89],
    ]
)
weights = np.array([0.4, 0.2, 0.4])

single_score = student @ weights
all_scores = students @ weights

print("student_vector:", student)
print("matrix_shape:", students.shape)
print("single_score:", round(single_score, 2))
print("all_scores:", all_scores.round(2))
```

出力：

```text
student_vector: [90 85 92]
matrix_shape: (3, 3)
single_score: 89.8
all_scores: [89.8 75.6 91.8]
```

`@` ではなく `*` を使うと、重み付きスコアではなく要素ごとの掛け算になります。最初にここを区別できるとかなり楽になります。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
数学対象：ベクトル、行列、固有値、基底、またはベクトル空間の概念
数値例：これを計算するために使う小さな数値または NumPy のスニペット
可視化または出力：形状、変換後の点、類似度スコア、固有方向、または射影
AI との関係: これが embeddings、バッチ、PCA、ニューラル層、または attention のどこに現れるか
期待される成果：計算と、それを AI の操作に結びつける1文
```

## この順番で学ぶ

| 順番 | 読む | まず見ること |
|---|---|---|
| 1 | [4.1.2 ベクトル](./01-vectors.md) | 対象 -> ベクトル、長さ、内積、コサイン類似度 |
| 2 | [4.1.3 行列](./02-matrices.md) | バッチデータ、行列積、`X @ W + b` |
| 3 | [4.1.4 固有値と固有ベクトル](./03-eigenvalues.md) | 特別な方向、PCA の直感 |
| 4 | [4.1.5 ベクトル空間](./04-vector-spaces.md) | 基底、次元、線形変換 |

## 合格ライン

1つのサンプルがベクトル、バッチが行列である理由、`@` が何をするか、そして RAG 類似度、PCA、ニューラルネットワーク層に再登場する理由を説明できれば合格です。


<details>
<summary>確認の考え方と解説</summary>

- 線形代数ルートを通過できる目安は、`X @ W` を shape の操作としても、内積のバッチとしても読めることです。
- 証拠として、ベクトル類似度の例、行列変換の例、PCA または固有ベクトルの図、SVD または rank チェックを 1 つずつ残します。
- 大事なのは記号の美しさではなく、方向、長さ、次元、冗長性がどう変わったかを説明できることです。

</details>
