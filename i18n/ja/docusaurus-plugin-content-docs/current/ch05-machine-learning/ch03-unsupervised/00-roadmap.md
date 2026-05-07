---
title: "5.3.1 教師なし学習ロードマップ：ラベルなしで構造を見つける"
sidebar_position: 6
description: "短い教師なし学習ロードマップです。クラスタリング、次元削減、異常検知、解釈の証拠を扱います。"
keywords: [教師なし学習ガイド, クラスタリング, 次元削減, 異常検知]
---

# 5.3.1 教師なし学習ロードマップ：ラベルなしで構造を見つける

教師なし学習は、データにラベルがないところから始まります。モデルは最終的な真実を教えるのではなく、あり得る構造を見つける手助けをします。

## まず構造マップを見る

![教師なし学習ロードマップ](/img/course/unsupervised-learning-roadmap-ja.png)

![教師なし学習章フロー](/img/course/ch05-unsupervised-chapter-flow-ja.png)

| やりたいこと | まず使うもの |
|---|---|
| 自然なグループを見つける | クラスタリング |
| 高次元データを圧縮する | 次元削減 |
| 普通ではない点を見つける | 異常検知 |

重要なのは「ラベルが正しいか」ではなく、「この構造に証拠と意味があるか」です。

## クラスタリング baseline を1つ動かす

`unsupervised_first_loop.py` を作り、`scikit-learn` をインストールしてから実行します。

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

X, _ = make_blobs(n_samples=30, centers=3, random_state=7, cluster_std=0.8)

model = KMeans(n_clusters=3, random_state=7, n_init="auto")
labels = model.fit_predict(X)

print("cluster_count:", len(set(labels)))
print("first_five_labels:", labels[:5].tolist())
print("inertia:", round(model.inertia_, 2))
```

出力：

```text
cluster_count: 3
first_five_labels: [2, 0, 0, 1, 0]
inertia: 43.44
```

クラスタリングが返すのはグループ番号であり、人間にとっての意味ではありません。グラフ、特徴量の要約、ドメイン解釈が必要です。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [5.3.2 クラスタリング](./01-clustering.md) | K-Means、クラスタ解釈、悪いクラスタ数選択 |
| 2 | [5.3.3 次元削減](./02-dimensionality-reduction.md) | PCA、可視化、圧縮 |
| 3 | [5.3.4 異常検知](./03-anomaly-detection.md) | 外れ値、しきい値、アラートの証拠 |

## 合格ライン

探している構造を説明し、教師なしモデルを1つ動かし、出力を絶対的な真実として扱わず慎重な解釈を書ければ合格です。
