---
title: "6.5.2 Attention 機構"
description: "Q/K/V のスコア、softmax 重み、causal mask、PyTorch MultiheadAttention の shape を計算しながら Attention を学びます。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Attention, Self-Attention, QKV, Transformer, Multi-Head, Mask"
---

# 6.5.2 Attention 機構

:::tip[この節の位置づけ]
RNN は情報を一歩ずつ渡します。Attention は、1 つの token が他の token を直接見て、どれが重要かを判断できるようにします。これが Transformer の中心的な転換です。
:::
## 学習目標

- Attention が長距離依存に役立つ理由を説明できる。
- Query、Key、Value を検索の比喩で理解できる。
- scaled dot-product attention を手で計算できる。
- 未来を見ないための causal mask を適用できる。
- PyTorch の `nn.MultiheadAttention` の shape を読める。

---

## まず Q/K/V を見る

![Self-Attention QKV 構造図](/img/course/self-attention-qkv-ja.webp)

Attention は重み付き検索として読むと分かりやすいです。

```text
Q が質問する -> K と照合する -> softmax が重みにする -> V が内容を返す -> 重み付きで足し合わせる
```

検索の比喩：

![Attention QKV の図書館検索比喩図](/img/course/ch06-attention-qkv-library-analogy-map-ja.webp)

| 役割 | 直感 | Attention での意味 |
|---|---|---|
| Query `Q` | 何を探しているか | 現在の token の質問 |
| Key `K` | 各項目が何に合うか | スコア計算に使う索引 |
| Value `V` | 返すべき内容 | 実際に混ぜられる情報 |

一文でいうと：

```text
Q が K とスコアを作り、その重みで V を混ぜる。
```

## なぜ Attention が必要だったのか

古い系列モデルでは、遠い情報は多くの recurrent step を通るか、1 つの固定ベクトルに圧縮される必要がありました。Attention はその経路を短くします。

```text
現在の token -> すべての token に直接スコアを付ける -> 役立つ文脈を選ぶ
```

実務上の利点は 3 つあります。

- 長距離の token 同士を直接つなげられる。
- 一歩ずつ処理する RNN より並列学習しやすい。
- token-to-token の混合重み行列を観察できる。

## 実験 1：Attention を手で計算する

学習用に、ここでは `Q = K = V = X` とします。

```python
import numpy as np

X = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

Q = K = V = X

scores = Q @ K.T
scaled_scores = scores / np.sqrt(K.shape[1])


def softmax(row):
    e = np.exp(row - row.max())
    return e / e.sum()


weights = np.apply_along_axis(softmax, 1, scaled_scores)
output = weights @ V

print("attention_lab")
print("scores")
print(np.round(scores, 3))
print("weights")
print(np.round(weights, 3))
print("output")
print(np.round(output, 3))
```

期待される出力：

```text
attention_lab
scores
[[1. 0. 1.]
 [0. 1. 1.]
 [1. 1. 2.]]
weights
[[0.401 0.198 0.401]
 [0.198 0.401 0.401]
 [0.248 0.248 0.503]]
output
[[0.802 0.599]
 [0.599 0.802]
 [0.752 0.752]]
```

3 つの手順で読みます。

| 手順 | コード | 意味 |
|---|---|---|
| スコア | `Q @ K.T` | 各 token が各 token とどれくらい合うか |
| 正規化 | `softmax(...)` | スコアを合計 1 の重みに変える |
| 混合 | `weights @ V` | 重みに従って token の内容を組み合わせる |

## Lab 1B：Q/K/V は学習された視点であり、3 つのコピーではない

手計算の実験では、数学を見やすくするために `Q = K = V = X` としました。実際の Transformer では、通常 3 つの射影行列を学習します。

```text
Q = XW_q
K = XW_k
V = XW_v
```

つまり同じ token 表現を、3 つの視点で見ることになります。

- `Q`: この位置が何を探しているか。
- `K`: この位置がどんな一致手がかりを持つか。
- `V`: 選ばれたときに、この位置がどんな内容を渡すか。

小さな例を実行します。

```python
import numpy as np

X = np.array(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ]
)

W_q = np.array([[1.0, 0.5], [0.0, 1.0]])
W_k = np.array([[0.5, 1.0], [1.0, 0.0]])
W_v = np.array([[1.0, -0.5], [0.5, 1.0]])

Q = X @ W_q
K = X @ W_k
V = X @ W_v

scores = Q @ K.T / np.sqrt(Q.shape[1])


def softmax(row):
    e = np.exp(row - row.max())
    return e / e.sum()


weights = np.apply_along_axis(softmax, 1, scores)
output = weights @ V

print("projection_lab")
for name, value in [("Q", Q), ("K", K), ("V", V), ("weights", weights), ("output", output)]:
    print(name)
    print(np.round(value, 3))
```

期待される出力：

```text
projection_lab
Q
[[1.  0.5]
 [0.  1. ]
 [1.  1.5]]
K
[[0.5 1. ]
 [1.  0. ]
 [1.5 1. ]]
V
[[ 1.  -0.5]
 [ 0.5  1. ]
 [ 1.5  0.5]]
weights
[[0.248 0.248 0.503]
 [0.401 0.198 0.401]
 [0.284 0.14  0.576]]
output
[[1.128 0.376]
 [1.102 0.198]
 [1.218 0.286]]
```

証拠を読みます。

- `Q`、`K`、`V` は同じ `X` から来ていますが、今は異なる値になっています。
- Attention 重みは `Q` と `K` から計算されます。
- 最終出力が混ぜるのは元の `X` ではなく `V` です。

だから Q/K/V は単なる 3 つの変数名として暗記しない方がよいです。これらは**一致判定**と**内容の混合**を分ける、学習された 3 つの視点です。

## 残す証拠

attention trace を 1 つ残します。

```text
スコア規則: Q @ K.T / sqrt(d_k)
重みのルール：softmax はスコアを各行の和が 1 になる行に変換する
出力ルール：weights @ V がvalue vectorsを混合する
QKVのルール：Q/K が一致を決め、V が内容を運ぶ
マスク規則：ブロックされた位置にはほぼゼロのアテンションが与えられる
LLM への橋渡し：因果アテンションでは、生成は過去のトークンのみを使う
```

## なぜ `sqrt(d_k)` で割るのか

Transformer の式は次の形です。

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

ベクトルの次元が大きくなると、dot product の値も大きくなりやすくなります。大きすぎるスコアは softmax を尖らせ、1 つの token がほとんど全重みを取ってしまいます。`sqrt(d_k)` で割るとスコアが穏やかになり、学習が安定しやすくなります。

## Self-Attention

Self-attention では、`Q`、`K`、`V` がすべて同じ系列から来ます。つまり、各 token が同じ系列内のすべての token を見ることができます。

例：

```text
"Alex gave Sam the notebook because he trusted him."
```

「he」や「him」を理解するには、現在の token だけでなく他の token が必要です。Self-attention はそこへ直接つながる経路を作ります。

## 実験 2：Causal Mask

生成タスクでは未来の token を見てはいけません。causal mask は下三角だけを見えるようにします。

![Causal Mask が未来の覗き見を防ぐ図](/img/course/ch06-causal-mask-no-peeking-map-ja.webp)

```python
import numpy as np

scores = np.array(
    [
        [2.0, 1.0, 0.5],
        [1.2, 2.1, 0.7],
        [0.8, 1.3, 2.2],
    ]
)

mask = np.tril(np.ones_like(scores))
masked_scores = np.where(mask == 1, scores, -1e9)


def softmax(row):
    e = np.exp(row - row.max())
    return e / e.sum()


weights = np.apply_along_axis(softmax, 1, masked_scores)

print("mask_lab")
print(np.round(weights, 3))
```

期待される出力：

```text
mask_lab
[[1.    0.    0.   ]
 [0.289 0.711 0.   ]
 [0.149 0.246 0.605]]
```

読み方：

- 位置 1 は自分だけを見る。
- 位置 2 は位置 1 と 2 を見る。
- 位置 3 は位置 1、2、3 を見る。

未来の答えは見えません。

## Multi-Head Attention

1 つの attention head は、1 種類の関係を学ぶことがあります。multi-head attention は、複数の関係空間を並列に見る仕組みです。

head によって注目しやすいものは異なります。

- 近い位置のパターン。
- 主語と目的語の関係。
- 繰り返し出てくる語。
- 長距離の参照。

複数 head の結果は結合され、もう一度 1 つの表現へ射影されます。

## 実験 3：PyTorch `MultiheadAttention`

```python
import torch
from torch import nn

torch.manual_seed(42)

attention = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
tokens = torch.randn(1, 4, 8)
output, weights = attention(tokens, tokens, tokens)

print("mha_lab")
print("tokens:", tuple(tokens.shape))
print("output:", tuple(output.shape))
print("weights:", tuple(weights.shape))
print("row0_sum:", round(float(weights[0, 0].sum().detach()), 4))
```

期待される出力：

```text
mha_lab
tokens: (1, 4, 8)
output: (1, 4, 8)
weights: (1, 4, 4)
row0_sum: 1.0
```

shape の読み方：

| Tensor | Shape | 意味 |
|---|---|---|
| `tokens` | `[1, 4, 8]` | batch 1、4 token、embedding size 8 |
| `output` | `[1, 4, 8]` | 各 token が文脈を含んだ新しい表現になる |
| `weights` | `[1, 4, 4]` | 各 query token が 4 個の key token に重みを配る |

## Attention 重みは完全な説明ではない

Attention 重みは便利ですが、言い過ぎには注意します。

分かること：

```text
この layer/head では、この query がそれらの key 位置からより多くの value を混ぜた
```

自動的には証明できないこと：

```text
モデルの最終判断は、その token が原因だった
```

Attention 重みは調査とデバッグの道具として使います。完全な因果説明として扱わないでください。

## よくある間違い

| 間違い | 直し方 |
|---|---|
| Q/K/V を謎の変数として覚える | 質問 / 索引 / 内容として読む |
| shape の意味を追わない | `[batch, seq_len, embed_dim]` と attention `[batch, query, key]` を追う |
| 生成で mask を使わない | causal mask で未来 token を隠す |
| `softmax` を間違った次元にかける | key 位置の方向で正規化する |
| Attention を推論の魔法だと思う | スコア -> softmax -> 重み付き和として読む |

## 練習

1. 実験 1 の 3 番目の token を `[2.0, 0.0]` に変えてください。weights はどう変わりますか。
2. Lab 1B で `W_v` だけを変えてください。どの表示値が変わり、どれが同じままですか。
3. mask の実験を `4 x 4` 行列に拡張してください。
4. 実験 3 で `num_heads` を `2` から `1` に変えてください。どの shape が同じままですか。
5. 普通の RNN より Attention が長距離 token の相互作用を扱いやすい理由を説明してください。
6. Attention 重みが役立つが、完全な説明ではない場面を 1 つ説明してください。

<details>
<summary>参考実装と解説</summary>

1. 3 番目の token は第 1 特徴方向の query に近くなるため、そのような query からの attention weight が上がりやすくなります。正確な数値は dot-product 表全体で決まります。
2. `W_v` だけを変えると、value vectors と最終 attention output が変わります。attention scores と weights は queries と keys から作るので変わりません。
3. `4 x 4` の causal mask は、各位置が自分自身と過去位置を見られ、未来位置を見られない形にします。
4. 最終出力 shape は `[batch, seq, embed_dim]` のままです。変わるのは embedding dimension を heads にどう分割するかです。
5. Attention では各 token が見える token へ直接アクセスできます。普通の RNN は情報を順番に何ステップも渡すため、長距離情報が弱くなりやすいです。
6. Attention weights は、ある層がどの token を重視したかの手がかりになります。ただし value projection、residual path、後続層、output head も最終結果を変えるので、完全な説明ではありません。

</details>

## まとめ

- Attention は token が関連する文脈を直接選べるようにする。
- Q/K/V は学習された視点で、一致判定と内容取得を分ける。
- scaled dot-product attention は、スコア、softmax、重み付き和でできている。
- causal mask は生成時の未来の覗き見を防ぐ。
- multi-head attention は複数の部分空間から関係を見る。
