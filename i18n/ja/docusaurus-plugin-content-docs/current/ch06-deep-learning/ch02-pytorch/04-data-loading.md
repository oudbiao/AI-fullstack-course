---
title: "2.6 データ読み込み"
sidebar_position: 4
description: "Dataset、DataLoader、batch、shuffle、そして訓練データの分割を理解して、モデルがデータを安定して少しずつ受け取れるようにしよう。"
keywords: [Dataset, DataLoader, batch, shuffle, random_split, PyTorch]
---

# データ読み込み

![Dataset DataLoader Batch の流れ図](/img/course/dataset-dataloader-batch-flow-ja.png)

## 学習目標

- 学習時に、なぜすべてのデータを一度にモデルへ入れないのかを理解する
- `Dataset` と `DataLoader` の役割分担を理解する
- もっともシンプルな自作データセットを書けるようになる
- `batch_size`、`shuffle`、訓練データ / 検証データの分割を理解する

---

## まず地図を作ろう

この節で、最初にしっかり見ておきたいのは次の流れです。

```mermaid
flowchart LR
    A["元のサンプル"] --> B["Dataset: 1件のサンプルの取り方を定義"]
    B --> C["DataLoader: batch を作る、シャッフルする、反復する"]
    C --> D["訓練ループが batch ごとに読み込む"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style D fill:#e8f5e9,stroke:#2e7d32,color:#333
```

つまり、この節で本当に解決したいのは、「2つのクラス名を覚えること」ではなく、

- データがどうやって安定して訓練ループに流れ込むのか

ということです。

## この節は前後の内容とどうつながるか

- これまでの節で、Tensor、勾配、モデルはすでに出てきました
- この節からは、「学習時のデータをどうやって batch 単位で流し込むか」を扱います
- 次の節では、訓練ループの中で「モデル + データ」を本当に結びつけます

つまりこの節は、学習の流れに「データ側」を足すパートです。

## 1. なぜデータローダーが必要なのか？

モデルにごはんを食べさせるところを想像してみましょう。

- データ全部を一気に入れる: 多すぎてつらいし、メモリが足りないかもしれない
- 少しずつ入れる: 安定しやすく、何度も学習しやすい

深層学習では、この「一口分」を **batch** と呼びます。

なので、普通はこうはしません。

```python
pred = model(all_data_once)
```

代わりに、こうします。

```python
for batch_x, batch_y in dataloader:
    pred = model(batch_x)
```

### 1.1 最初に `batch` について覚えるなら、何が大事？

まずはこの1文だけ覚えれば十分です。

> **batch = 1回のパラメータ更新で、モデルが見る少量のサンプル。**

これはとても大事です。なぜなら、この後に出てくる

- `batch_size`
- `shuffle`
- `steps per epoch`

は、すべてこの考え方を中心に動いているからです。

---

## 2. `Dataset` と `DataLoader` はそれぞれ何をするのか？

次のように考えるとわかりやすいです。

| コンポーネント | たとえ | 役割 |
|---|---|---|
| `Dataset` | 倉庫 | PyTorch に「i 番目のデータはこれ」と教える |
| `DataLoader` | 運搬車 | batch 化、シャッフル、並列読み込みを担当する |

一言でいうと、

- `Dataset` は「1件のデータをどう取るか」を担当
- `DataLoader` は「1件のデータをどうやって batch にまとめるか」を担当

### 2.1 なぜこの2つを分けるのか？

理由は、解決している問題の階層が違うからです。

- `Dataset` は「データの定義」に近い
- `DataLoader` は「学習時にどう流すか」に近い

このように分けると、次の利点があります。

- 同じデータセットでも、違う batch 設定を使える
- 同じ `DataLoader` の考え方を、別のデータセットにも使える

---

## 3. まずは最小の `Dataset` を見よう

```python
import torch
from torch.utils.data import Dataset

class StudentDataset(Dataset):
    def __init__(self):
        # 2つの特徴量: 勉強時間、練習問題の完了数
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0]
        ])

        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()

print("データセットのサイズ:", len(dataset))
print("0 番目のサンプル:", dataset[0])
print("3 番目のサンプル:", dataset[3])
```

### 自作データセットで最低限必要なものは？

基本的には、次の2つのメソッドが必要です。

- `__len__()`：全サンプル数を返す
- `__getitem__(idx)`：`idx` 番目のデータを返す

### 3.1 初めて `Dataset` を自分で書くとき、何を最初に確認する？

まずは次の3つをチェックしましょう。

1. `len(dataset)` が正しいか
2. `dataset[i]` が `(x, y)` のように、期待した形で返るか
3. 各サンプルの shape と dtype が合っているか

この部分が不安定だと、その後の `DataLoader` や訓練ループで問題を見つけるのがどんどん難しくなります。

---

## 4. データセットを `DataLoader` に渡す

```python
import torch
from torch.utils.data import Dataset, DataLoader

class StudentDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0]
        ])
        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"batch {batch_idx}")
    print("batch_x:\n", batch_x)
    print("batch_y:\n", batch_y)
```

### ここで特に大事な2つの引数

| 引数 | 役割 |
|---|---|
| `batch_size=2` | 1回で 2 件のサンプルを取る |
| `shuffle=True` | 各 epoch の最初に順番を入れ替える |

### 4.1 `DataLoader` の段階で shape を確認するのが大事な理由

ここは、訓練前にデータを確認できる最後のわかりやすい場所だからです。  
最初に `DataLoader` を書いたら、ぜひ次をやってみましょう。

```python
for batch_x, batch_y in loader:
    print(batch_x.shape, batch_y.shape)
    break
```

これで、すぐに次のことがわかります。

- batch が正しく作れているか
- ラベルの shape は適切か
- データが訓練ループにそのまま入れられる形になっているか

---

## 5. なぜデータをシャッフルするのか？

もし元のデータが、ある順番で並んでいたとします。たとえば、

- 最初の 100 件は低得点
- 後ろの 100 件は高得点

このような並びだと、モデルは学習の前半で似たデータばかりを見ることになり、学習が偏りやすくなります。  
そのため、訓練データでは普通 `shuffle=True` を使います。

ただし、検証データ / テストデータでは、普通はシャッフルしません。

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

## 6. 訓練データと検証データはどう分ける？

PyTorch には `random_split` があります。

```python
import torch
from torch.utils.data import Dataset, random_split

class StudentDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0]
        ])
        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()

train_dataset, val_dataset = random_split(
    dataset,
    [4, 2],
    generator=torch.Generator().manual_seed(42)
)

print("訓練データサイズ:", len(train_dataset))
print("検証データサイズ:", len(val_dataset))
```

### ここで乱数シードを設定するのはなぜ？

設定しないと、毎回分割結果が変わる可能性があるからです。  
学習やデバッグのときは、再現しやすいように乱数シードを固定しておくと便利です。

---

## 7. ひと通り動く小さな例

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class StudentDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0]
        ])
        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0],
            [96.0],
            [99.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()

train_dataset, val_dataset = random_split(
    dataset,
    [6, 2],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

print("訓練データの batch:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\n検証データの batch:")
for x, y in val_loader:
    print(x.shape, y.shape)
```

---

## 8. `batch_size` はどう選べばいい？

初心者のうちは、まず感覚でこう覚えれば大丈夫です。

- 小さい batch: 更新回数が多く、ばらつきも大きい
- 大きい batch: より安定するが、メモリを多く使う

今が学習用の小さな例なら、まずは次あたりで十分です。

- `8`
- `16`
- `32`

もっと大きなモデルを扱うようになったら、そこで初めてメモリや処理速度のバランスを考えれば大丈夫です。

### 8.1 まずは安定した設定を選ぶ考え方

初心者の段階では、次の順番で考えるとよいです。

- まず、自分の環境で無理なく動く `batch_size` を選ぶ
- 次に、loss が安定しているか、速度が受け入れられるかを見る
- 最初から「大きい batch の方が上級者向け」と考えすぎない

なぜなら、学習の入り口で大事なのは性能の限界ではなく、

- 訓練の流れが通ること
- shape が正しいこと
- loss がちゃんと下がること

だからです。

---

## 9. 初心者がよくやる勘違い

### 1. `Dataset` はすべてのデータをメモリに読み込むものだと思う

必ずしもそうではありません。  
この教材ではわかりやすくそう書いていますが、実際の開発では `__getitem__()` のタイミングでディスクから読み込むこともよくあります。

### 2. 訓練データをシャッフルしない

動くことはありますが、普通はよい習慣ではありません。

### 3. 配列だけを書いて、データセットクラスを書かない

小さな実験ならそれでもよいですが、少しきちんとしたプロジェクトでは `Dataset` として書くのがおすすめです。

---

## まとめ

この節でいちばん大事なのは、クラス名を覚えることではなく、「データの流れ」をイメージできるようになることです。

1. データはまず `Dataset` に 1 件ずつ整理される
2. それを `DataLoader` が batch にまとめる
3. そして batch ごとにモデルへ渡す

次の節では、モデル、損失関数、最適化器、そしてデータローダーをつなげて、完全な訓練フローを書いていきます。

## この節でいちばん持ち帰ってほしいこと

1文にまとめるなら、こうです。

> **`Dataset` は「1件のデータがどういう形か」を決め、`DataLoader` は「そのデータをどう batch にして訓練へ送るか」を決める。**

---

## 練習

1. `StudentDataset` のサンプル数を 12 件に増やし、訓練データと検証データをもう一度分けてみましょう。
2. `batch_size` を `1`、`2`、`4` に変えて、各 epoch の batch 数を観察しましょう。
3. `shuffle=True` のときに、2 回続けて読み込んだ最初の batch を表示して、順番が変わるか確認しましょう。
