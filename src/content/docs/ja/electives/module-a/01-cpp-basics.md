---
title: "E.A.1 C++ プログラミング基礎"
description: "推論らしい小さな実行例で、モデルデプロイに必要な C++ の直感を作ります。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "C++, basics, vector, reference, function, class, deployment"
---

# E.A.1 C++ プログラミング基礎

![C++ ランタイムとメモリモデル図](/img/course/elective-cpp-runtime-memory-ja.webp)

デプロイコードを読む前に、C++ の専門家になる必要はありません。まず、何度も出る小さな範囲だけ押さえます。型、関数、`std::vector`、参照、コンパイル、明確な出力です。

## 最小の推論風プログラムを動かす

`demo.cpp` を作成します。

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<float> logits = {1.2f, 0.3f, 2.1f};
    int best_index = 0;

    for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
        if (logits[i] > logits[best_index]) {
            best_index = i;
        }
    }

    std::cout << "best_class=" << best_index << "\n";
    std::cout << "score=" << logits[best_index] << "\n";
    return 0;
}
```

実行します。

```bash
c++ -std=c++17 demo.cpp -o demo
./demo
```

期待される出力:

```text
best_class=2
score=2.1
```

## 注目するところ

| C++ の要素 | デプロイでの意味 |
|---|---|
| `std::vector<float>` | tensor に近いシンプルなコンテナ |
| 明示的な `float` / `int` | コンパイラが型を知る必要がある |
| `static_cast<int>(...)` | 型変換を意図的に行う |
| コンパイルしてから実行 | デプロイではスクリプトだけでなくバイナリを作ることが多い |
| 出力を表示する | デプロイテストには再現できる証拠が必要 |

## 少し変えてみる

logits を次のように変えます。

```cpp
std::vector<float> logits = {3.4f, 0.3f, 2.1f};
```

もう一度実行すると、`best_class` は `0` になるはずです。

<details>
<summary>操作例と確認ポイント</summary>

変更後は `3.4`、`0.3`、`2.1` を比較するので、最大値はインデックス `0` にあります。そのため出力されるクラスは `0` になります。大事なのは「数値が大きい」だけでなく、この推論補助コードが logit ベクトルを走査し、最大値の位置を返していることです。

残す証拠は次の通りです。

- コンパイルコマンドが成功する。
- 出力が `best_class=0` に変わる。
- `std::vector<float>` が小さな tensor またはモデル出力配列の代わりになっていると説明できる。

</details>

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
デプロイ先：ローカル推論、エッジデバイス、モデルサーバー、または最適化実験
成果物: C++ スニペット、ベンチマーク、model artifact、serving 設定、または deployment メモ
指標：レイテンシ、メモリ、スループット、モデルサイズ、accuracy 低下、または信頼性
失敗確認：ABI/ビルドの問題、ハードウェア不一致、量子化損失、または配信ボトルネック
期待される成果: 理論メモだけでなく、再現可能なデプロイまたは最適化の証拠
```

## 合格チェック

ファイルをコンパイルでき、入力値を変え、なぜ選ばれるクラスが変わったのか説明でき、推論プログラムで `std::vector<float>` が何を表すか言えれば合格です。
