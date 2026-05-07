---
title: "E.A.1 C++ プログラミング基礎"
sidebar_position: 1
description: "推論らしい小さな実行例で、モデルデプロイに必要な C++ の直感を作ります。"
keywords: [C++, basics, vector, reference, function, class, deployment]
---

# E.A.1 C++ プログラミング基礎

![C++ ランタイムとメモリモデル図](/img/course/elective-cpp-runtime-memory-ja.png)

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

## 合格チェック

ファイルをコンパイルでき、入力値を変え、なぜ選ばれるクラスが変わったのか説明でき、推論プログラムで `std::vector<float>` が何を表すか言えれば合格です。
