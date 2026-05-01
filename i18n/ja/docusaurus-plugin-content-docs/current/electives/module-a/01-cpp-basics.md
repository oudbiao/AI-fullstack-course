---
title: "1.1 C++ プログラミング基礎"
sidebar_position: 1
description: "変数、関数、参照、vector、簡単なクラスから始めて、モデルデプロイ場面に向いた C++ の基礎感覚を身につけます。"
keywords: [C++, basics, vector, reference, function, class, deployment]
---

# C++ プログラミング基礎

![C++ の実行とメモリモデル図](/img/course/elective-cpp-runtime-memory.png)

:::tip この節の位置づけ
AI アプリを作る同学の多くは、ふだん Python に慣れています。  
でも、モデルデプロイ、推論サービス、エッジデバイス、高性能モジュールに入ると、C++ によく出会います。

この授業では、従来の言語教材のように最初から文法を全部埋めるのではなく、  
「モデルデプロイで何を使うか」という観点から、よく使う C++ の基礎を先に補います。
:::

## 学習目標

- モデルデプロイの場面で C++ に触れることが多い理由を理解する
- 変数、関数、参照、`std::vector`、簡単なクラスの基礎を身につける
- コンパイルできる例を通して、「C++ がどうデータとロジックを組み立てるか」の感覚をつかむ
- 次の発展編で RAII、スマートポインタ、抽象インターフェースを引き続き学ぶ理由を知る

---

## 一、なぜモデルデプロイで C++ に出会うのか？

### 1.1 それがより低レベルの実行環境に近いから

よくある場面は次の通りです。

- 推論エンジン SDK
- ONNX / TensorRT / OpenVINO の低レベル API
- 高性能な後処理モジュール
- エッジデバイスでのローカル推論

### 1.2 「かっこいいから」ではない

多くの場合、C++ のほうがかっこいいから使うのではなく、  
デプロイ場面では次の点がより重視されるからです。

- 性能
- メモリ制御
- ネイティブライブラリとの統合

### 1.3 現実的な目標

多くの AI エンジニアにとって、  
最初の目標は「C++ の文法を全部極めること」ではなく、  
次のようなものです。

> **基本コードを読めて、簡単なモジュールを書けて、デプロイの流れにつなげられること。**

---

## 二、まずはよく出る基本概念を押さえる

### 2.1 変数と型

C++ は静的型付き言語です。  
通常は変数の型を明示して書きます。

- `int`
- `float`
- `bool`
- `std::string`

### 2.2 関数

関数では次を宣言する必要があります。

- 戻り値の型
- 引数の型

これによりインターフェースが明確になり、コンパイラのチェックもしやすくなります。

### 2.3 参照

参照はとてもよく使われます。なぜなら、不要なコピーを避けられるからです。  
特に大きなベクトルやテンソルを扱うときに重要です。

例えば：

- `const std::vector<float>& logits`

これは次を意味します。

- 読み取り専用の参照
- データ全体をコピーしない

### 2.4 `std::vector`

デプロイコードではよく見かけます。  
まずは次のように理解するとよいです。

- 「Python の list より型が固定された動的配列」

---

## 三、デプロイ場面にかなり近い C++ の例を実際に動かしてみる

次の例では、非常に典型的な処理を行います。

- 分類スコアの集まりを受け取る
- top-1 のカテゴリを見つける

これは、単に `hello world` を表示するよりも、モデルの後処理に近いです。

```cpp
#include <iostream>
#include <vector>
#include <string>

int argmax(const std::vector<float>& logits) {
    int best_idx = 0;
    float best_score = logits[0];

    for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
        if (logits[i] > best_score) {
            best_score = logits[i];
            best_idx = i;
        }
    }
    return best_idx;
}

int main() {
    std::vector<std::string> labels = {"cat", "dog", "bird"};
    std::vector<float> logits = {1.2f, 0.8f, 2.5f};

    int best_idx = argmax(logits);
    std::cout << "predicted label = " << labels[best_idx] << std::endl;
    return 0;
}
```

コンパイル方法：

```bash
c++ -std=c++17 demo.cpp -o demo
./demo
```

### 3.1 この例でまず見るべきところ

まずは次の 3 点です。

1. `std::vector<float>`  
   データがどう整理されているか
2. `const std::vector<float>&`  
   なぜ関数引数で参照をよく使うのか
3. `argmax`  
   デプロイ後処理の定番関数がどんな形か

### 3.2 なぜここで参照を強調するのか？

もし値渡しにすると、

- `std::vector<float> logits`

関数呼び出しのたびにデータ全体をコピーすることになります。  
デプロイや推論の経路では、こうした不要なコピーはよく起きますし、無駄も大きいです。

---

## 四、クラスはデプロイコードでどう現れるのか？

### 4.1 クラスはオブジェクト指向の試験問題だけではない

デプロイ場面では、クラスはよく次のようなものを表します。

- モデル runner
- tokenizer
- 後処理器

### 4.2 簡単なクラスの例

```cpp
#include <iostream>
#include <vector>

class ThresholdFilter {
public:
    explicit ThresholdFilter(float threshold) : threshold_(threshold) {}

    std::vector<float> apply(const std::vector<float>& values) const {
        std::vector<float> kept;
        for (float v : values) {
            if (v >= threshold_) {
                kept.push_back(v);
            }
        }
        return kept;
    }

private:
    float threshold_;
};

int main() {
    ThresholdFilter filter(0.5f);
    std::vector<float> scores = {0.2f, 0.6f, 0.9f};
    std::vector<float> kept = filter.apply(scores);

    for (float v : kept) {
        std::cout << v << std::endl;
    }
}
```

この例で伝えたいのは次の点です。

- クラスは設定と動作をひとまとめにできる

これはデプロイシステムでとてもよくある形です。

---

## 五、Python が得意な人がつまずきやすい点

### 5.1 コンパイル

Python は解釈実行ですが、  
C++ は通常、先にコンパイルしてから実行します。

### 5.2 型を先に明示する必要がある

最初は冗長に感じるかもしれませんが、  
その分、早い段階でエラーを見つけやすくなります。

### 5.3 メモリとコピーに注意しやすい

Python ではコピーの細かい違いがあまり見えないことがありますが、  
C++ やデプロイ性能の経路では、ここがとても重要になります。

---

## 六、よくある誤解

### 6.1 誤解 1：Python が書ければ、C++ もすぐに自然に読める

ロジックは移せますが、  
型、参照、所有権といった概念は別途慣れが必要です。

### 6.2 誤解 2：基礎文法は意味がない、すぐエンジンに入ればいい

変数、関数、クラス、参照の基礎がないと、  
その後の多くのデプロイ SDK はかなり読みづらくなります。

### 6.3 誤解 3：C++ の基礎授業は AI と完全に関係ないはず

この選修では、次のように学ぶのがいちばんよいです。

- デプロイと推論の流れに直接つなげて学ぶ

---

## まとめ

この節でいちばん大事なのは、C++ を独立した専門分野として完璧にすることではなく、  
まずデプロイにやさしい基礎感覚を持つことです。

> **基本的な型、関数、参照、vector、簡単なクラスが読めれば、モデルデプロイや推論エンジンの本線に十分進めます。**

この 4〜5 個のポイントが安定すれば、後で出てくる「いかにも低レベル」に見えるコードも、そんなに怖くなくなります。

---

## 練習

1. `argmax` を top-2 のインデックスを返すように変えて、`vector` の操作を練習してみましょう。
2. `ThresholdFilter` の閾値を、コンストラクタ以外からも動的に設定できるようにしてみましょう。
3. なぜデプロイの流れでは「不要なコピーを避ける」ことが重要なのでしょうか？
4. 自分の言葉で説明してみましょう：`const std::vector<float>&` は、なぜ値渡しよりここに向いているのでしょうか？
