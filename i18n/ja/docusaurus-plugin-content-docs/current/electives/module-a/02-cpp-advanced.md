---
title: "E.A.2 C++ 応用"
sidebar_position: 2
description: "デプロイコードでよく見る C++ 応用概念、所有権、RAII、スマートポインタ、インターフェースを学びます。"
keywords: [C++, RAII, smart pointer, virtual, move semantics, threading, deployment]
---

# E.A.2 C++ 応用

![C++ RAII と所有権マップ](/img/course/elective-cpp-raii-ownership-map-ja.webp)

デプロイで出てくる C++ 応用の中心は、ほとんどこの問いです。誰がリソースを所有し、いつ解放するのか。

## 所有権とインターフェースの例を動かす

`advanced.cpp` を作成します。

```cpp
#include <iostream>
#include <memory>

struct Engine {
    virtual ~Engine() = default;
    virtual float run(float input) = 0;
};

struct CpuEngine : Engine {
    float run(float input) override {
        return input * 0.84f;
    }
};

class Session {
public:
    explicit Session(std::unique_ptr<Engine> engine)
        : engine_(std::move(engine)) {}

    void predict() {
        std::cout << "cpu_score=" << engine_->run(1.0f) << "\n";
        std::cout << "session_done\n";
    }

private:
    std::unique_ptr<Engine> engine_;
};

int main() {
    Session session(std::make_unique<CpuEngine>());
    session.predict();
    return 0;
}
```

実行します。

```bash
c++ -std=c++17 advanced.cpp -o advanced
./advanced
```

期待される出力:

```text
cpu_score=0.84
session_done
```

## 注目するところ

| C++ の要素 | デプロイでの意味 |
|---|---|
| `Engine` インターフェース | ビジネスコードが CPU/GPU/runtime backend を切り替えられる |
| `std::unique_ptr` | 1 つの所有者だけが engine リソースを管理する |
| `std::move` | 所有権を `Session` に移す |
| `virtual ~Engine()` | インターフェース経由でも安全に片付けられる |
| RAII | リソース寿命がオブジェクト寿命に従う |

## 少し変えてみる

2 つ目の engine を追加します。

```cpp
struct FastEngine : Engine {
    float run(float input) override {
        return input * 0.91f;
    }
};
```

それから `CpuEngine` を `FastEngine` に置き換えます。`Session` の他のコードは変えないでください。

<details>
<summary>参考解答と解説</summary>

`FastEngine` は同じ `Engine` インターフェースを実装します。そのため `Session` 側では、構築時に別のオブジェクトを渡すだけで済みます。

```cpp
Session session(std::make_unique<FastEngine>());
```

重要なのは、`Session` が `CpuEngine` そのものではなくインターフェースに依存していることです。`std::unique_ptr` により所有権が明確になり、session が 1 つの engine インスタンスを所有し、終了時に自動的に解放されます。

</details>

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
deployment_target: local inference, edge device, model server, or optimization experiment
artifact: C++ snippet, benchmark, model artifact, serving config, or deployment note
metric: latency, memory, throughput, model size, accuracy drop, or reliability
failure_check: ABI/build issue, hardware mismatch, quantization loss, or serving bottleneck
Expected_output: reproducible deployment or optimization evidence, not only theory notes
```

## 合格チェック

`Session` が engine を所有する理由、ここで `unique_ptr` が raw pointer より安全な理由、インターフェースで runtime backend を差し替えられる理由を説明できれば合格です。
