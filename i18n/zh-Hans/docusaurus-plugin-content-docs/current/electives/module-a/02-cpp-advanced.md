---
title: "E.A.2 C++ 进阶"
sidebar_position: 2
description: "学习部署代码里最常见的 C++ 进阶概念：所有权、RAII、智能指针和接口。"
keywords: [C++, RAII, 智能指针, virtual, move semantics, threading, deployment]
---

# E.A.2 C++ 进阶

![C++ RAII 与所有权地图](/img/course/elective-cpp-raii-ownership-map.webp)

部署里的 C++ 进阶问题，核心通常只有一句：谁拥有资源，资源什么时候释放？

## 运行所有权与接口示例

创建 `advanced.cpp`：

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

运行：

```bash
c++ -std=c++17 advanced.cpp -o advanced
./advanced
```

预期输出：

```text
cpu_score=0.84
session_done
```

## 重点看什么

| C++ 点 | 部署含义 |
|---|---|
| `Engine` 接口 | 业务代码可以切换 CPU/GPU/runtime 后端 |
| `std::unique_ptr` | 只有一个所有者控制 engine 资源 |
| `std::move` | 所有权被转移进 `Session` |
| `virtual ~Engine()` | 通过接口清理资源也安全 |
| RAII | 资源生命周期跟随对象生命周期 |

## 动手改一下

新增一个 engine：

```cpp
struct FastEngine : Engine {
    float run(float input) override {
        return input * 0.91f;
    }
};
```

然后把 `CpuEngine` 换成 `FastEngine`。`Session` 的其他代码不应该改。

## 通过检查

你能解释为什么 `Session` 拥有 engine、为什么这里 `unique_ptr` 比裸指针更安全，以及接口如何让部署路径切换 runtime 后端，就算通过。
