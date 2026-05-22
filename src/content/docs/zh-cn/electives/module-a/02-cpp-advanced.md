---
title: "E.A.2 C++ 进阶"
description: "学习部署代码里最常见的 C++ 进阶概念：所有权、RAII、智能指针和接口。"
sidebar:
  order: 2
head:
  - tag: meta
    attrs:
      name: keywords
      content: "C++, RAII, 智能指针, virtual, move semantics, threading, deployment"
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

<details>
<summary>操作参考与检查点</summary>

`FastEngine` 应该实现同一个 `Engine` 接口，所以 `Session` 只需要在构造时接收另一个对象：

```cpp
Session session(std::make_unique<FastEngine>());
```

关键点是：`Session` 依赖接口，而不是直接依赖 `CpuEngine`。`std::unique_ptr` 把所有权说清楚：session 拥有唯一的 engine 实例，session 结束时资源会自动释放。

</details>

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
部署目标：本地推理、边缘设备、模型服务器或优化实验
工件：C++ 代码片段、基准测试、模型工件、服务配置或部署说明
指标：延迟、内存、吞吐量、模型大小、准确率下降或可靠性
失败检查：ABI/构建问题、硬件不匹配、量化损失或服务瓶颈
期望产出：可复现的部署或优化证据，而不只是理论笔记
```

## 通过检查

你能解释为什么 `Session` 拥有 engine、为什么这里 `unique_ptr` 比裸指针更安全，以及接口如何让部署路径切换 runtime 后端，就算通过。
