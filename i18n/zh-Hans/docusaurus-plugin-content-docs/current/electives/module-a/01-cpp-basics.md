---
title: "E.A.1 C++ 编程基础"
sidebar_position: 1
description: "用一个可运行的推理风格小例子建立 C++ 部署直觉。"
keywords: [C++, 基础, vector, 引用, 函数, 类, 部署]
---

# E.A.1 C++ 编程基础

![C++ 运行时与内存模型图](/img/course/elective-cpp-runtime-memory.webp)

读部署代码前，你不需要先变成 C++ 专家。先掌握反复出现的小集合：类型、函数、`std::vector`、引用、编译和清晰输出。

## 运行最小推理风格程序

创建 `demo.cpp`：

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

运行：

```bash
c++ -std=c++17 demo.cpp -o demo
./demo
```

预期输出：

```text
best_class=2
score=2.1
```

## 重点看什么

| C++ 点 | 部署含义 |
|---|---|
| `std::vector<float>` | 一个简单的类 tensor 容器 |
| 明确类型 `float` / `int` | 编译器必须知道数据和值类型 |
| `static_cast<int>(...)` | 明确转换类型，而不是赌它能跑 |
| 先编译再运行 | 部署通常产出二进制，而不只是脚本 |
| 打印结果 | 每个部署测试都需要可复现证据 |

## 动手改一下

把 logits 改成：

```cpp
std::vector<float> logits = {3.4f, 0.3f, 2.1f};
```

再次运行，预期 `best_class` 变成 `0`。

## 通过检查

你能编译文件、修改输入值、解释为什么类别改变，并说清楚 `std::vector<float>` 在推理程序里代表什么，就算通过。
