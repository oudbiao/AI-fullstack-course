---
title: "1.1 C++ 编程基础"
sidebar_position: 1
description: "从变量、函数、引用、vector 和简单类出发，建立面向模型部署场景的 C++ 基础直觉。"
keywords: [C++, basics, vector, reference, function, class, deployment]
---

# C++ 编程基础

:::tip 本节定位
很多做 AI 应用的同学平时更熟悉 Python。  
但一旦进入模型部署、推理服务、边缘设备或高性能模块，就很容易碰到 C++。

这节课不是按传统语言教材那样从头铺满语法，  
而是站在“模型部署会用到什么”的角度，先把最常见的 C++ 基础补齐。
:::

## 学习目标

- 理解为什么模型部署场景经常会接触 C++
- 掌握变量、函数、引用、`std::vector` 和简单类这几块基础
- 通过可编译示例建立“C++ 怎么组织数据和逻辑”的直觉
- 知道后面进阶课为什么会继续讲 RAII、智能指针和抽象接口

---

## 一、为什么模型部署会碰到 C++？

### 1.1 因为它更接近底层执行环境

常见场景包括：

- 推理引擎 SDK
- ONNX / TensorRT / OpenVINO 的底层接口
- 高性能后处理模块
- 边缘设备本地推理

### 1.2 它不是“为了炫技”

很多时候不是因为 C++ 更酷，  
而是因为部署场景更关注：

- 性能
- 内存控制
- 原生库集成

### 1.3 一个现实目标

对很多 AI 工程同学来说，  
第一目标通常不是“精通所有 C++ 语法”，  
而是：

> **看懂基本代码，能写简单模块，能和部署链路接起来。**

---

## 二、先建立最常见的几个基本概念

### 2.1 变量和类型

C++ 是静态类型语言。  
你通常需要明确写出变量类型：

- `int`
- `float`
- `bool`
- `std::string`

### 2.2 函数

函数需要声明：

- 返回类型
- 参数类型

这会让接口更明确，也更利于编译器检查。

### 2.3 引用

引用很常见，因为它能避免不必要拷贝。  
尤其在处理大向量或张量时很重要。

例如：

- `const std::vector<float>& logits`

表示：

- 只读引用
- 不复制整份数据

### 2.4 `std::vector`

在部署代码里你会经常看到它。  
它可以先理解成：

- “比 Python list 更类型固定的动态数组”

---

## 三、先跑一个真正和部署场景接近的 C++ 示例

下面这个例子会做一件非常典型的事：

- 给一组分类分数
- 找出 top-1 类别

这比纯粹打印 `hello world` 更贴近模型后处理场景。

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

编译方式：

```bash
c++ -std=c++17 demo.cpp -o demo
./demo
```

### 3.1 这个例子最该看什么？

先看三处：

1. `std::vector<float>`  
   数据如何组织
2. `const std::vector<float>&`  
   为什么函数参数常用引用
3. `argmax`  
   最常见的部署后处理函数长什么样

### 3.2 为什么这里强调引用？

因为如果直接按值传参：

- `std::vector<float> logits`

函数调用时会复制整份数据。  
在部署和推理路径里，这类不必要拷贝会很常见，也很浪费。

---

## 四、类在部署代码里通常怎么出现？

### 4.1 类不只是面向对象考试题

在部署场景里，类经常用来表示：

- 一个模型 runner
- 一个 tokenizer
- 一个后处理器

### 4.2 一个简单类示例

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

这个例子想表达的是：

- 类可以把配置和行为包在一起

这在部署系统里很常见。

---

## 五、Python 背景同学最容易卡的地方

### 5.1 编译

Python 是解释执行，  
C++ 通常先编译再运行。

### 5.2 类型要提前写清楚

这会让一开始感觉啰嗦，  
但也能更早发现错误。

### 5.3 内存和拷贝更值得注意

在 Python 里很多拷贝细节没那么明显，  
但在 C++ 和部署性能路径里，这会变得很关键。

---

## 六、常见误区

### 6.1 误区一：只要会写 Python，就能直接无痛看懂 C++

逻辑能迁移，  
但类型、引用、所有权这些概念还是需要单独适应。

### 6.2 误区二：基础语法没意义，直接上引擎就行

没有变量、函数、类和引用这些基础，  
后面很多部署 SDK 会看得非常吃力。

### 6.3 误区三：C++ 基础课就该和 AI 完全无关

对这门选修来说，更好的方式就是：

- 直接围绕部署和推理路径来学

---

## 小结

这节最重要的，不是把 C++ 学成一门独立专业，  
而是先建立一个部署友好的基础：

> **看懂基本类型、函数、引用、vector 和简单类，已经足够支撑你继续进入模型部署和推理引擎的主线。**

只要这几块稳住，后面很多“看起来很底层”的代码就不会那么吓人了。

---

## 练习

1. 把 `argmax` 改成返回 top-2 下标，练一下 `vector` 操作。
2. 试着把 `ThresholdFilter` 的阈值改成构造参数之外还能动态设置。
3. 为什么说部署路径里“避免不必要拷贝”很重要？
4. 用自己的话解释：`const std::vector<float>&` 为什么比按值传参更适合这里？
