---
title: "1.2 迭代器与生成器进阶"
sidebar_position: 9
description: "从惰性计算、流式处理、生成器管道到 `yield from`，理解迭代器和生成器为什么特别适合数据与服务代码。"
keywords: [iterator, generator, yield, yield from, lazy evaluation, streaming]
---

# 迭代器与生成器进阶

:::tip 本节定位
迭代器和生成器最容易被误解成“语法技巧”。  
但在真实工程里，它们最重要的价值其实是：

> **让数据一边产生、一边消费，而不是一次性全塞进内存。**

这在数据处理、日志流、批量任务和服务端代码里非常常见。
:::

## 学习目标

- 理解迭代器和生成器在工程中的核心价值
- 理解惰性计算为什么能显著降低内存压力
- 学会构建简单生成器管道
- 通过可运行示例掌握 `yield` 和 `yield from` 的使用场景

---

## 一、为什么工程代码很喜欢生成器？

### 1.1 因为很多数据是“流”，不是“块”

例如：

- 日志流
- 文件逐行读取
- 网络请求结果
- 大批量样本处理

如果每次都先全部读进列表，  
很容易变成：

- 内存浪费
- 延迟增加

### 1.2 生成器的核心价值

它让你可以：

- 需要时再产出下一个值

这就是惰性计算。

### 1.3 一个类比

列表像一次性备好一大桌菜。  
生成器像按桌号一道一道上菜。

如果客人很多、菜很多，后者通常更省资源。

---

## 二、先看一个滑动窗口生成器

```python
def sliding_window(nums, size):
    for i in range(len(nums) - size + 1):
        yield nums[i : i + size]


for window in sliding_window([1, 2, 3, 4, 5], 3):
    print(window)
```

### 2.1 这段代码为什么有价值？

因为它已经展示了生成器的本质：

- 不是一次性返回所有窗口
- 而是一个一个产出

### 2.2 这类写法在哪常见？

例如：

- 时间序列窗口
- NLP 分块
- 批处理切片

---

## 三、生成器管道：把多个步骤串起来

工程里更常见的不是一个生成器，  
而是一串生成器组成的流水线。

```python
def read_lines():
    lines = [
        "INFO request ok",
        "ERROR db timeout",
        "INFO cache hit",
        "ERROR auth failed",
    ]
    for line in lines:
        yield line


def filter_errors(lines):
    for line in lines:
        if "ERROR" in line:
            yield line


def normalize(lines):
    for line in lines:
        yield line.lower()


pipeline = normalize(filter_errors(read_lines()))

for item in pipeline:
    print(item)
```

### 3.1 这个例子最想教什么？

工程里很多数据处理都可以拆成：

- 读取
- 过滤
- 变换

如果每一步都生成完整列表，  
链路会更重；  
用生成器管道则更自然。

### 3.2 为什么这对 AI 工程也有用？

因为你会经常处理：

- 样本流
- 日志流
- 检索结果流

这类场景天然适合生成器管道。

---

## 四、`yield from` 为什么值得学？

### 4.1 它解决什么问题？

当一个生成器只是想把另一个可迭代对象继续往外转发时，  
`yield from` 会让代码更清晰。

```python
def chunk_batches():
    yield [1, 2]
    yield [3, 4]


def flatten():
    for batch in chunk_batches():
        yield from batch


print(list(flatten()))
```

### 4.2 为什么它比双重循环更值得学？

因为它表达意图更明确：

- “把子迭代器的内容继续向外产出”

---

## 五、最容易踩的坑

### 5.1 误区一：生成器一定更快

它通常更省内存，  
但不代表所有场景都绝对更快。

### 5.2 误区二：生成器只能遍历一次

很多时候这是设计特征，不是 bug。  
如果你需要重复消费，就要重新创建它。

### 5.3 误区三：为了用生成器而用生成器

如果数据量很小、逻辑很简单，  
直接列表也许更好读。

---

## 小结

这节最重要的是建立一个工程直觉：

> **生成器和迭代器最适合处理“逐步产生、逐步消费”的数据流，它们的价值主要体现在节省内存、降低中间副本和组织流水线。**

只要这层理解清楚，  
后面你在做日志处理、样本管道和流式服务时就会自然想到它们。

---

## 练习

1. 把 `sliding_window` 改成按固定 batch size 产出数据块。
2. 用 `yield from` 再写一个把嵌套列表拉平的例子。
3. 想一想：什么时候列表更合适，什么时候生成器更合适？
4. 你能否把一个现有的数据处理函数改写成生成器管道？
