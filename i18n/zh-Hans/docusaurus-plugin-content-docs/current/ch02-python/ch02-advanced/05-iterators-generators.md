---
title: "2.2.5 迭代器与生成器"
sidebar_position: 5
description: "理解 Python 迭代协议和生成器的高效数据处理方式"
---

# 2.2.5 迭代器与生成器

![生成器流式数据处理图](/img/course/ch02-generator-streaming-data.webp)

## 本节定位

这一节解释 for 循环背后的机制，并引入更省内存的数据处理方式。迭代器和生成器在处理大文件、流式数据、训练数据加载时很有价值，先理解思想，再掌握最常见的 yield 写法。

## 学习目标

- 理解迭代器协议（`__iter__` 和 `__next__`）
- 掌握生成器函数（`yield`）的用法
- 理解生成器表达式
- 了解为什么生成器在处理大数据时非常重要

---

## 什么是迭代？

你已经用过很多次 `for` 循环了：

```python
for item in [1, 2, 3]:
    print(item)

for char in "Hello":
    print(char)

for key in {"a": 1, "b": 2}:
    print(key)
```

`for...in` 能遍历这些东西，是因为它们都是**可迭代对象（Iterable）**。那么问题来了：`for` 循环的背后到底发生了什么？

---

## 迭代器协议

### 手动迭代

`for` 循环的本质是这样的：

```python
numbers = [10, 20, 30]

# for 循环写法
for n in numbers:
    print(n)

# 等价的手动写法
iterator = iter(numbers)   # 1. 获取迭代器
print(next(iterator))      # 2. 获取下一个元素 → 10
print(next(iterator))      # 3. 获取下一个元素 → 20
print(next(iterator))      # 4. 获取下一个元素 → 30
# print(next(iterator))    # 5. 没有更多元素了 → 抛出 StopIteration
```

**迭代器协议**：
- `iter(对象)` → 获取迭代器
- `next(迭代器)` → 获取下一个元素
- 元素用完时抛出 `StopIteration` 异常

### 自定义迭代器

```python
class Countdown:
    """倒计时迭代器"""

    def __init__(self, start):
        self.current = start

    def __iter__(self):
        return self   # 返回自身作为迭代器

    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        value = self.current
        self.current -= 1
        return value

# 使用
for num in Countdown(5):
    print(num, end=" ")
# 输出: 5 4 3 2 1
```

不过手写迭代器比较麻烦——接下来介绍的**生成器**是更简洁的方式。

---

## 生成器函数（Generator）

生成器是一种**特殊的迭代器**，用 `yield` 关键字代替 `return`。

### 基本用法

```python
def countdown(n):
    """倒计时生成器"""
    while n > 0:
        yield n    # 暂停，返回 n，下次从这里继续
        n -= 1

# 使用方式和迭代器一样
for num in countdown(5):
    print(num, end=" ")
# 输出: 5 4 3 2 1
```

### yield vs return 的区别

```python
# return：函数执行完毕，一次性返回所有结果
def get_squares_return(n):
    result = []
    for i in range(n):
        result.append(i ** 2)
    return result

# yield：每次返回一个结果，暂停等待下次调用
def get_squares_yield(n):
    for i in range(n):
        yield i ** 2

# 使用效果一样
print(list(get_squares_return(5)))  # [0, 1, 4, 9, 16]
print(list(get_squares_yield(5)))   # [0, 1, 4, 9, 16]
```

**关键区别：**

| 特点 | `return` | `yield` |
|------|---------|---------|
| 返回方式 | 一次返回所有 | 每次返回一个 |
| 内存使用 | 全部加载到内存 | 按需生成，几乎不占内存 |
| 执行方式 | 执行完毕 | 暂停/恢复 |

### 生成器的执行过程

```python
def simple_gen():
    print("第一步")
    yield 1
    print("第二步")
    yield 2
    print("第三步")
    yield 3
    print("结束")

gen = simple_gen()   # 创建生成器，但不执行任何代码

print(next(gen))     # 执行到第一个 yield，打印"第一步"，返回 1
print(next(gen))     # 从上次暂停处继续，打印"第二步"，返回 2
print(next(gen))     # 打印"第三步"，返回 3
# next(gen)          # 打印"结束"，然后抛出 StopIteration
```

输出：

```
第一步
1
第二步
2
第三步
3
```

---

## 为什么需要生成器？—— 处理大数据

这是生成器最重要的应用场景。

### 问题：一次性加载太多数据

```python
# 假设你要处理一个 10GB 的文件
# 错误做法：一次性读入所有行
lines = open("huge_file.txt").readlines()  # 💥 内存爆炸！

# 正确做法：用生成器逐行处理
def read_large_file(filepath):
    with open(filepath, "r") as f:
        for line in f:   # 文件对象本身就是迭代器，逐行读取
            yield line.strip()

for line in read_large_file("huge_file.txt"):
    process(line)  # 一次只有一行在内存中
```

### 对比内存使用

```python
import sys

# 列表：所有元素都在内存中
big_list = [i ** 2 for i in range(1_000_000)]
print(f"列表占用内存: {sys.getsizeof(big_list):,} 字节")  # ~8MB

# 生成器：只记住当前状态
big_gen = (i ** 2 for i in range(1_000_000))
print(f"生成器占用内存: {sys.getsizeof(big_gen):,} 字节")  # ~200 字节！
```

8MB vs 200 字节——差了 4 万倍！当数据量更大时（比如处理几百万条训练数据），这个差距就是"程序能跑"和"内存溢出崩溃"的区别。

---

## 生成器表达式

列表推导式的 `[]` 换成 `()`，就变成了**生成器表达式**：

```python
# 列表推导式 → 立即生成所有元素
squares_list = [x ** 2 for x in range(10)]

# 生成器表达式 → 按需生成
squares_gen = (x ** 2 for x in range(10))

print(type(squares_list))  # <class 'list'>
print(type(squares_gen))   # <class 'generator'>

# 生成器表达式常用在函数参数中
total = sum(x ** 2 for x in range(1000))  # 不需要额外的括号
print(total)

max_score = max(s["score"] for s in students)
```

---

## 实用生成器模式

### 无限序列

```python
def infinite_counter(start=0, step=1):
    """无限计数器"""
    n = start
    while True:
        yield n
        n += step

# 生成前 10 个偶数
counter = infinite_counter(0, 2)
for _ in range(10):
    print(next(counter), end=" ")
# 0 2 4 6 8 10 12 14 16 18
```

### 数据管道

生成器可以链式组合，形成数据处理管道：

```python
def read_lines(filename):
    """读取文件每一行"""
    with open(filename) as f:
        for line in f:
            yield line.strip()

def filter_comments(lines):
    """过滤掉注释行"""
    for line in lines:
        if not line.startswith("#") and line:
            yield line

def parse_numbers(lines):
    """将每行转为数字"""
    for line in lines:
        try:
            yield float(line)
        except ValueError:
            continue  # 跳过无法转换的行

# 管道组合：读取 → 过滤 → 转换
# 内存中始终只有一行数据！
sample = ["# note", "1", "2.5", "bad", "4"]
numbers = parse_numbers(filter_comments(sample))
total = sum(numbers)
print(total)
```

### 批量处理

```python
def batch(iterable, size):
    """将数据分成固定大小的批次"""
    batch_data = []
    for item in iterable:
        batch_data.append(item)
        if len(batch_data) == size:
            yield batch_data
            batch_data = []
    if batch_data:  # 最后不满一批的数据
        yield batch_data

# 模拟训练数据的批量处理
data = list(range(1, 11))  # [1, 2, 3, ..., 10]

for b in batch(data, 3):
    print(f"处理批次: {b}")
# 处理批次: [1, 2, 3]
# 处理批次: [4, 5, 6]
# 处理批次: [7, 8, 9]
# 处理批次: [10]
```

---

## itertools：迭代器工具箱

Python 标准库的 `itertools` 提供了很多实用的迭代器工具：

```python
import itertools

# chain：连接多个迭代器
for item in itertools.chain([1, 2], [3, 4], [5, 6]):
    print(item, end=" ")  # 1 2 3 4 5 6

# islice：切片迭代器（对生成器很有用）
gen = (x ** 2 for x in range(100))
first_five = list(itertools.islice(gen, 5))
print(first_five)  # [0, 1, 4, 9, 16]

# zip_longest：长度不等时填充
names = ["张三", "李四", "王五"]
scores = [85, 92]
for name, score in itertools.zip_longest(names, scores, fillvalue="缺考"):
    print(f"{name}: {score}")
# 张三: 85, 李四: 92, 王五: 缺考

# product：笛卡尔积
for combo in itertools.product(["红", "蓝"], ["大", "小"]):
    print(combo)
# ('红', '大'), ('红', '小'), ('蓝', '大'), ('蓝', '小')

# count：无限计数
for i in itertools.islice(itertools.count(10, 5), 5):
    print(i, end=" ")  # 10 15 20 25 30
```

---

## 综合案例：AI 数据加载器

```python
import random

def data_loader(dataset, batch_size=32, shuffle=True):
    """
    模拟 AI 训练的数据加载器。
    用生成器实现，内存友好。
    """
    indices = list(range(len(dataset)))

    if shuffle:
        random.shuffle(indices)

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        batch_data = [dataset[i] for i in batch_indices]
        yield batch_data

# 模拟数据集
dataset = [f"sample_{i}" for i in range(100)]

# 训练循环
for epoch in range(3):
    print(f"\n=== Epoch {epoch + 1} ===")
    for batch_idx, batch in enumerate(data_loader(dataset, batch_size=32)):
        print(f"  Batch {batch_idx + 1}: {len(batch)} 个样本 "
              f"(首个: {batch[0]}, 末个: {batch[-1]})")
```

---

## 动手练习

### 练习 1：斐波那契生成器

```python
def fibonacci(n=None):
    """生成斐波那契数。n 为 None 时生成无限序列。"""
    count = 0
    a, b = 0, 1
    while n is None or count < n:
        yield a
        a, b = b, a + b
        count += 1

for num in fibonacci(10):
    print(num, end=" ")
# 0 1 1 2 3 5 8 13 21 34
```

### 练习 2：文件搜索器

```python
from pathlib import Path

def search_files(directory, pattern):
    """递归生成匹配 pattern 的文件路径。"""
    yield from Path(directory).rglob(pattern)

for filepath in search_files(".", "*.py"):
    print(filepath)
```

### 练习 3：滑动窗口

```python
def sliding_window(data, window_size):
    """生成固定大小的滑动窗口。"""
    for index in range(len(data) - window_size + 1):
        yield data[index:index + window_size]

for window in sliding_window([1, 2, 3, 4, 5], 3):
    print(window)
```

<details>
<summary>参考答案与讲解</summary>

1. `fibonacci(n)` 应该逐个 `yield` 数值，并在传入 `n` 时只生成前 `n` 个结果。示例循环应按顺序打印前十个斐波那契数。
2. `search_files` 应使用 `Path(directory).rglob(pattern)` 和 `yield from`，这样文件会以惰性方式流式返回，而不是一次性收集。
3. `sliding_window` 应生成连续的固定大小切片。如果窗口比输入还长，循环体不会执行，这就是正确的空结果。

</details>

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
pattern: class, exception, file IO, functional pipeline, generator, or type hint
code_artifact: minimal runnable example and one realistic use case
output: printed object state, caught error, saved file, yielded values, or type-check note
failure_check: hidden mutation, swallowed exception, file path issue, lazy iterator confusion, or misleading annotation
Expected_output: small advanced-Python example with a debugging note
```

## 小结

| 概念 | 说明 | 关键点 |
|------|------|--------|
| **迭代器** | 实现了 `__iter__` 和 `__next__` 的对象 | `for` 循环的底层机制 |
| **生成器函数** | 包含 `yield` 的函数 | 创建迭代器的简洁方式 |
| **生成器表达式** | `(x for x in iterable)` | 列表推导式的惰性版本 |
| **yield** | 暂停函数并返回值 | 下次调用时从暂停处继续 |
| **itertools** | 标准库的迭代器工具箱 | `chain`, `islice`, `product` 等 |

:::tip 核心理解
生成器的本质是**惰性求值（Lazy Evaluation）**——不是一次算出所有结果，而是需要一个算一个。这就像自助餐厅和外卖的区别：列表像把整桌菜一次端来（占满整张桌子），生成器像一道一道上菜（桌上永远只有一盘）。在处理大数据集和数据流时，生成器是必不可少的工具。
:::
