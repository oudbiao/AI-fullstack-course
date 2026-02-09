---
title: "1.4 函数式编程基础"
sidebar_position: 4
description: "掌握 Python 函数式编程的核心工具"
---

# 函数式编程基础

## 学习目标

- 理解函数式编程的基本思想
- 掌握 lambda 匿名函数
- 熟练使用 `map()`、`filter()`、`sorted()` 的 key 参数
- 理解闭包和装饰器的基本概念

---

## 什么是函数式编程？

简单来说，函数式编程就是**把函数当作数据来传递和使用**。

在 Python 中，函数是**一等公民**——它和数字、字符串一样，可以：
- 赋值给变量
- 作为参数传给另一个函数
- 作为返回值返回

```python
# 函数可以赋值给变量
def greet(name):
    return f"你好，{name}！"

say_hi = greet   # 把函数赋值给变量（注意没有括号）
print(say_hi("小明"))  # 你好，小明！

# 函数可以放进列表
def add(a, b): return a + b
def sub(a, b): return a - b
def mul(a, b): return a * b

operations = [add, sub, mul]
for op in operations:
    print(op(10, 3))  # 13, 7, 30
```

---

## Lambda 匿名函数

lambda 是一种**一次性的小函数**，不需要用 `def` 定义，也不需要名字。

### 基本语法

```python
# 普通函数
def square(x):
    return x ** 2

# 等价的 lambda
square = lambda x: x ** 2

print(square(5))  # 25
```

语法：`lambda 参数: 表达式`

```python
# 一个参数
double = lambda x: x * 2
print(double(5))  # 10

# 多个参数
add = lambda a, b: a + b
print(add(3, 5))  # 8

# 带条件的
grade = lambda score: "及格" if score >= 60 else "不及格"
print(grade(75))  # 及格
print(grade(45))  # 不及格
```

### lambda 的主要用途

lambda 最常见的用法是**作为参数传给其他函数**：

```python
# 场景：按特定规则排序
students = [
    {"name": "张三", "score": 85},
    {"name": "李四", "score": 92},
    {"name": "王五", "score": 78},
]

# 按成绩排序
students.sort(key=lambda s: s["score"])
print([s["name"] for s in students])  # ['王五', '张三', '李四']

# 按成绩降序
students.sort(key=lambda s: s["score"], reverse=True)
print([s["name"] for s in students])  # ['李四', '张三', '王五']
```

:::tip lambda 使用原则
- **简单逻辑**用 lambda：`lambda x: x * 2`
- **复杂逻辑**用 def：如果 lambda 写出来很长、很难读，就应该用 `def` 定义命名函数
- lambda 只能写**一个表达式**，不能写多行代码
:::

---

## map()：对每个元素做同样的操作

`map(函数, 可迭代对象)` 对序列中的**每个元素**应用函数，返回新的序列。

```python
# 把列表中的每个数字平方
numbers = [1, 2, 3, 4, 5]

# 方法 1：用 for 循环
squares = []
for n in numbers:
    squares.append(n ** 2)

# 方法 2：用 map
squares = list(map(lambda x: x ** 2, numbers))
print(squares)  # [1, 4, 9, 16, 25]

# 方法 3：用列表推导式（通常更推荐）
squares = [x ** 2 for x in numbers]
print(squares)  # [1, 4, 9, 16, 25]
```

### 实际应用

```python
# 批量转换数据类型
str_numbers = ["10", "20", "30", "40"]
numbers = list(map(int, str_numbers))
print(numbers)  # [10, 20, 30, 40]

# 批量处理字符串
names = ["  alice  ", " BOB", "charlie  "]
clean_names = list(map(str.strip, names))
print(clean_names)  # ['alice', 'BOB', 'charlie']

# 使用已有函数
temperatures_c = [0, 20, 37, 100]
def c_to_f(c):
    return c * 9/5 + 32

temperatures_f = list(map(c_to_f, temperatures_c))
print(temperatures_f)  # [32.0, 68.0, 98.6, 212.0]
```

---

## filter()：筛选满足条件的元素

`filter(函数, 可迭代对象)` 保留函数返回 `True` 的元素。

```python
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 筛选偶数
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# 等价的列表推导式
evens = [x for x in numbers if x % 2 == 0]
print(evens)  # [2, 4, 6, 8, 10]
```

### 实际应用

```python
# 筛选及格的成绩
scores = [45, 78, 55, 92, 88, 30, 67, 100]
passed = list(filter(lambda s: s >= 60, scores))
print(f"及格的: {passed}")  # [78, 92, 88, 67, 100]

# 筛选非空字符串
data = ["hello", "", "world", "", "python", ""]
non_empty = list(filter(None, data))  # filter(None, ...) 过滤掉假值
print(non_empty)  # ['hello', 'world', 'python']

# 筛选特定类型的文件
files = ["data.csv", "model.py", "readme.md", "train.py", "config.json"]
py_files = list(filter(lambda f: f.endswith(".py"), files))
print(py_files)  # ['model.py', 'train.py']
```

---

## sorted() 的 key 参数

`sorted()` 的 `key` 参数让你自定义排序规则：

```python
# 按绝对值排序
numbers = [-5, 3, -1, 4, -2]
result = sorted(numbers, key=abs)
print(result)  # [-1, -2, 3, 4, -5]

# 按字符串长度排序
words = ["python", "AI", "deep", "learning"]
result = sorted(words, key=len)
print(result)  # ['AI', 'deep', 'python', 'learning']

# 按字典的某个键排序
students = [
    {"name": "张三", "age": 20, "score": 85},
    {"name": "李四", "age": 22, "score": 92},
    {"name": "王五", "age": 19, "score": 78},
]

# 按成绩排序
by_score = sorted(students, key=lambda s: s["score"], reverse=True)
for s in by_score:
    print(f"{s['name']}: {s['score']}分")
# 李四: 92分
# 张三: 85分
# 王五: 78分

# 按多个条件排序（先按成绩降序，成绩相同按年龄升序）
students2 = [
    {"name": "A", "age": 20, "score": 85},
    {"name": "B", "age": 22, "score": 85},
    {"name": "C", "age": 19, "score": 92},
]
result = sorted(students2, key=lambda s: (-s["score"], s["age"]))
for s in result:
    print(f"{s['name']}: score={s['score']}, age={s['age']}")
# C: score=92, age=19
# A: score=85, age=20
# B: score=85, age=22
```

---

## 闭包（Closure）

闭包是一个函数，它**记住了外层函数的变量**，即使外层函数已经执行完毕。

```python
def make_multiplier(factor):
    """创建一个乘法器"""
    def multiplier(x):
        return x * factor  # factor 来自外层函数
    return multiplier

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15
print(double(10))  # 20
```

### 闭包的实际应用

```python
# 创建计数器
def make_counter(start=0):
    count = [start]   # 用列表包装，以便在内层函数中修改
    def counter():
        count[0] += 1
        return count[0]
    return counter

counter = make_counter()
print(counter())  # 1
print(counter())  # 2
print(counter())  # 3

# 创建带前缀的日志函数
def make_logger(prefix):
    def log(message):
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{prefix}] {timestamp} {message}")
    return log

info = make_logger("INFO")
error = make_logger("ERROR")

info("程序启动")      # [INFO] 14:30:01 程序启动
error("文件未找到")   # [ERROR] 14:30:01 文件未找到
```

---

## 装饰器（Decorator）

装饰器是一种**给函数添加额外功能**的优雅方式，本质上就是闭包的应用。

### 问题场景

假设你想给多个函数加上执行时间的统计：

```python
import time

# 不用装饰器的做法：每个函数都要加计时代码
def train_model():
    start = time.time()
    # ... 训练逻辑 ...
    time.sleep(1)
    end = time.time()
    print(f"train_model 耗时: {end - start:.2f}秒")

def process_data():
    start = time.time()
    # ... 处理逻辑 ...
    time.sleep(0.5)
    end = time.time()
    print(f"process_data 耗时: {end - start:.2f}秒")
```

每个函数都要重复写计时代码——太烦了！

### 装饰器解决方案

```python
import time

def timer(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"⏱ {func.__name__} 耗时: {end - start:.2f}秒")
        return result
    return wrapper

# 用 @ 语法使用装饰器
@timer
def train_model():
    """训练模型"""
    time.sleep(1)
    print("训练完成！")

@timer
def process_data(filename):
    """处理数据"""
    time.sleep(0.5)
    print(f"处理 {filename} 完成！")

train_model()
# 训练完成！
# ⏱ train_model 耗时: 1.00秒

process_data("data.csv")
# 处理 data.csv 完成！
# ⏱ process_data 耗时: 0.50秒
```

`@timer` 等价于 `train_model = timer(train_model)`。

### 常用的装饰器模式

```python
# 重试装饰器
def retry(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"第 {attempt} 次尝试失败: {e}")
                    if attempt == max_attempts:
                        raise
        return wrapper
    return decorator

@retry(max_attempts=3)
def risky_operation():
    import random
    if random.random() < 0.7:
        raise ConnectionError("连接失败")
    return "成功！"
```

---

## map / filter vs 列表推导式

| 方式 | 适用场景 | 示例 |
|------|---------|------|
| 列表推导式 | **大多数情况**（推荐） | `[x**2 for x in nums]` |
| `map()` | 已有函数可以直接用 | `list(map(int, strings))` |
| `filter()` | 配合已有判断函数 | `list(filter(str.isdigit, items))` |

```python
# 当已经有现成函数时，map 更简洁
numbers = ["1", "2", "3"]
list(map(int, numbers))        # 简洁
[int(x) for x in numbers]     # 也行，但稍长

# 当需要变换+条件时，列表推导式更清晰
[x**2 for x in range(10) if x % 2 == 0]
# 比 list(filter(lambda x: x%2==0, map(lambda x: x**2, range(10)))) 清晰得多
```

---

## 动手练习

### 练习 1：数据处理管道

```python
# 用 map 和 filter 处理以下数据
raw_data = ["  23  ", "abc", "45.6", "", "78", "not_a_number", "90.1"]

# 1. 去除空白
# 2. 过滤掉无法转换为数字的字符串
# 3. 转换为浮点数
# 4. 过滤掉小于 50 的数
# 提示：可以结合使用 map、filter 和列表推导式
```

### 练习 2：自定义排序

```python
products = [
    {"name": "笔记本", "price": 5999, "rating": 4.5},
    {"name": "鼠标", "price": 199, "rating": 4.8},
    {"name": "键盘", "price": 599, "rating": 4.2},
    {"name": "显示器", "price": 2999, "rating": 4.7},
]

# 1. 按价格从低到高排序
# 2. 按评分从高到低排序
# 3. 按性价比（rating/price）从高到低排序
```

### 练习 3：写一个装饰器

写一个 `@log` 装饰器，在函数执行前后打印日志：

```python
@log
def add(a, b):
    return a + b

add(3, 5)
# 应该输出:
# 调用 add，参数: (3, 5) {}
# add 返回: 8
```

---

## 小结

| 概念 | 说明 | 示例 |
|------|------|------|
| **lambda** | 匿名函数 | `lambda x: x * 2` |
| **map()** | 对每个元素应用函数 | `map(int, ["1", "2"])` |
| **filter()** | 筛选满足条件的元素 | `filter(lambda x: x>0, nums)` |
| **sorted(key=)** | 自定义排序 | `sorted(data, key=lambda x: x["score"])` |
| **闭包** | 函数记住外层变量 | 工厂函数模式 |
| **装饰器** | 给函数添加额外功能 | `@timer` |

:::tip 核心理解
函数式编程的核心是**把函数当作数据**——你可以把函数存起来、传来传去、组合使用。这种思维在数据处理中特别有用，因为你经常需要对一组数据做"变换 → 筛选 → 排序"的操作链。不需要完全掌握函数式编程，但 lambda、map/filter、装饰器这几个工具一定要会用。
:::
