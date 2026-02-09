---
title: "1.7 函数基础"
sidebar_position: 7
description: "掌握函数的定义、参数、返回值和作用域"
---

# 函数基础

## 学习目标

- 理解函数是什么，为什么需要函数
- 掌握函数的定义和调用
- 理解参数（位置参数、默认参数、关键字参数）
- 掌握返回值的使用
- 理解变量作用域

---

## 为什么需要函数？

假设你在写一个数据处理脚本，需要多次计算平均值：

```python
# 第一次计算
scores1 = [85, 92, 78, 95, 88]
total1 = sum(scores1)
avg1 = total1 / len(scores1)
print(f"平均分: {avg1:.1f}")

# 第二次计算（又写一遍一模一样的逻辑）
scores2 = [90, 85, 92, 88, 95, 87]
total2 = sum(scores2)
avg2 = total2 / len(scores2)
print(f"平均分: {avg2:.1f}")

# 第三次计算（再写一遍……）
scores3 = [75, 80, 68, 72, 88]
total3 = sum(scores3)
avg3 = total3 / len(scores3)
print(f"平均分: {avg3:.1f}")
```

同样的逻辑写了 3 遍——如果以后要改计算方式（比如去掉最高最低分），你得改 3 个地方。

用函数解决：

```python
def calculate_average(scores):
    """计算平均分"""
    return sum(scores) / len(scores)

# 现在一行搞定
print(f"平均分: {calculate_average([85, 92, 78, 95, 88]):.1f}")
print(f"平均分: {calculate_average([90, 85, 92, 88, 95, 87]):.1f}")
print(f"平均分: {calculate_average([75, 80, 68, 72, 88]):.1f}")
```

**函数的核心价值：**

| 好处 | 说明 |
|------|------|
| **复用** | 写一次，用多次 |
| **抽象** | 把复杂逻辑藏在函数名后面，调用时只需要知道"做什么"，不用管"怎么做" |
| **维护** | 要改逻辑只改一个地方 |
| **可读** | 函数名就是注释，`calculate_average(scores)` 一目了然 |

---

## 定义和调用函数

### 基本语法

```python
def greet(name):
    """向某人打招呼"""  # 文档字符串（docstring），描述函数做什么
    print(f"你好，{name}！欢迎学习 Python！")

# 调用函数
greet("小明")     # 你好，小明！欢迎学习 Python！
greet("小红")     # 你好，小红！欢迎学习 Python！
```

语法解读：
- `def` 关键字表示"定义一个函数"
- `greet` 是函数名（命名规则和变量一样，小写加下划线）
- `(name)` 是参数列表
- `:` 冒号不能忘
- 函数体需要缩进
- `"""..."""` 是文档字符串，描述函数的功能

### 没有参数的函数

```python
def say_hello():
    print("Hello, World!")

say_hello()  # Hello, World!
```

### 多个参数的函数

```python
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")

add(3, 5)    # 3 + 5 = 8
add(10, 20)  # 10 + 20 = 30
```

---

## 返回值

函数可以用 `return` 把结果返回给调用者：

```python
def add(a, b):
    return a + b

# 函数的返回值可以赋给变量
result = add(3, 5)
print(result)       # 8

# 也可以直接使用返回值
print(add(10, 20))  # 30

# 在表达式中使用
total = add(1, 2) + add(3, 4)
print(total)  # 10
```

### 返回多个值

```python
def get_min_max(numbers):
    """返回列表中的最小值和最大值"""
    return min(numbers), max(numbers)

# 用元组解包接收
smallest, largest = get_min_max([3, 1, 4, 1, 5, 9])
print(f"最小值: {smallest}, 最大值: {largest}")
# 最小值: 1, 最大值: 9
```

### 没有 return 的函数

如果函数没有 `return` 语句，或者 `return` 后面没有值，函数返回 `None`：

```python
def greet(name):
    print(f"你好，{name}！")
    # 没有 return

result = greet("小明")   # 打印: 你好，小明！
print(result)            # None
```

### return 的另一个用途：提前结束函数

```python
def divide(a, b):
    if b == 0:
        print("错误：除数不能为 0！")
        return None   # 提前结束函数
    return a / b

print(divide(10, 3))   # 3.333...
print(divide(10, 0))   # 错误：除数不能为 0！ 然后返回 None
```

---

## 参数详解

### 位置参数

按照顺序传入的参数：

```python
def describe_pet(animal, name):
    print(f"我有一只{animal}，名叫{name}")

describe_pet("猫", "咪咪")   # 我有一只猫，名叫咪咪
describe_pet("咪咪", "猫")   # 我有一只咪咪，名叫猫 —— 顺序错了！
```

### 关键字参数

通过参数名传值，不用在乎顺序：

```python
def describe_pet(animal, name):
    print(f"我有一只{animal}，名叫{name}")

# 用关键字参数，顺序无所谓
describe_pet(name="咪咪", animal="猫")   # 我有一只猫，名叫咪咪
describe_pet(animal="狗", name="旺财")   # 我有一只狗，名叫旺财
```

### 默认参数

给参数一个默认值，调用时可以不传：

```python
def train_model(epochs=10, lr=0.001, batch_size=32):
    print(f"训练参数: epochs={epochs}, lr={lr}, batch_size={batch_size}")

# 使用全部默认值
train_model()
# 训练参数: epochs=10, lr=0.001, batch_size=32

# 只修改部分参数
train_model(epochs=50)
# 训练参数: epochs=50, lr=0.001, batch_size=32

train_model(epochs=100, lr=0.01)
# 训练参数: epochs=100, lr=0.01, batch_size=32
```

:::caution 默认参数的陷阱
默认参数的值在函数定义时就确定了。不要用可变对象（如列表、字典）作为默认值：

```python
# 错误做法 ❌
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] —— 出bug了！上次的 'a' 还在

# 正确做法 ✅
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```
:::

### *args：接收任意数量的位置参数

```python
def calculate_sum(*numbers):
    """计算任意数量数字的和"""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2))           # 3
print(calculate_sum(1, 2, 3, 4, 5))  # 15
print(calculate_sum(10))             # 10
```

### **kwargs：接收任意数量的关键字参数

```python
def print_info(**info):
    """打印任意数量的信息"""
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="小明", age=20, city="北京")
# name: 小明
# age: 20
# city: 北京
```

### 参数顺序规则

当多种参数混合使用时，顺序是：

```python
def func(pos_arg, default_arg=10, *args, **kwargs):
    print(f"pos_arg={pos_arg}")
    print(f"default_arg={default_arg}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, name="test")
# pos_arg=1
# default_arg=2
# args=(3, 4)
# kwargs={'name': 'test'}
```

---

## 变量作用域

变量的"作用域"就是它的**生效范围**。

### 局部变量 vs 全局变量

```python
# 全局变量：定义在函数外面
message = "我是全局变量"

def my_function():
    # 局部变量：定义在函数里面
    local_var = "我是局部变量"
    print(message)      # 可以读取全局变量
    print(local_var)    # 可以读取局部变量

my_function()
print(message)          # 可以访问全局变量
# print(local_var)      # 报错！局部变量在函数外不存在
```

### 同名变量

```python
x = 10  # 全局变量

def my_function():
    x = 20  # 这是一个新的局部变量，不是修改全局变量
    print(f"函数内的 x: {x}")  # 20

my_function()
print(f"函数外的 x: {x}")    # 10（全局变量没有被修改）
```

### global 关键字

如果你确实需要在函数内修改全局变量（一般不推荐）：

```python
count = 0

def increment():
    global count   # 声明要使用全局变量 count
    count += 1

increment()
increment()
increment()
print(count)  # 3
```

:::tip 最佳实践
尽量**不要使用全局变量**。函数应该通过参数接收数据，通过返回值传出结果。这样的函数更容易测试、更容易理解。
:::

---

## 文档字符串（docstring）

好的函数应该有清晰的文档说明：

```python
def calculate_bmi(weight, height):
    """
    计算身体质量指数（BMI）。

    参数:
        weight (float): 体重，单位千克
        height (float): 身高，单位米

    返回:
        float: BMI 值

    示例:
        >>> calculate_bmi(70, 1.75)
        22.857142857142858
    """
    return weight / (height ** 2)

# 查看函数的文档
help(calculate_bmi)
```

---

## 综合案例

### 案例 1：成绩分析工具

```python
def analyze_scores(scores, subject="未知科目"):
    """
    分析一组成绩，返回统计信息。

    参数:
        scores: 成绩列表
        subject: 科目名称
    返回:
        字典，包含统计信息
    """
    if not scores:
        return {"error": "成绩列表为空"}

    avg = sum(scores) / len(scores)
    passed = [s for s in scores if s >= 60]
    failed = [s for s in scores if s < 60]

    return {
        "subject": subject,
        "count": len(scores),
        "average": round(avg, 1),
        "max": max(scores),
        "min": min(scores),
        "pass_rate": f"{len(passed) / len(scores):.1%}",
        "passed": len(passed),
        "failed": len(failed)
    }

def print_report(stats):
    """打印格式化的成绩报告"""
    print(f"\n{'='*30}")
    print(f"  {stats['subject']} 成绩分析报告")
    print(f"{'='*30}")
    print(f"  参加人数: {stats['count']}")
    print(f"  平均分:   {stats['average']}")
    print(f"  最高分:   {stats['max']}")
    print(f"  最低分:   {stats['min']}")
    print(f"  及格率:   {stats['pass_rate']}")
    print(f"  及格人数: {stats['passed']}")
    print(f"  不及格:   {stats['failed']}")
    print(f"{'='*30}")

# 使用
math_scores = [85, 92, 45, 78, 95, 55, 88, 72, 60, 98]
english_scores = [70, 55, 88, 45, 92, 78, 65, 82, 90, 58]

math_stats = analyze_scores(math_scores, "数学")
english_stats = analyze_scores(english_scores, "英语")

print_report(math_stats)
print_report(english_stats)
```

### 案例 2：简单的密码生成器

```python
import random
import string

def generate_password(length=12, use_upper=True, use_digits=True, use_special=True):
    """
    生成随机密码。

    参数:
        length: 密码长度，默认 12
        use_upper: 是否包含大写字母
        use_digits: 是否包含数字
        use_special: 是否包含特殊字符
    """
    chars = string.ascii_lowercase  # 小写字母

    if use_upper:
        chars += string.ascii_uppercase
    if use_digits:
        chars += string.digits
    if use_special:
        chars += "!@#$%^&*"

    password = ''.join(random.choice(chars) for _ in range(length))
    return password

# 生成不同类型的密码
print(f"默认密码: {generate_password()}")
print(f"纯字母:   {generate_password(length=8, use_digits=False, use_special=False)}")
print(f"超强密码: {generate_password(length=20)}")
```

---

## 动手练习

### 练习 1：温度转换函数

写两个函数，实现摄氏度和华氏度的互相转换：

```python
def celsius_to_fahrenheit(celsius):
    """摄氏度 → 华氏度: F = C × 9/5 + 32"""
    pass  # 补充代码

def fahrenheit_to_celsius(fahrenheit):
    """华氏度 → 摄氏度: C = (F - 32) × 5/9"""
    pass  # 补充代码

# 测试
print(celsius_to_fahrenheit(100))  # 应该输出 212.0
print(fahrenheit_to_celsius(32))   # 应该输出 0.0
```

### 练习 2：列表统计函数

写一个函数，接收一个数字列表，返回最大值、最小值、平均值、中位数：

```python
def list_stats(numbers):
    """
    返回列表的统计信息。
    不要使用 max()、min()、sum() 内置函数，自己实现！
    """
    pass  # 补充代码

# 测试
stats = list_stats([3, 1, 4, 1, 5, 9, 2, 6, 5])
print(stats)
```

### 练习 3：猜数字游戏（函数版）

把之前的猜数字游戏改写成函数版本：

```python
def guess_number_game(min_val=1, max_val=100, max_attempts=7):
    """猜数字游戏"""
    pass  # 补充代码

# 运行游戏
guess_number_game()
guess_number_game(1, 50, 5)  # 范围更小，次数更少
```

---

## 小结

| 概念 | 说明 | 示例 |
|------|------|------|
| **定义函数** | `def 函数名(参数):` | `def add(a, b):` |
| **返回值** | `return 值` | `return a + b` |
| **默认参数** | 参数有默认值 | `def f(x=10):` |
| **关键字参数** | 按名字传参 | `f(x=5, y=10)` |
| **`*args`** | 接收任意位置参数 | `def f(*args):` |
| **`**kwargs`** | 接收任意关键字参数 | `def f(**kwargs):` |
| **局部变量** | 函数内定义，函数外不可用 | — |
| **全局变量** | 函数外定义，函数内可读 | — |

:::tip 核心理解
函数是编程中的**基本积木**。好的代码应该由一个个小函数组成，每个函数只做一件事，做好一件事。如果你的函数超过 20 行，考虑把它拆成更小的函数。
:::
