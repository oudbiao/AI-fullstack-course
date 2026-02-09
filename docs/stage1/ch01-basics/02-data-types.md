---
title: "1.2 数据类型与变量"
sidebar_position: 2
description: "掌握 Python 中的基本数据类型和变量使用"
---

# 数据类型与变量

## 学习目标

- 理解什么是变量，掌握变量的命名规则
- 掌握 Python 的基本数据类型：整数、浮点数、字符串、布尔值
- 学会数据类型之间的转换
- 理解动态类型的含义

---

## 什么是变量？

想象变量是一个**贴了标签的盒子**。你可以把东西放进去，用标签来找到它。

```python
name = "小明"       # 盒子上贴了"name"标签，里面放了"小明"
age = 20           # 盒子上贴了"age"标签，里面放了 20
height = 1.75      # 盒子上贴了"height"标签，里面放了 1.75
```

`=` 在 Python 里不是"等于"，而是**赋值**——把右边的值放进左边的盒子里。

```python
# 赋值的方向：从右到左
x = 10      # 把 10 放进 x 这个盒子

# 可以修改盒子里的内容
x = 20      # 现在 x 里面是 20 了（10 被丢掉了）

# 可以用变量的值来计算
y = x + 5   # y = 20 + 5 = 25
print(y)    # 输出: 25
```

### 变量命名规则

Python 对变量名有一些规定：

| 规则 | 正确示例 | 错误示例 |
|------|---------|---------|
| 只能包含字母、数字、下划线 | `user_name`, `age2` | `user-name`, `age!` |
| 不能以数字开头 | `name1` | `1name` |
| 不能用 Python 关键字 | `my_class` | `class`, `if`, `for` |
| 大小写敏感 | `Name` 和 `name` 是不同变量 | — |

### 命名惯例（不是强制的，但大家都这么做）

```python
# 好的命名 ✅ —— 用小写字母加下划线（snake_case）
student_name = "小明"
learning_rate = 0.001
max_epochs = 100

# 不好的命名 ❌ —— 不是不能用，而是不够清晰
a = "小明"        # 看不出来 a 是什么
x1 = 0.001       # x1 代表什么？
SN = "小明"       # 缩写太短，别人看不懂
```

:::tip 命名的黄金法则
变量名应该让人**一眼就知道它是什么**。宁可名字长一点（`student_count`），也不要用看不懂的缩写（`sc`）。
:::

---

## 数字类型

### 整数（int）

整数就是没有小数点的数字，可以是正数、负数或零。

```python
age = 25
temperature = -10
count = 0
big_number = 1_000_000  # 下划线分隔，方便阅读，等同于 1000000

print(type(age))  # <class 'int'>
```

:::info type() 函数
`type()` 可以查看任何值的类型，学习阶段经常会用到，帮你确认变量的类型。
:::

Python 的整数没有大小限制（不像 C/Java 有 int 范围限制）：

```python
huge = 99999999999999999999999999999999
print(huge + 1)  # 完全没问题
```

### 浮点数（float）

浮点数就是带小数点的数字。

```python
pi = 3.14159
weight = 65.5
negative = -0.001

print(type(pi))  # <class 'float'>
```

**注意浮点数的精度问题**——这是所有编程语言都有的问题：

```python
>>> 0.1 + 0.2
0.30000000000000004    # 不是精确的 0.3！
```

这不是 Python 的 bug，而是计算机用二进制存储小数时的固有问题。在 AI 开发中，这个微小的误差通常不影响结果。但如果你做金融计算需要精确结果，可以用 `decimal` 模块。

### 整数和浮点数的运算

```python
a = 10
b = 3

print(a + b)    # 13    加法
print(a - b)    # 7     减法
print(a * b)    # 30    乘法
print(a / b)    # 3.333... 除法（结果总是 float）
print(a // b)   # 3     整除（向下取整）
print(a % b)    # 1     取余数
print(a ** b)   # 1000  幂运算（10 的 3 次方）
```

一个常见的坑：

```python
# 除法 / 的结果永远是 float，即使能整除
>>> 10 / 2
5.0         # 不是 5，而是 5.0

# 如果想得到整数结果，用 //
>>> 10 // 2
5
```

---

## 字符串（str）

字符串就是**文本**——一串字符。用引号包裹。

### 创建字符串

```python
# 单引号和双引号都可以，效果一样
name = '小明'
greeting = "你好"

# 如果字符串里有引号，用另一种引号包裹
sentence = "他说：'你好'"
sentence2 = '英文名叫 "Tom"'

# 三引号：可以写多行文本
poem = """
静夜思
床前明月光，
疑是地上霜。
"""
print(poem)

print(type(name))  # <class 'str'>
```

### 字符串拼接

```python
first_name = "张"
last_name = "三"

# 方法 1：用 + 拼接
full_name = first_name + last_name
print(full_name)  # 张三

# 方法 2：用 f-string（推荐！Python 3.6+）
age = 20
intro = f"我叫{full_name}，今年{age}岁"
print(intro)  # 我叫张三，今年20岁

# 方法 3：用 format()
intro2 = "我叫{}，今年{}岁".format(full_name, age)
print(intro2)  # 我叫张三，今年20岁
```

:::tip f-string 是最佳实践
f-string（`f"...{变量}..."`）是现代 Python 中最常用的字符串格式化方式，简洁直观。后面的课程中我们会大量使用它。
:::

### 常用字符串操作

```python
text = "Hello, Python!"

# 获取长度
print(len(text))         # 14

# 大小写转换
print(text.upper())      # HELLO, PYTHON!
print(text.lower())      # hello, python!

# 查找子字符串
print(text.find("Python"))  # 7（从第 7 个位置开始）
print("Python" in text)     # True

# 替换
print(text.replace("Python", "AI"))  # Hello, AI!

# 去除首尾空白
messy = "  hello  "
print(messy.strip())    # "hello"

# 分割
csv_line = "张三,20,北京"
parts = csv_line.split(",")
print(parts)  # ['张三', '20', '北京']
```

### 字符串索引和切片

字符串里的每个字符都有一个**位置编号（索引）**，从 0 开始：

```python
text = "Python"
#       P y t h o n
# 索引: 0 1 2 3 4 5
# 负索引: -6 -5 -4 -3 -2 -1

print(text[0])    # P（第一个字符）
print(text[5])    # n（第六个字符）
print(text[-1])   # n（最后一个字符）
print(text[-2])   # o（倒数第二个字符）
```

**切片**可以取出一段子字符串：

```python
text = "Python"

print(text[0:3])   # Pyt（从索引0到索引3，不包含3）
print(text[2:5])   # tho
print(text[:3])    # Pyt（从头开始，可以省略 0）
print(text[3:])    # hon（到末尾，可以省略结束位置）
print(text[:])     # Python（整个字符串的拷贝）
print(text[::2])   # Pto（每隔一个字符取一个）
print(text[::-1])  # nohtyP（反转字符串！）
```

:::info 切片语法
`text[start:stop:step]` —— 从 `start` 开始，到 `stop` 结束（不包含），每 `step` 个取一个。记住：**左闭右开**（包含起始，不包含结束）。
:::

### 字符串是不可变的

```python
text = "Hello"
# text[0] = "h"  # 报错！TypeError: 'str' object does not support item assignment

# 如果想修改，需要创建新字符串
text = "h" + text[1:]  # "hello"
```

---

## 布尔值（bool）

布尔值只有两个：`True`（真）和 `False`（假）。注意首字母大写。

```python
is_student = True
is_raining = False

print(type(is_student))  # <class 'bool'>
```

布尔值通常来自**比较运算**：

```python
print(5 > 3)       # True
print(5 < 3)       # False
print(5 == 5)      # True（注意是两个等号，一个等号是赋值）
print(5 != 3)      # True（!= 表示不等于）
print("abc" == "abc")  # True
```

布尔值在后面学习流程控制（if/else）时会大量使用。

### 真值和假值

在 Python 中，很多东西可以被当作布尔值使用。以下值被认为是"假"：

```python
# 以下都是 "假值"（Falsy）
bool(0)        # False
bool(0.0)      # False
bool("")       # False（空字符串）
bool([])       # False（空列表）
bool(None)     # False

# 其他都是 "真值"（Truthy）
bool(1)        # True
bool(-1)       # True（非零就是真）
bool("hello")  # True（非空字符串就是真）
bool([1, 2])   # True（非空列表就是真）
```

---

## None 类型

`None` 是 Python 中的特殊值，表示**"什么都没有"**。

```python
result = None
print(result)        # None
print(type(result))  # <class 'NoneType'>
```

`None` 常用来表示"还没有值"或"没有结果"：

```python
# 函数没有返回值时，默认返回 None
def say_hello():
    print("Hello!")

result = say_hello()   # 打印 Hello!
print(result)          # None
```

---

## 类型转换

有时候你需要把一种类型转换成另一种。

```python
# 字符串 → 数字
age_str = "25"
age = int(age_str)      # 字符串转整数
print(age + 1)          # 26

price_str = "99.9"
price = float(price_str)  # 字符串转浮点数
print(price)             # 99.9

# 数字 → 字符串
score = 95
score_str = str(score)   # 整数转字符串
print("分数: " + score_str)  # 分数: 95

# 整数 ↔ 浮点数
x = int(3.7)    # 3（直接截断小数部分，不是四舍五入）
y = float(5)    # 5.0
```

**常见错误**：字符串和数字不能直接用 `+` 拼接

```python
age = 20
# print("年龄: " + age)  # 报错！TypeError

# 正确做法 1：转成字符串
print("年龄: " + str(age))

# 正确做法 2：用 f-string（推荐）
print(f"年龄: {age}")

# 正确做法 3：用逗号分隔（print 会自动加空格）
print("年龄:", age)
```

### 类型转换速查表

| 转换 | 函数 | 示例 | 结果 |
|------|------|------|------|
| → 整数 | `int()` | `int("25")` | `25` |
| → 浮点数 | `float()` | `float("3.14")` | `3.14` |
| → 字符串 | `str()` | `str(100)` | `"100"` |
| → 布尔值 | `bool()` | `bool(0)` | `False` |

---

## 动态类型

Python 是**动态类型**语言——变量不需要提前声明类型，而且同一个变量可以随时换类型。

```python
x = 10          # x 是整数
print(type(x))  # <class 'int'>

x = "hello"     # 现在 x 变成字符串了
print(type(x))  # <class 'str'>

x = True        # 现在 x 又变成布尔值了
print(type(x))  # <class 'bool'>
```

这很灵活，但也要小心——别把一个本来存数字的变量意外改成了字符串。

对比一下 Java（静态类型语言）：

```java
int x = 10;       // 声明 x 是整数
x = "hello";      // 报错！Java 不允许改变类型
```

---

## 多重赋值

Python 支持一些便捷的赋值方式：

```python
# 同时给多个变量赋值
a, b, c = 1, 2, 3
print(a, b, c)  # 1 2 3

# 交换两个变量的值（Python 独有的简洁写法）
a, b = b, a
print(a, b)  # 2 1

# 多个变量赋同一个值
x = y = z = 0
print(x, y, z)  # 0 0 0
```

交换变量这个写法非常 Pythonic（Python 风格），在其他语言里通常需要一个临时变量：

```python
# 其他语言的写法
temp = a
a = b
b = temp

# Python 的写法
a, b = b, a  # 一行搞定！
```

---

## 动手练习

### 练习 1：个人信息卡

创建变量存储你的信息，然后用 f-string 打印出来：

```python
name = "你的名字"
age = 25
city = "你的城市"
is_student = True

print(f"姓名: {name}")
print(f"年龄: {age}")
print(f"城市: {city}")
print(f"是否学生: {is_student}")
print(f"明年我就 {age + 1} 岁了")
```

### 练习 2：温度转换器

摄氏度转华氏度的公式：`F = C × 9/5 + 32`

```python
celsius = 37.5
fahrenheit = celsius * 9 / 5 + 32
print(f"{celsius}°C = {fahrenheit}°F")
```

试着修改 `celsius` 的值，算几个不同温度的结果。

### 练习 3：字符串操作

```python
email = "  ZhangSan@Example.COM  "

# 1. 去掉首尾空格
# 2. 转成小写
# 3. 找出 @ 的位置
# 4. 取出用户名部分（@ 前面的部分）
```

提示：可以组合使用 `.strip()`、`.lower()`、`.find()`、切片。

### 练习 4：类型侦探

用 `type()` 检查以下每个值的类型，先猜再验证：

```python
print(type(42))
print(type(3.14))
print(type("3.14"))
print(type(True))
print(type(None))
print(type(1 + 2))
print(type(1 + 2.0))    # 整数 + 浮点数 = ？
print(type("1" + "2"))  # 字符串 + 字符串 = ？
```

---

## 小结

| 类型 | 关键字 | 示例 | 用途 |
|------|--------|------|------|
| **整数** | `int` | `42`, `-10`, `0` | 计数、索引 |
| **浮点数** | `float` | `3.14`, `-0.5` | 精确数值、科学计算 |
| **字符串** | `str` | `"hello"`, `'world'` | 文本数据 |
| **布尔值** | `bool` | `True`, `False` | 条件判断 |
| **空值** | `NoneType` | `None` | 表示"没有值" |

:::tip 核心理解
Python 中**一切皆对象**。数字是对象，字符串是对象，甚至 `True` 和 `None` 也是对象。每个对象都有一个类型（`type`），类型决定了你能对它做什么操作。
:::
