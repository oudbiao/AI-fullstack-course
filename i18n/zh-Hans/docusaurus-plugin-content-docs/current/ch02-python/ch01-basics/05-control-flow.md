---
title: "2.1.5 流程控制"
sidebar_position: 5
description: "掌握条件判断和循环结构"
---

# 2.1.5 流程控制

![Python 流程控制执行路径图](/img/course/ch02-control-flow-paths.webp)

## 本节定位

这一节学习让程序“做判断”和“重复执行”。条件判断和循环是所有自动化脚本、数据处理流程和模型训练代码的骨架，掌握它们后，你的代码就不再只是从上到下执行。

## 学习目标

- 掌握 `if/elif/else` 条件判断
- 掌握 `for` 循环和 `while` 循环
- 学会使用 `break`、`continue` 控制循环
- 能编写包含嵌套逻辑的程序

---

## 什么是流程控制？

到目前为止，你写的代码都是**从上到下一行一行执行**的。但真实的程序需要做判断、需要重复执行——这就是流程控制。

想象你早上出门的决策过程：

```
如果 下雨了:
    带伞
否则如果 太阳很大:
    戴帽子
否则:
    直接出门
```

这就是**条件判断**。

再想象你背单词：

```
重复 100 次:
    看一个新单词
    记住它
```

这就是**循环**。

---

## 条件判断：if / elif / else

### 基本的 if

```python
temperature = 35

if temperature > 30:
    print("今天很热，注意防暑！")
```

**语法规则：**
1. `if` 后面跟条件表达式
2. 条件后面有一个**冒号 `:`**（很多新手会忘记）
3. 条件成立时执行的代码需要**缩进 4 个空格**

### if...else

```python
age = 15

if age >= 18:
    print("你已经成年了")
    print("可以看这部电影")
else:
    print("你还未成年")
    print("需要家长陪同")
```

### if...elif...else

`elif` 是 "else if" 的缩写，用来检查多个条件：

```python
score = 85

if score >= 90:
    grade = "A（优秀）"
elif score >= 80:
    grade = "B（良好）"
elif score >= 70:
    grade = "C（中等）"
elif score >= 60:
    grade = "D（及格）"
else:
    grade = "F（不及格）"

print(f"你的成绩: {score} 分，等级: {grade}")
# 输出: 你的成绩: 85 分，等级: B（良好）
```

:::caution 注意执行顺序
Python 从上到下依次检查每个条件，**一旦某个条件成立，就执行对应的代码块，然后跳过剩余所有的 elif 和 else**。所以条件的顺序很重要！

```python
score = 95

# 错误的顺序 ❌
if score >= 60:
    print("及格")      # 95 >= 60 成立，直接执行这个
elif score >= 90:
    print("优秀")      # 不会被执行！

# 正确的顺序 ✅：从严到宽
if score >= 90:
    print("优秀")      # 95 >= 90 成立，执行这个
elif score >= 60:
    print("及格")
```
:::

### 条件判断的简写

```python
# 三元表达式（一行搞定简单的 if-else）
age = 20
status = "成年" if age >= 18 else "未成年"
print(status)  # 成年

# 等价于
if age >= 18:
    status = "成年"
else:
    status = "未成年"
```

### 嵌套的 if

条件里面可以再套条件：

```python
has_ticket = True
age = 15

if has_ticket:
    if age >= 18:
        print("请入场")
    else:
        print("未成年人需要家长陪同")
else:
    print("请先购票")
```

不过嵌套太多层会让代码难以阅读，通常不建议超过 3 层。

---

## for 循环

`for` 循环用来**遍历**一个序列（列表、字符串、范围等）中的每个元素。

### 遍历列表

```python
fruits = ["苹果", "香蕉", "橙子", "葡萄"]

for fruit in fruits:
    print(f"我喜欢吃{fruit}")

# 输出:
# 我喜欢吃苹果
# 我喜欢吃香蕉
# 我喜欢吃橙子
# 我喜欢吃葡萄
```

理解方式：`for fruit in fruits` 的意思是"对于 fruits 中的每一个 fruit，执行下面的代码"。

### 遍历字符串

```python
word = "Python"

for char in word:
    print(char, end=" ")

# 输出: P y t h o n
```

### range() 函数

`range()` 生成一个数字序列，是 `for` 循环最常用的搭档：

```python
# range(5) 生成 0, 1, 2, 3, 4
for i in range(5):
    print(i, end=" ")
# 输出: 0 1 2 3 4

# range(start, stop) 从 start 到 stop-1
for i in range(1, 6):
    print(i, end=" ")
# 输出: 1 2 3 4 5

# range(start, stop, step) 带步长
for i in range(0, 10, 2):
    print(i, end=" ")
# 输出: 0 2 4 6 8

# 倒数
for i in range(5, 0, -1):
    print(i, end=" ")
# 输出: 5 4 3 2 1
```

### 实际案例：计算 1 到 100 的和

```python
total = 0
for i in range(1, 101):
    total += i
print(f"1 到 100 的和是: {total}")  # 5050
```

### enumerate()：同时获取索引和值

```python
students = ["张三", "李四", "王五"]

# 普通写法
for i in range(len(students)):
    print(f"第 {i+1} 名: {students[i]}")

# 更 Pythonic 的写法：用 enumerate
for i, name in enumerate(students):
    print(f"第 {i+1} 名: {name}")

# 指定起始编号
for i, name in enumerate(students, start=1):
    print(f"第 {i} 名: {name}")
```

---

## while 循环

`while` 循环在**条件成立时**持续执行，直到条件不成立为止。

### 基本用法

```python
count = 0

while count < 5:
    print(f"当前计数: {count}")
    count += 1   # 别忘了更新条件！

print("循环结束")

# 输出:
# 当前计数: 0
# 当前计数: 1
# 当前计数: 2
# 当前计数: 3
# 当前计数: 4
# 循环结束
```

:::caution 小心死循环！
如果忘了更新条件变量，循环永远不会停止：

```python
# 死循环示例（不要运行！）
count = 0
while count < 5:
    print(count)
    # 忘了 count += 1，count 永远是 0，循环永不结束
```

如果不小心陷入死循环，按 `Ctrl+C` 强制终止程序。
:::

### while 的典型场景

`while` 适合**不确定循环次数**的情况：

```python
# 场景：猜数字游戏
import random

target = random.randint(1, 100)
guess = 0
attempts = 0

print("我想了一个 1 到 100 的数字，猜猜看！")

while guess != target:
    guess = int(input("你的猜测: "))
    attempts += 1

    if guess < target:
        print("太小了！")
    elif guess > target:
        print("太大了！")
    else:
        print(f"恭喜你猜对了！用了 {attempts} 次")
```

### for vs while 怎么选？

| 场景 | 推荐 | 原因 |
|------|------|------|
| 遍历列表/字符串 | `for` | 天然适合 |
| 循环固定次数 | `for + range()` | 简洁明确 |
| 不确定循环次数 | `while` | 灵活控制 |
| 等待某个条件成立 | `while` | 自然直觉 |

**经验法则：能用 `for` 就用 `for`，它更安全（不会死循环）。**

---

## break 和 continue

### break：立即终止循环

```python
# 找到第一个偶数就停下
numbers = [1, 3, 7, 4, 9, 2]

for num in numbers:
    if num % 2 == 0:
        print(f"找到第一个偶数: {num}")
        break
    print(f"{num} 不是偶数，继续找...")

# 输出:
# 1 不是偶数，继续找...
# 3 不是偶数，继续找...
# 7 不是偶数，继续找...
# 找到第一个偶数: 4
```

### continue：跳过本次循环，继续下一次

```python
# 打印所有奇数，跳过偶数
for i in range(1, 11):
    if i % 2 == 0:
        continue   # 跳过偶数
    print(i, end=" ")

# 输出: 1 3 5 7 9
```

### break 和 continue 的区别

```python
# break：直接离开循环
for i in range(10):
    if i == 5:
        break       # 循环到 5 就整个停了
    print(i, end=" ")
# 输出: 0 1 2 3 4

# continue：跳过当前，继续下一个
for i in range(10):
    if i == 5:
        continue    # 跳过 5，继续 6, 7, 8, 9
    print(i, end=" ")
# 输出: 0 1 2 3 4 6 7 8 9
```

---

## 循环中的 else

Python 的循环有一个独特的 `else` 子句——当循环**正常结束**（不是被 `break` 终止）时执行：

```python
# 检查一个数是否是质数
num = 17

for i in range(2, num):
    if num % i == 0:
        print(f"{num} 不是质数，可以被 {i} 整除")
        break
else:
    # 循环没有被 break 终止，说明没有找到因子
    print(f"{num} 是质数")

# 输出: 17 是质数
```

---

## 嵌套循环

循环里面可以再套循环：

```python
# 打印九九乘法表
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j}", end="\t")
    print()   # 每行结束后换行
```

输出：

```
1×1=1
1×2=2	2×2=4
1×3=3	2×3=6	3×3=9
...
1×9=9	2×9=18	3×9=27	...	9×9=81
```

---

## 综合案例

### 案例 1：模拟 AI 模型训练过程

```python
import random

print("=== 开始训练模型 ===")
print(f"{'Epoch':<10}{'Loss':<15}{'Accuracy':<15}{'Status'}")
print("-" * 50)

loss = 2.5
accuracy = 0.10

for epoch in range(1, 21):
    # 模拟训练：损失逐渐下降，准确率逐渐上升
    loss *= random.uniform(0.85, 0.95)
    accuracy = min(accuracy + random.uniform(0.03, 0.06), 1.0)

    # 判断训练状态
    if accuracy >= 0.95:
        status = "✅ 达标"
    elif accuracy >= 0.80:
        status = "📈 良好"
    else:
        status = "🔄 训练中"

    print(f"{epoch:<10}{loss:<15.4f}{accuracy:<15.2%}{status}")

    # 如果准确率达到 98%，提前停止
    if accuracy >= 0.98:
        print(f"\n提前停止！在第 {epoch} 轮达到目标准确率")
        break
else:
    print(f"\n训练完成！最终准确率: {accuracy:.2%}")
```

### 案例 2：密码强度检查器

```python
password = input("请输入密码: ")

has_upper = False    # 是否有大写字母
has_lower = False    # 是否有小写字母
has_digit = False    # 是否有数字
has_special = False  # 是否有特殊字符

for char in password:
    if char.isupper():
        has_upper = True
    elif char.islower():
        has_lower = True
    elif char.isdigit():
        has_digit = True
    else:
        has_special = True

# 计算强度分数
score = 0
if len(password) >= 8:
    score += 1
if has_upper:
    score += 1
if has_lower:
    score += 1
if has_digit:
    score += 1
if has_special:
    score += 1

# 输出结果
print(f"\n密码强度: {'★' * score}{'☆' * (5 - score)} ({score}/5)")

if score <= 2:
    print("弱密码！建议加强")
elif score <= 4:
    print("中等强度")
else:
    print("强密码！")
```

---

## 动手练习

### 练习 1：FizzBuzz

这是经典的编程面试题：

打印 1 到 50 的数字，但是：
- 如果数字能被 3 整除，打印 "Fizz"
- 如果数字能被 5 整除，打印 "Buzz"
- 如果数字同时能被 3 和 5 整除，打印 "FizzBuzz"
- 其他情况打印数字本身

```python
for i in range(1, 51):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

提示：先判断能否被 15 整除（3 和 5 的公倍数），再判断 3 和 5。

### 练习 2：猜数字游戏（限制次数）

改进猜数字游戏：最多允许猜 7 次，超过就失败。

```python
import random
target = random.randint(1, 100)
max_attempts = 7

for attempt in range(1, max_attempts + 1):
    raw = input(f"第 {attempt}/{max_attempts} 次，请输入你的猜测：")

    if not raw.isdigit():
        print("请输入整数。")
        continue

    guess = int(raw)
    if guess == target:
        print("猜对了！")
        break
    elif guess < target:
        print("太小了")
    else:
        print("太大了")
else:
    print(f"失败，答案是 {target}。")
```

:::tip 如何减少调试挫败感
学习流程控制时，交互和随机数会让调试变难。你可以先把 `target = random.randint(1, 100)` 临时改成 `target = 42`，分别测试“太小、太大、猜对”三个分支，确认逻辑没问题后再改回随机版本。
:::

### 练习 3：画三角形

用循环打印以下图案：

```
*
**
***
****
*****
```

然后试着打印倒三角形：

```
*****
****
***
**
*
```

### 练习 4：求质数

打印 1 到 100 之间的所有质数。

提示：质数是大于 1 的自然数，且只能被 1 和自身整除。

<details>
<summary>参考实现与讲解</summary>

1. FizzBuzz 要先判断能否被 `15` 整除，否则 `15` 可能会提前输出成 `Fizz` 或 `Buzz`。
2. 如果把目标数固定为 `42`，至少测试过小、过大、猜中、输入非整数、次数用尽这几条路径。
3. 三角形图案可以用 `for n in range(1, 6): print("*" * n)` 打印，倒三角则用反向 `range`。
4. 判断质数时要跳过 `1`，并且只输出从 `2` 到 `n - 1` 或 `sqrt(n)` 都没有约数的数字。
5. 注意边界错误：`range(1, 51)` 包含 `50`，`range(1, 50)` 不包含 `50`。

</details>

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
概念：变量、类型、运算符、输入/输出、分支、循环、结构、函数或模块
代码：用于说明该概念的最小可运行 Python 代码片段
输出：打印值、类型、分支结果、循环 trace，或返回值
失败检查：类型不匹配、缩进错误、越界、可变数据或导入路径问题
期望产出：代码和打印结果，证明概念可行
```

## 小结

| 语法 | 用途 | 关键点 |
|------|------|--------|
| `if/elif/else` | 条件判断 | 条件从上到下检查，冒号和缩进不能忘 |
| `for...in` | 遍历序列 | 配合 `range()`、列表、字符串使用 |
| `while` | 条件循环 | 注意更新条件，避免死循环 |
| `break` | 终止循环 | 立即跳出整个循环 |
| `continue` | 跳过本次 | 跳过当前迭代，继续下一次 |
| `range()` | 生成数字序列 | `range(start, stop, step)` |

:::tip 核心理解
流程控制是编程的**骨架**。变量是数据，运算符是操作，而流程控制决定了"在什么条件下做什么"和"做多少次"。学会了流程控制，你就能写出有"逻辑"的程序了。
:::
