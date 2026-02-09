---
title: "1.3 运算符与表达式"
sidebar_position: 3
description: "掌握 Python 中的各种运算符和表达式"
---

# 运算符与表达式

## 学习目标

- 掌握算术运算符、比较运算符、逻辑运算符
- 理解运算符的优先级
- 学会使用赋值运算符和成员运算符
- 能写出正确的条件表达式

---

## 先看一个场景

你在开发一个 AI 数据处理脚本，需要：
- 计算模型准确率：`correct / total * 100`
- 判断是否及格：`accuracy >= 60`
- 检查两个条件：`accuracy >= 60 and loss < 0.5`

这些操作都离不开**运算符**。运算符就是告诉 Python "对数据做什么操作"的符号。

---

## 算术运算符

最基础的数学运算：

| 运算符 | 含义 | 示例 | 结果 |
|--------|------|------|------|
| `+` | 加法 | `5 + 3` | `8` |
| `-` | 减法 | `5 - 3` | `2` |
| `*` | 乘法 | `5 * 3` | `15` |
| `/` | 除法 | `5 / 3` | `1.6667` |
| `//` | 整除 | `5 // 3` | `1` |
| `%` | 取余 | `5 % 3` | `2` |
| `**` | 幂运算 | `5 ** 3` | `125` |

### 实际案例

```python
# 场景：计算 AI 模型训练的一些指标

total_samples = 1000     # 总样本数
correct = 873            # 正确预测数
epochs = 50              # 训练轮数
batch_size = 32          # 批次大小

# 计算准确率
accuracy = correct / total_samples * 100
print(f"准确率: {accuracy}%")  # 87.3%

# 计算需要多少个批次才能跑完一个 epoch
batches_per_epoch = total_samples // batch_size
remaining = total_samples % batch_size

print(f"每个 epoch 有 {batches_per_epoch} 个完整批次")  # 31
print(f"最后一个批次有 {remaining} 个样本")              # 8

# 计算指数衰减的学习率
initial_lr = 0.01
decay = 0.95
current_lr = initial_lr * (decay ** epochs)
print(f"第 {epochs} 轮的学习率: {current_lr:.6f}")  # 0.000769
```

### 除法的两种形式

这是新手常混淆的地方：

```python
print(7 / 2)    # 3.5   ← 普通除法，结果是 float
print(7 // 2)   # 3     ← 整除，丢掉小数部分
print(-7 // 2)  # -4    ← 注意！向下取整，不是向零取整

# 取余的妙用
print(10 % 3)   # 1    ← 10 除以 3 余 1
print(15 % 5)   # 0    ← 整除时余数为 0

# 判断奇偶数
number = 42
if number % 2 == 0:
    print(f"{number} 是偶数")  # 42 是偶数
```

---

## 比较运算符

比较运算符的结果总是布尔值（`True` 或 `False`）：

| 运算符 | 含义 | 示例 | 结果 |
|--------|------|------|------|
| `==` | 等于 | `5 == 5` | `True` |
| `!=` | 不等于 | `5 != 3` | `True` |
| `>` | 大于 | `5 > 3` | `True` |
| `<` | 小于 | `5 < 3` | `False` |
| `>=` | 大于等于 | `5 >= 5` | `True` |
| `<=` | 小于等于 | `5 <= 3` | `False` |

```python
# 场景：判断模型表现
accuracy = 87.3
loss = 0.35

print(accuracy > 90)      # False —— 准确率没超过 90
print(accuracy >= 80)      # True  —— 准确率达到了 80 以上
print(loss < 0.5)          # True  —— 损失值低于 0.5
print(accuracy == 87.3)    # True  —— 准确率恰好是 87.3
```

:::caution 常见错误：= 和 == 的区别
- `=` 是**赋值**：`x = 5` 把 5 赋给 x
- `==` 是**比较**：`x == 5` 判断 x 是否等于 5

初学者最容易犯的错误就是在判断时写成 `=` 而不是 `==`。
:::

### 链式比较（Python 特有）

Python 允许链式比较，这在其他语言中是做不到的：

```python
age = 25

# 判断 age 是否在 18 到 65 之间
print(18 <= age <= 65)   # True

# 等价于
print(18 <= age and age <= 65)   # True，但上面的写法更简洁

# 更多示例
x = 5
print(1 < x < 10)      # True
print(1 < x < 3)       # False
```

---

## 逻辑运算符

逻辑运算符用来组合多个条件：

| 运算符 | 含义 | 说明 |
|--------|------|------|
| `and` | 与 | **两个都为真**才是真 |
| `or` | 或 | **至少一个为真**就是真 |
| `not` | 非 | **取反**，真变假，假变真 |

```python
age = 25
has_id = True
has_ticket = False

# and：两个条件都满足
can_enter = age >= 18 and has_id
print(f"能否入场: {can_enter}")   # True（年龄够了，也有证件）

# or：至少满足一个条件
has_pass = has_id or has_ticket
print(f"有通行凭证: {has_pass}")  # True（有证件就行）

# not：取反
is_minor = not (age >= 18)
print(f"是否未成年: {is_minor}")   # False
```

### 实际案例：AI 模型评估

```python
accuracy = 92.5
loss = 0.15
training_time = 3.5  # 小时

# 好模型的标准：准确率 > 90 且 损失 < 0.3
is_good_model = accuracy > 90 and loss < 0.3
print(f"是好模型吗: {is_good_model}")  # True

# 需要重新训练：准确率太低 或 损失太高
need_retrain = accuracy < 80 or loss > 1.0
print(f"需要重新训练吗: {need_retrain}")  # False

# 实用模型：好模型 且 训练时间合理
is_practical = is_good_model and not (training_time > 24)
print(f"是否实用: {is_practical}")  # True
```

### 短路求值

Python 的 `and` 和 `or` 有一个聪明的特性——**短路求值**：

```python
# and：如果第一个条件是 False，不会检查第二个条件
# 因为第一个已经是 False 了，结果必然是 False
False and print("这句话不会被执行")

# or：如果第一个条件是 True，不会检查第二个条件
# 因为第一个已经是 True 了，结果必然是 True
True or print("这句话也不会被执行")
```

这个特性在实际编程中经常用来做**安全检查**：

```python
# 先检查列表是否为空，再访问元素（避免报错）
data = []
# 如果 data 为空，len(data) > 0 是 False，后面的不会执行
if len(data) > 0 and data[0] > 10:
    print("第一个元素大于 10")
```

---

## 赋值运算符

除了基本的 `=`，还有一些简写形式：

| 运算符 | 等价写法 | 示例 |
|--------|---------|------|
| `+=` | `a = a + b` | `a += 5` |
| `-=` | `a = a - b` | `a -= 3` |
| `*=` | `a = a * b` | `a *= 2` |
| `/=` | `a = a / b` | `a /= 4` |
| `//=` | `a = a // b` | `a //= 3` |
| `%=` | `a = a % b` | `a %= 2` |
| `**=` | `a = a ** b` | `a **= 3` |

```python
score = 0

score += 10   # score = 0 + 10 = 10
score += 20   # score = 10 + 20 = 30
score -= 5    # score = 30 - 5 = 25
score *= 2    # score = 25 * 2 = 50

print(f"最终分数: {score}")  # 50
```

这些简写在循环中特别常用：

```python
# 累加 1 到 100
total = 0
for i in range(1, 101):
    total += i
print(f"1 到 100 的和: {total}")  # 5050
```

---

## 成员运算符

`in` 和 `not in` 用来检查某个值**是否在**一个集合中：

```python
# 在字符串中查找
print("Python" in "I love Python")     # True
print("Java" in "I love Python")       # False
print("Java" not in "I love Python")   # True

# 在列表中查找
fruits = ["苹果", "香蕉", "橙子"]
print("苹果" in fruits)      # True
print("西瓜" in fruits)      # False

# 实际应用：检查文件扩展名
filename = "model.py"
if ".py" in filename:
    print("这是一个 Python 文件")
```

---

## 身份运算符

`is` 和 `is not` 用来检查两个变量是否是**同一个对象**（不是值相等，而是内存中的同一个东西）：

```python
a = None

# 检查是否是 None（推荐用 is，不用 ==）
print(a is None)       # True
print(a is not None)   # False

# is 和 == 的区别
x = [1, 2, 3]
y = [1, 2, 3]
z = x

print(x == y)    # True  —— 值相等
print(x is y)    # False —— 不是同一个对象（两个不同的列表）
print(x is z)    # True  —— z 指向 x，是同一个对象
```

:::tip 什么时候用 is？
99% 的情况下你用 `==` 就够了。`is` 主要用于和 `None` 比较：
- 好：`if x is None:`
- 不好：`if x == None:`
:::

---

## 运算符优先级

当一个表达式里有多个运算符时，Python 按照**优先级**从高到低计算：

| 优先级（高→低） | 运算符 |
|-----------------|--------|
| 1（最高） | `**` 幂运算 |
| 2 | `+x`, `-x` 正负号 |
| 3 | `*`, `/`, `//`, `%` |
| 4 | `+`, `-` |
| 5 | `==`, `!=`, `>`, `<`, `>=`, `<=` |
| 6 | `not` |
| 7 | `and` |
| 8（最低） | `or` |

```python
# 不加括号
result = 2 + 3 * 4      # 先乘后加：2 + 12 = 14
result = 2 ** 3 ** 2     # 幂运算从右到左：2 ** 9 = 512

# 加括号更清晰（推荐）
result = (2 + 3) * 4     # 20
result = (2 ** 3) ** 2   # 64
```

:::tip 实用建议
**不确定优先级的时候，加括号！** 括号不仅能确保计算顺序正确，还能让代码更容易读懂。没人会因为你多写了括号而笑话你。
:::

---

## 综合案例：BMI 计算器

把今天学的运算符综合运用一下：

```python
# BMI 计算器
name = "小明"
weight = 70      # 公斤
height = 1.75    # 米

# 计算 BMI：体重 / 身高的平方
bmi = weight / height ** 2  # 先算 height**2，再算除法
print(f"{name} 的 BMI: {bmi:.1f}")  # 22.9

# 判断体重范围
is_underweight = bmi < 18.5
is_normal = 18.5 <= bmi < 24
is_overweight = 24 <= bmi < 28
is_obese = bmi >= 28

print(f"体重过轻: {is_underweight}")  # False
print(f"体重正常: {is_normal}")        # True
print(f"超重: {is_overweight}")        # False
print(f"肥胖: {is_obese}")            # False

# 综合判断
is_healthy = is_normal and not is_underweight
print(f"健康: {is_healthy}")           # True
```

---

## 动手练习

### 练习 1：成绩等级判断

用比较运算符和逻辑运算符判断成绩等级：

```python
score = 85

is_excellent = score >= 90               # 优秀
is_good = score >= 80 and score < 90     # 良好
is_pass = score >= 60 and score < 80     # 及格
is_fail = score < 60                     # 不及格

# 打印结果
print(f"分数: {score}")
print(f"优秀: {is_excellent}")
print(f"良好: {is_good}")
print(f"及格: {is_pass}")
print(f"不及格: {is_fail}")
```

修改 `score` 的值，试试不同分数的结果。

### 练习 2：闰年判断

闰年规则：能被 4 整除但不能被 100 整除，或者能被 400 整除。

```python
year = 2024

# 提示：用 % 判断能否整除，用 and、or 组合条件
is_leap = ___  # 补全这个表达式

print(f"{year} 是闰年吗？{is_leap}")
```

### 练习 3：三角形判断

判断三条边能否构成三角形（任意两边之和大于第三边）：

```python
a, b, c = 3, 4, 5

# 补全判断条件
is_triangle = ___

print(f"边长 {a}, {b}, {c} 能构成三角形吗？{is_triangle}")
```

---

## 小结

| 运算符类型 | 常用符号 | 用途 |
|-----------|---------|------|
| **算术** | `+`, `-`, `*`, `/`, `//`, `%`, `**` | 数学计算 |
| **比较** | `==`, `!=`, `>`, `<`, `>=`, `<=` | 条件判断，结果是 True/False |
| **逻辑** | `and`, `or`, `not` | 组合多个条件 |
| **赋值** | `=`, `+=`, `-=`, `*=` 等 | 给变量赋值 |
| **成员** | `in`, `not in` | 检查元素是否在集合中 |
| **身份** | `is`, `is not` | 检查是否是同一个对象 |

:::tip 核心理解
运算符是编程的基础"动词"。变量和数据是"名词"，运算符是"动词"，它们组合在一起构成"表达式"——也就是你告诉计算机要做什么。
:::
