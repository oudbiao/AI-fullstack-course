---
title: "1.6 数据结构"
sidebar_position: 6
description: "掌握 Python 中的四大数据结构：列表、元组、字典、集合"
---

# 数据结构

## 学习目标

- 掌握列表（list）的创建和常用操作
- 理解元组（tuple）的特点和使用场景
- 掌握字典（dict）的键值对操作
- 了解集合（set）的去重和集合运算
- 能根据场景选择合适的数据结构

---

## 为什么需要数据结构？

到目前为止，你学的变量一次只能存一个值。但在真实场景中，你经常需要处理**一组数据**：

- 100 个学生的成绩
- 一个模型的所有参数
- 用户的个人信息（姓名、年龄、邮箱……）

数据结构就是用来**组织和存储多个数据**的容器。

Python 有 4 种内置数据结构：

| 数据结构 | 符号 | 有序 | 可变 | 允许重复 | 典型用途 |
|---------|------|------|------|---------|---------|
| **列表** list | `[]` | ✅ | ✅ | ✅ | 有序数据集合 |
| **元组** tuple | `()` | ✅ | ❌ | ✅ | 不可变的数据 |
| **字典** dict | `{}` | ✅ | ✅ | 键不可重复 | 键值对映射 |
| **集合** set | `{}` | ❌ | ✅ | ❌ | 去重、集合运算 |

---

## 列表（list）—— 最常用的数据结构

列表就像一个**可以伸缩的柜子**，你可以往里面放任何东西，也可以随时增删改。

### 创建列表

```python
# 创建列表
scores = [85, 92, 78, 95, 88]
names = ["张三", "李四", "王五"]
mixed = [1, "hello", 3.14, True]   # 可以混合类型（但不推荐）
empty = []                          # 空列表

print(type(scores))  # <class 'list'>
print(len(scores))   # 5
```

### 访问元素（索引）

```python
fruits = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]
#          0       1       2       3       4
#         -5      -4      -3      -2      -1

print(fruits[0])     # 苹果（第一个）
print(fruits[2])     # 橙子（第三个）
print(fruits[-1])    # 西瓜（最后一个）
print(fruits[-2])    # 葡萄（倒数第二个）
```

### 切片

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:5])    # [2, 3, 4]（索引 2 到 4）
print(numbers[:3])     # [0, 1, 2]（前 3 个）
print(numbers[7:])     # [7, 8, 9]（从索引 7 到末尾）
print(numbers[::2])    # [0, 2, 4, 6, 8]（每隔一个取一个）
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]（反转）
```

### 修改元素

```python
scores = [85, 92, 78, 95, 88]

# 修改单个元素
scores[2] = 80
print(scores)  # [85, 92, 80, 95, 88]

# 修改多个元素（通过切片）
scores[1:3] = [90, 85]
print(scores)  # [85, 90, 85, 95, 88]
```

### 添加元素

```python
fruits = ["苹果", "香蕉"]

# 在末尾添加
fruits.append("橙子")
print(fruits)  # ['苹果', '香蕉', '橙子']

# 在指定位置插入
fruits.insert(1, "葡萄")
print(fruits)  # ['苹果', '葡萄', '香蕉', '橙子']

# 添加多个元素
fruits.extend(["西瓜", "草莓"])
print(fruits)  # ['苹果', '葡萄', '香蕉', '橙子', '西瓜', '草莓']
```

### 删除元素

```python
fruits = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]

# 按值删除（删除第一个匹配项）
fruits.remove("橙子")
print(fruits)  # ['苹果', '香蕉', '葡萄', '西瓜']

# 按索引删除
deleted = fruits.pop(1)    # 删除索引 1 的元素，并返回它
print(deleted)             # 香蕉
print(fruits)              # ['苹果', '葡萄', '西瓜']

# 删除最后一个
last = fruits.pop()
print(last)    # 西瓜

# 按索引删除（不需要返回值）
del fruits[0]
print(fruits)  # ['葡萄']
```

### 列表常用操作

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# 排序
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 5, 6, 9]

# 降序排序
numbers.sort(reverse=True)
print(numbers)  # [9, 6, 5, 5, 4, 3, 2, 1, 1]

# 不修改原列表的排序
original = [3, 1, 4, 1, 5]
sorted_list = sorted(original)
print(original)    # [3, 1, 4, 1, 5]（原列表不变）
print(sorted_list) # [1, 1, 3, 4, 5]

# 反转
numbers = [1, 2, 3, 4, 5]
numbers.reverse()
print(numbers)  # [5, 4, 3, 2, 1]

# 查找
print(numbers.index(3))    # 2（元素 3 的索引）
print(numbers.count(5))    # 1（元素 5 出现的次数）
print(3 in numbers)        # True

# 统计
scores = [85, 92, 78, 95, 88]
print(len(scores))    # 5
print(sum(scores))    # 438
print(max(scores))    # 95
print(min(scores))    # 78
print(sum(scores) / len(scores))  # 87.6（平均分）
```

### 列表推导式（非常 Pythonic！）

列表推导式是创建新列表的简洁方式：

```python
# 传统方式
squares = []
for i in range(1, 6):
    squares.append(i ** 2)
print(squares)  # [1, 4, 9, 16, 25]

# 列表推导式（一行搞定！）
squares = [i ** 2 for i in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]

# 带条件的列表推导式
even_squares = [i ** 2 for i in range(1, 11) if i % 2 == 0]
print(even_squares)  # [4, 16, 36, 64, 100]

# 实际应用：批量处理数据
names = ["  Alice  ", "BOB", "  charlie "]
clean_names = [name.strip().lower() for name in names]
print(clean_names)  # ['alice', 'bob', 'charlie']
```

:::tip 列表推导式的公式
`[表达式 for 变量 in 可迭代对象 if 条件]`

翻译成中文就是：对于每一个满足条件的元素，计算表达式，放进新列表。
:::

---

## 元组（tuple）—— 不可变的列表

元组和列表几乎一样，唯一的区别是：**元组创建后不能修改**。

### 创建元组

```python
# 用圆括号创建
point = (3, 4)
colors = ("红", "绿", "蓝")
single = (42,)          # 只有一个元素时，必须加逗号！
empty = ()

# 其实圆括号可以省略
coordinates = 3, 4      # 也是元组
print(type(coordinates)) # <class 'tuple'>
```

### 元组的操作

```python
colors = ("红", "绿", "蓝", "黄", "紫")

# 访问（和列表一样）
print(colors[0])     # 红
print(colors[-1])    # 紫
print(colors[1:3])   # ('绿', '蓝')

# 遍历
for color in colors:
    print(color)

# 查找
print(len(colors))          # 5
print("红" in colors)       # True
print(colors.count("红"))   # 1
print(colors.index("蓝"))   # 2

# 但是不能修改！
# colors[0] = "黑"  # 报错！TypeError: 'tuple' object does not support item assignment
```

### 元组的解包

```python
# 把元组的值分别赋给多个变量
point = (10, 20)
x, y = point
print(f"x={x}, y={y}")  # x=10, y=20

# 函数返回多个值时，实际返回的是元组
def get_name_and_age():
    return "小明", 25

name, age = get_name_and_age()
print(f"{name}, {age}岁")  # 小明, 25岁

# 用 * 收集多余的值
first, *rest = [1, 2, 3, 4, 5]
print(first)  # 1
print(rest)   # [2, 3, 4, 5]
```

### 什么时候用元组？

- 数据不应该被修改时（比如坐标、颜色 RGB 值）
- 字典的键（列表不能做字典的键，元组可以）
- 函数返回多个值时

---

## 字典（dict）—— 键值对存储

字典是 Python 中**最重要的数据结构之一**。它用**键（key）** 来查找**值（value）**，就像真实的字典用单词查释义一样。

### 创建字典

```python
# 用花括号创建
student = {
    "name": "小明",
    "age": 20,
    "city": "北京",
    "scores": [85, 92, 78]
}

# 空字典
empty = {}

# 用 dict() 创建
config = dict(learning_rate=0.001, epochs=100, batch_size=32)
print(config)  # {'learning_rate': 0.001, 'epochs': 100, 'batch_size': 32}

print(type(student))  # <class 'dict'>
```

### 访问值

```python
student = {"name": "小明", "age": 20, "city": "北京"}

# 方法 1：用 [] 访问
print(student["name"])   # 小明
# print(student["phone"])  # 报错！KeyError: 'phone'

# 方法 2：用 .get() 访问（更安全）
print(student.get("name"))    # 小明
print(student.get("phone"))   # None（不存在时返回 None，不会报错）
print(student.get("phone", "未填写"))  # 未填写（不存在时返回默认值）
```

:::tip 推荐使用 .get()
当你不确定键是否存在时，用 `.get()` 比 `[]` 更安全，不会导致程序崩溃。
:::

### 添加和修改

```python
student = {"name": "小明", "age": 20}

# 添加新键值对
student["city"] = "北京"
student["email"] = "xiaoming@example.com"

# 修改已有的值
student["age"] = 21

print(student)
# {'name': '小明', 'age': 21, 'city': '北京', 'email': 'xiaoming@example.com'}

# 批量更新
student.update({"age": 22, "phone": "13800000000"})
print(student)
```

### 删除

```python
student = {"name": "小明", "age": 20, "city": "北京"}

# 删除指定键
del student["city"]
print(student)  # {'name': '小明', 'age': 20}

# pop：删除并返回值
age = student.pop("age")
print(age)      # 20
print(student)  # {'name': '小明'}
```

### 遍历字典

```python
scores = {"语文": 85, "数学": 92, "英语": 78}

# 遍历键
for subject in scores:
    print(subject)

# 遍历值
for score in scores.values():
    print(score)

# 遍历键值对（最常用）
for subject, score in scores.items():
    print(f"{subject}: {score} 分")

# 输出:
# 语文: 85 分
# 数学: 92 分
# 英语: 78 分
```

### 字典推导式

```python
# 创建一个数字到平方的映射
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 过滤字典
scores = {"张三": 85, "李四": 45, "王五": 92, "赵六": 58}
passed = {name: score for name, score in scores.items() if score >= 60}
print(passed)  # {'张三': 85, '王五': 92}
```

### 实际案例：统计字符出现次数

```python
text = "hello world"
char_count = {}

for char in text:
    if char in char_count:
        char_count[char] += 1
    else:
        char_count[char] = 1

print(char_count)
# {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
```

---

## 集合（set）—— 去重利器

集合是**无序且不重复**的元素集合。

### 创建集合

```python
# 用花括号创建
fruits = {"苹果", "香蕉", "橙子", "苹果"}  # 重复的会自动去掉
print(fruits)  # {'香蕉', '橙子', '苹果'}（顺序可能不同）

# 从列表创建（去重！）
numbers = [1, 2, 2, 3, 3, 3, 4, 4, 4, 4]
unique = set(numbers)
print(unique)  # {1, 2, 3, 4}

# 注意：空集合要用 set()，不能用 {}
empty_set = set()     # 空集合
empty_dict = {}       # 这是空字典！

print(type(fruits))   # <class 'set'>
```

### 集合操作

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# 交集（两个都有的）
print(a & b)          # {4, 5}
print(a.intersection(b))

# 并集（合在一起，去重）
print(a | b)          # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))

# 差集（a 有但 b 没有的）
print(a - b)          # {1, 2, 3}
print(a.difference(b))

# 对称差集（各自独有的）
print(a ^ b)          # {1, 2, 3, 6, 7, 8}
```

### 实际应用

```python
# 场景：找出两门课都选了的学生
math_students = {"张三", "李四", "王五", "赵六"}
english_students = {"李四", "王五", "钱七", "孙八"}

both = math_students & english_students
print(f"两门课都选了: {both}")  # {'李四', '王五'}

only_math = math_students - english_students
print(f"只选了数学: {only_math}")  # {'张三', '赵六'}

all_students = math_students | english_students
print(f"所有选课学生: {all_students}")
```

---

## 数据结构选择指南

| 需求 | 推荐 | 原因 |
|------|------|------|
| 有序集合，需要增删改 | **列表** | 最通用的容器 |
| 数据不应被修改 | **元组** | 不可变，更安全 |
| 通过键查找值 | **字典** | O(1) 查找速度 |
| 去重 | **集合** | 自动去重 |
| 统计出现次数 | **字典** | 键为元素，值为计数 |
| 检查元素是否存在 | **集合/字典** | 比列表快得多 |

---

## 动手练习

### 练习 1：成绩统计

```python
scores = [85, 92, 78, 95, 88, 76, 90, 82, 97, 73]

# 1. 计算最高分、最低分、平均分
# 2. 找出所有 90 分以上的成绩（用列表推导式）
# 3. 把成绩从高到低排序
```

### 练习 2：通讯录

用字典实现一个简单的通讯录：

```python
contacts = {}

# 1. 添加 3 个联系人（姓名 → 电话）
# 2. 查找某个联系人的电话
# 3. 修改某个联系人的电话
# 4. 删除一个联系人
# 5. 打印所有联系人
```

### 练习 3：单词频率统计

```python
text = "the quick brown fox jumps over the lazy dog the fox"

# 统计每个单词出现的次数
# 提示：先 split() 分割成列表，再用字典统计
```

### 练习 4：列表去重（保持顺序）

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# 去除重复元素，但保持原来的顺序
# 期望输出: [3, 1, 4, 5, 9, 2, 6]
# 提示：用集合记录已出现的元素
```

---

## 小结

| 数据结构 | 创建方式 | 特点 | 常用场景 |
|---------|---------|------|---------|
| **列表** | `[1, 2, 3]` | 有序、可变、可重复 | 存储一组同类数据 |
| **元组** | `(1, 2, 3)` | 有序、不可变 | 坐标、返回多个值 |
| **字典** | `{"a": 1}` | 键值对、键不可重复 | 配置、映射关系 |
| **集合** | `{1, 2, 3}` | 无序、不重复 | 去重、集合运算 |

:::tip 核心理解
选择数据结构就像选择收纳工具：列表像**抽屉**（有序排列），字典像**标签柜**（用标签找东西），集合像**筛子**（自动去重），元组像**密封袋**（放进去就不能改了）。选对工具，事半功倍。
:::
