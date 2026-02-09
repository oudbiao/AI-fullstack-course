---
title: "1.2 异常处理"
sidebar_position: 2
description: "掌握 Python 异常处理机制，让程序更健壮"
---

# 异常处理

## 学习目标

- 理解什么是异常，为什么需要处理异常
- 掌握 `try/except/else/finally` 的用法
- 学会捕获不同类型的异常
- 能编写健壮的、不会轻易崩溃的程序

---

## 什么是异常？

异常就是程序运行时发生的**错误**。没有异常处理的程序，一遇到错误就会直接崩溃：

```python
# 这些代码都会导致程序崩溃
print(10 / 0)           # ZeroDivisionError: 除以零
print(int("abc"))        # ValueError: 无法转换
print([1, 2, 3][10])     # IndexError: 索引越界
print({"a": 1}["b"])     # KeyError: 键不存在

# 程序崩溃意味着后面的代码都不会执行
print("这行永远不会被执行")
```

在真实的程序中，错误是**不可避免的**——用户可能输入非法数据、文件可能不存在、网络可能断开。异常处理让你能**优雅地应对这些问题**，而不是让程序直接崩溃。

---

## 常见的异常类型

| 异常类型 | 触发场景 | 示例 |
|---------|---------|------|
| `ZeroDivisionError` | 除以零 | `1 / 0` |
| `TypeError` | 类型操作不匹配 | `"hello" + 5` |
| `ValueError` | 值不合法 | `int("abc")` |
| `IndexError` | 列表索引越界 | `[1, 2][5]` |
| `KeyError` | 字典键不存在 | `{"a": 1}["b"]` |
| `FileNotFoundError` | 文件不存在 | `open("不存在.txt")` |
| `AttributeError` | 属性不存在 | `"hello".foo()` |
| `NameError` | 变量未定义 | `print(xyz)` |
| `ImportError` | 导入失败 | `import 不存在的模块` |

---

## try / except 基本用法

`try/except` 的逻辑是：**尝试执行代码，如果出错了，执行备选方案。**

```python
try:
    number = int(input("请输入一个数字: "))
    print(f"你输入的是: {number}")
except ValueError:
    print("输入无效！请输入一个数字。")

print("程序继续运行...")  # 不管有没有异常，这行都会执行
```

运行效果：

```
# 正常输入
请输入一个数字: 42
你输入的是: 42
程序继续运行...

# 输入非数字
请输入一个数字: abc
输入无效！请输入一个数字。
程序继续运行...
```

关键点：**有了 `try/except`，程序不会因为错误而崩溃。**

---

## 捕获不同类型的异常

### 捕获多种异常

```python
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("错误：不能除以零！")
        return None
    except TypeError:
        print("错误：请传入数字！")
        return None

print(safe_divide(10, 3))    # 3.333...
print(safe_divide(10, 0))    # 错误：不能除以零！ → None
print(safe_divide("10", 3))  # 错误：请传入数字！ → None
```

### 捕获多种异常（合并写法）

```python
try:
    # 可能出错的代码
    value = int(input("请输入数字: "))
    result = 100 / value
    print(f"结果: {result}")
except (ValueError, ZeroDivisionError) as e:
    print(f"出错了: {e}")
```

### 获取异常信息

```python
try:
    number = int("abc")
except ValueError as e:
    print(f"异常类型: {type(e).__name__}")  # ValueError
    print(f"异常信息: {e}")                 # invalid literal for int() with base 10: 'abc'
```

### 捕获所有异常（谨慎使用）

```python
try:
    # 一些代码
    result = risky_operation()
except Exception as e:
    print(f"发生了意外错误: {type(e).__name__}: {e}")
```

:::caution 不要滥用 except Exception
捕获所有异常听起来很方便，但会**掩盖真正的 bug**。你应该尽量捕获**具体的异常类型**，只在最外层使用 `except Exception` 作为兜底。

```python
# 不好的做法 ❌
try:
    do_something()
except:  # 捕获所有异常，包括 KeyboardInterrupt
    pass   # 而且还什么都不做！

# 好的做法 ✅
try:
    do_something()
except ValueError:
    handle_value_error()
except FileNotFoundError:
    handle_file_not_found()
except Exception as e:
    logging.error(f"未预期的错误: {e}")
```
:::

---

## try / except / else / finally

完整的异常处理结构有四个部分：

```python
try:
    # 尝试执行的代码
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    # 出错时执行
    print("文件不存在！")
else:
    # 没有出错时执行
    print(f"文件内容: {content}")
finally:
    # 不管有没有出错都执行（通常用来清理资源）
    print("操作完成")
```

| 子句 | 何时执行 | 用途 |
|------|---------|------|
| `try` | 总是执行 | 放可能出错的代码 |
| `except` | 只在出错时执行 | 处理错误 |
| `else` | 只在没出错时执行 | 放成功后的逻辑 |
| `finally` | 不管有没有出错都执行 | 清理资源（关闭文件、断开连接） |

### finally 的典型用途

```python
file = None
try:
    file = open("data.txt", "r")
    data = file.read()
    # 处理数据...
except FileNotFoundError:
    print("文件不存在")
finally:
    if file:
        file.close()   # 不管有没有出错，都要关闭文件
        print("文件已关闭")
```

:::tip 更好的方式：with 语句
在后面的"文件操作"章节中，你会学到 `with` 语句，它可以自动处理资源的关闭，比 `finally` 更简洁。
:::

---

## 抛出异常

除了处理异常，你也可以**主动抛出异常**——当你发现一个不合理的状态时，告诉调用者"出问题了"。

### raise 语句

```python
def set_age(age):
    if not isinstance(age, int):
        raise TypeError("年龄必须是整数")
    if age < 0 or age > 150:
        raise ValueError(f"年龄 {age} 不合理，应该在 0-150 之间")
    return age

# 正常使用
print(set_age(25))      # 25

# 触发异常
try:
    set_age(-5)
except ValueError as e:
    print(f"错误: {e}")  # 错误: 年龄 -5 不合理，应该在 0-150 之间

try:
    set_age("二十")
except TypeError as e:
    print(f"错误: {e}")  # 错误: 年龄必须是整数
```

### 自定义异常

当内置异常类型不够用时，可以自定义：

```python
class InsufficientFundsError(Exception):
    """余额不足异常"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"余额不足：当前余额 {balance}，尝试取出 {amount}")

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return self.balance

# 使用
account = BankAccount(1000)
try:
    account.withdraw(1500)
except InsufficientFundsError as e:
    print(f"交易失败: {e}")
    print(f"当前余额: {e.balance}, 请求金额: {e.amount}")
```

---

## 实战模式

### 模式 1：LBYL vs EAFP

Python 社区推崇 **EAFP**（Easier to Ask Forgiveness than Permission，先做再说）而不是 **LBYL**（Look Before You Leap，先检查再做）：

```python
# LBYL 风格（先检查再操作）—— 不够 Pythonic
if key in my_dict:
    value = my_dict[key]
else:
    value = default_value

# EAFP 风格（先操作，出错再处理）—— 更 Pythonic
try:
    value = my_dict[key]
except KeyError:
    value = default_value

# 当然，字典有更好的写法
value = my_dict.get(key, default_value)
```

### 模式 2：重试机制

```python
import time

def fetch_data_with_retry(url, max_retries=3):
    """带重试的数据获取"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"第 {attempt} 次尝试...")
            # 模拟网络请求
            import random
            if random.random() < 0.5:
                raise ConnectionError("网络连接失败")
            return "获取到的数据"
        except ConnectionError as e:
            print(f"  失败: {e}")
            if attempt < max_retries:
                wait = attempt * 2  # 递增等待时间
                print(f"  {wait} 秒后重试...")
                time.sleep(wait)
            else:
                print("  所有重试均失败！")
                raise  # 最后一次重试失败，抛出异常

try:
    data = fetch_data_with_retry("https://api.example.com")
    print(f"成功: {data}")
except ConnectionError:
    print("最终获取数据失败")
```

### 模式 3：安全的用户输入

```python
def get_number(prompt, min_val=None, max_val=None):
    """安全地获取用户输入的数字"""
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"请输入不小于 {min_val} 的数")
                continue
            if max_val is not None and value > max_val:
                print(f"请输入不大于 {max_val} 的数")
                continue
            return value
        except ValueError:
            print("请输入有效的数字！")

# 使用
age = get_number("请输入年龄: ", min_val=0, max_val=150)
print(f"你的年龄是: {age}")
```

---

## 综合案例：安全的成绩管理系统

```python
class GradeManager:
    def __init__(self):
        self.students = {}

    def add_student(self, name, score):
        """添加学生成绩"""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("学生姓名不能为空")
        if not isinstance(score, (int, float)):
            raise TypeError(f"成绩必须是数字，收到: {type(score).__name__}")
        if not 0 <= score <= 100:
            raise ValueError(f"成绩 {score} 超出范围（0-100）")

        self.students[name] = score
        print(f"✅ 添加成功: {name} - {score}分")

    def get_average(self):
        """获取平均分"""
        if not self.students:
            raise RuntimeError("没有学生数据，无法计算平均分")
        return sum(self.students.values()) / len(self.students)

    def get_student(self, name):
        """查询学生成绩"""
        if name not in self.students:
            raise KeyError(f"找不到学生: {name}")
        return self.students[name]

# 使用
gm = GradeManager()

# 安全地添加学生
test_data = [
    ("张三", 85),
    ("李四", 92),
    ("王五", "优秀"),  # 类型错误
    ("赵六", 150),     # 范围错误
    ("", 80),          # 姓名为空
    ("钱七", 78),
]

for name, score in test_data:
    try:
        gm.add_student(name, score)
    except (ValueError, TypeError) as e:
        print(f"❌ 添加失败: {e}")

# 查询
print(f"\n平均分: {gm.get_average():.1f}")

try:
    print(gm.get_student("孙八"))
except KeyError as e:
    print(f"查询失败: {e}")
```

---

## 动手练习

### 练习 1：安全的计算器

```python
def safe_calculator():
    """安全的四则运算器，能处理所有可能的错误"""
    # 1. 获取两个数字（处理非法输入）
    # 2. 获取运算符（+、-、*、/）
    # 3. 计算结果（处理除以零）
    # 4. 询问是否继续
    pass

safe_calculator()
```

### 练习 2：文件读取器

```python
def read_file_safely(filename):
    """安全地读取文件内容"""
    # 处理文件不存在、权限不足等情况
    # 返回文件内容或 None
    pass

content = read_file_safely("test.txt")
if content:
    print(content)
```

### 练习 3：批量类型转换

```python
def convert_to_numbers(data_list):
    """
    将字符串列表转换为数字列表。
    无法转换的元素用 None 替代，并记录错误信息。

    输入: ["10", "20.5", "abc", "30", "xyz"]
    输出: ([10.0, 20.5, None, 30.0, None], ["abc 无法转换", "xyz 无法转换"])
    """
    pass
```

---

## 小结

| 语法 | 作用 | 何时使用 |
|------|------|---------|
| `try` | 包裹可能出错的代码 | 任何可能出错的地方 |
| `except` | 捕获并处理异常 | 指定要处理的异常类型 |
| `else` | 没有异常时执行 | 成功后的逻辑 |
| `finally` | 始终执行 | 清理资源 |
| `raise` | 主动抛出异常 | 输入不合法、状态不对时 |
| 自定义异常 | 创建业务相关的异常 | 内置异常不够描述性时 |

:::tip 核心理解
异常处理的本质是：**预见可能的问题，准备好应对方案。** 好的程序不是不会出错，而是出错时能够**优雅地处理**——给用户友好的提示，记录错误信息，或者自动重试。这是专业开发者和初学者的重要区别。
:::
