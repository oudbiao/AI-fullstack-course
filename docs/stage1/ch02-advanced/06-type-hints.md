---
title: "1.6 类型注解与代码质量"
sidebar_position: 6
description: "掌握 Python 类型注解和代码质量工具"
---

# 类型注解与代码质量

## 学习目标

- 理解为什么需要类型注解
- 掌握 Python 类型注解的基本语法
- 了解常用的代码质量工具（linter、formatter）
- 养成编写高质量代码的习惯

---

## 为什么需要类型注解？

Python 是动态类型语言——变量不需要声明类型。这让写代码很灵活，但也带来一个问题：

```python
def calculate_total(items, tax):
    return sum(items) * (1 + tax)

# 用的时候，你得猜：
# items 是什么？列表？元组？
# tax 是什么？0.1？还是 "10%"？
# 返回什么？数字？字符串？
```

当项目变大时，没有类型信息的代码就像**没有路标的高速公路**——你得靠猜。

类型注解的作用：

| 好处 | 说明 |
|------|------|
| **自文档化** | 一看就知道函数要什么参数、返回什么 |
| **IDE 智能提示** | VS Code 能给出更准确的自动补全 |
| **静态检查** | 在运行前就发现类型错误 |
| **团队协作** | 降低沟通成本，代码自己说话 |

---

## 基本类型注解

### 变量注解

```python
# 基本类型
name: str = "小明"
age: int = 25
height: float = 1.75
is_student: bool = True

# Python 不会强制检查类型注解
# 以下代码不会报错，但静态检查工具会警告
age: int = "二十五"  # 类型注解说是 int，实际赋了 str
```

:::info 类型注解只是"建议"
Python 的类型注解**不会在运行时强制执行**。即使类型不匹配，程序也能运行。它主要是给**开发者和工具**看的。真正的类型检查需要用 mypy 等静态检查工具。
:::

### 函数注解

```python
def greet(name: str) -> str:
    """
    name: str  → 参数 name 的类型是 str
    -> str     → 返回值的类型是 str
    """
    return f"你好，{name}！"

def calculate_bmi(weight: float, height: float) -> float:
    """计算 BMI"""
    return weight / (height ** 2)

def train_model(epochs: int = 10, lr: float = 0.001) -> None:
    """返回 None 的函数"""
    print(f"训练 {epochs} 轮，学习率 {lr}")
```

有了类型注解，VS Code 的智能提示会变得非常准确——当你输入 `greet(` 时，它会提示你参数类型是 `str`。

---

## 复合类型注解

### 列表和字典

```python
# Python 3.9+：直接用内置类型
scores: list[int] = [85, 92, 78]
student: dict[str, int] = {"张三": 85, "李四": 92}
coordinates: tuple[float, float] = (3.14, 2.71)
unique_ids: set[int] = {1, 2, 3}

# Python 3.8 及更早：需要从 typing 导入
from typing import List, Dict, Tuple, Set

scores: List[int] = [85, 92, 78]
student: Dict[str, int] = {"张三": 85, "李四": 92}
```

### Optional：可能为 None 的值

```python
from typing import Optional

def find_student(name: str) -> Optional[dict]:
    """查找学生，找不到返回 None"""
    students = {"张三": {"age": 20}, "李四": {"age": 21}}
    return students.get(name)

# Python 3.10+ 可以用更简洁的写法
def find_student(name: str) -> dict | None:
    ...
```

### Union：多种可能的类型

```python
from typing import Union

def process(data: Union[str, list]) -> str:
    """接受字符串或列表"""
    if isinstance(data, list):
        return ", ".join(str(item) for item in data)
    return data

# Python 3.10+ 的简洁写法
def process(data: str | list) -> str:
    ...
```

### Callable：函数类型

```python
from typing import Callable

def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
    """接受一个函数作为参数"""
    return func(a, b)

result = apply_func(lambda x, y: x + y, 3, 5)  # 8
```

### 更多类型注解

```python
from typing import Any, Literal

# Any：任意类型
def log(message: Any) -> None:
    print(message)

# Literal：只接受特定的值
def set_mode(mode: Literal["train", "eval", "test"]) -> None:
    print(f"模式: {mode}")

set_mode("train")   # ✅
set_mode("play")    # 静态检查会警告
```

---

## 类型注解实战

### 为函数添加类型注解

```python
def analyze_scores(
    scores: list[float],
    subject: str = "未知",
    pass_line: float = 60.0
) -> dict[str, float | int | str]:
    """分析成绩，返回统计信息"""
    if not scores:
        return {"error": "成绩列表为空"}

    return {
        "subject": subject,
        "count": len(scores),
        "average": sum(scores) / len(scores),
        "max": max(scores),
        "min": min(scores),
        "pass_count": sum(1 for s in scores if s >= pass_line)
    }
```

### 为类添加类型注解

```python
class DataProcessor:
    def __init__(self, name: str, data: list[dict[str, Any]]) -> None:
        self.name: str = name
        self.data: list[dict[str, Any]] = data
        self._processed: bool = False

    def filter_by(self, key: str, value: Any) -> list[dict[str, Any]]:
        """按条件过滤数据"""
        return [item for item in self.data if item.get(key) == value]

    def get_column(self, key: str) -> list[Any]:
        """提取某一列"""
        return [item[key] for item in self.data if key in item]
```

---

## 代码质量工具

好的代码不仅要能运行，还要**可读、规范、少 bug**。以下工具帮你做到这一点。

### 代码格式化：black

`black` 是 Python 最流行的代码格式化工具，它会自动帮你把代码格式化成统一风格。

```bash
# 安装
pip install black

# 格式化单个文件
black my_script.py

# 格式化整个目录
black src/

# 只检查不修改
black --check my_script.py
```

格式化前：

```python
x = {  'a':37,'b':42,
'c':927}
y = 'hello ''world'
z = 'hello '+'world'
a = [1,2,3,4,5,]
```

格式化后：

```python
x = {"a": 37, "b": 42, "c": 927}
y = "hello " "world"
z = "hello " + "world"
a = [1, 2, 3, 4, 5]
```

### 代码检查：ruff

`ruff` 是新一代的 Python linter，速度极快，能发现很多常见问题。

```bash
# 安装
pip install ruff

# 检查代码
ruff check my_script.py

# 自动修复
ruff check --fix my_script.py

# 格式化（ruff 也可以替代 black）
ruff format my_script.py
```

### 类型检查：mypy

```bash
# 安装
pip install mypy

# 检查类型
mypy my_script.py
```

```python
# example.py
def add(a: int, b: int) -> int:
    return a + b

result = add("hello", "world")  # mypy 会报错：参数类型不对！
```

```bash
$ mypy example.py
example.py:4: error: Argument 1 to "add" has incompatible type "str"; expected "int"
```

### VS Code 集成

在 VS Code 中安装以下扩展，可以**实时**看到代码质量问题：

| 扩展 | 功能 |
|------|------|
| **Pylance** | 类型检查和智能提示（VS Code 自带推荐） |
| **Ruff** | 实时代码检查 |
| **Black Formatter** | 保存时自动格式化 |

推荐在 VS Code 设置中添加：

```json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

---

## Python 代码风格指南（PEP 8）

PEP 8 是 Python 官方的代码风格指南，以下是最重要的几条：

### 命名规范

```python
# 变量和函数：小写加下划线（snake_case）
student_name = "张三"
def calculate_average(scores):
    pass

# 类：首字母大写（PascalCase）
class DataProcessor:
    pass

# 常量：全大写加下划线
MAX_RETRY = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# "私有"属性：前缀下划线
class MyClass:
    def __init__(self):
        self._internal_state = None
```

### 空行和空格

```python
# 函数之间空两行
def function_one():
    pass


def function_two():
    pass


# 类之间空两行
class ClassOne:
    pass


class ClassTwo:
    pass

# 运算符周围加空格
x = 1 + 2       # ✅
x = 1+2          # ❌

# 逗号后面加空格
items = [1, 2, 3]     # ✅
items = [1,2,3]        # ❌

# 函数参数的默认值不加空格
def func(x=10):       # ✅
def func(x = 10):     # ❌
```

### 每行长度

```python
# 单行不超过 79 字符（或 88/120 字符，取决于团队规范）

# 太长的行可以用括号换行
result = (
    first_variable
    + second_variable
    + third_variable
)

# 函数参数太多时
def complex_function(
    param1: str,
    param2: int,
    param3: float = 0.0,
    param4: bool = True,
) -> dict:
    pass
```

---

## 编写文档字符串（docstring）

好的文档字符串让别人（和未来的你）能快速理解代码：

```python
def train_model(
    data: list[dict],
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> dict[str, float]:
    """
    训练模型并返回训练指标。

    Args:
        data: 训练数据列表，每个元素是一个样本字典
        epochs: 训练轮数，默认 100
        learning_rate: 学习率，默认 0.001
        batch_size: 批次大小，默认 32

    Returns:
        包含训练指标的字典，例如:
        {"accuracy": 0.95, "loss": 0.05}

    Raises:
        ValueError: 当 data 为空时
        RuntimeError: 当 GPU 不可用时

    Example:
        >>> result = train_model(data, epochs=50)
        >>> print(result["accuracy"])
        0.95
    """
    if not data:
        raise ValueError("训练数据不能为空")
    # ... 训练逻辑 ...
    return {"accuracy": 0.95, "loss": 0.05}
```

---

## 动手练习

### 练习 1：为旧代码添加类型注解

给以下代码添加完整的类型注解：

```python
def process_students(students, min_score):
    results = []
    for student in students:
        if student["score"] >= min_score:
            results.append({
                "name": student["name"],
                "score": student["score"],
                "passed": True
            })
    return results

def calculate_stats(numbers):
    if not numbers:
        return None
    return {
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers),
        "count": len(numbers)
    }
```

### 练习 2：安装和使用代码质量工具

```bash
# 1. 安装 ruff
pip install ruff

# 2. 创建一个有格式问题的 Python 文件

# 3. 运行 ruff check 查看问题

# 4. 运行 ruff format 自动格式化

# 5. 对比前后差异
```

### 练习 3：编写高质量代码

用你学到的所有规范，重写以下"糟糕"的代码：

```python
# 糟糕的代码
def f(l,n):
 r=[]
 for x in l:
  if x>n:r.append(x)
 return r

def g(d):
 s=0
 for k in d:s+=d[k]
 return s/len(d)
```

要求：
1. 改成有意义的命名
2. 添加类型注解
3. 添加文档字符串
4. 符合 PEP 8 规范

---

## 小结

| 工具/概念 | 作用 | 推荐程度 |
|-----------|------|---------|
| **类型注解** | 标注参数和返回值类型 | 强烈推荐 |
| **PEP 8** | Python 代码风格规范 | 必须遵循 |
| **black / ruff format** | 自动格式化代码 | 强烈推荐 |
| **ruff** | 代码质量检查 | 强烈推荐 |
| **mypy** | 静态类型检查 | 推荐 |
| **docstring** | 文档字符串 | 公开函数必须有 |

:::tip 核心理解
代码是写给人看的，顺便让机器执行。类型注解和代码规范不会让你的代码跑得更快，但会让你的代码**更容易被理解、被维护、被协作**。在 AI 项目中，一个人写的代码往往需要多人使用和修改——养成好习惯，从现在开始。
:::
