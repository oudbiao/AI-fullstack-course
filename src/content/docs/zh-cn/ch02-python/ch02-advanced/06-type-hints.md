---
title: "2.2.6 类型注解与代码质量"
description: "掌握 Python 类型注解和代码质量工具"
sidebar:
  order: 6
---
![类型注解与代码质量流程图](/img/course/ch02-type-hints-quality-flow.webp)

## 本节定位

这一节把注意力从“代码能跑”推进到“代码好维护”。类型注解、格式化工具和代码检查工具能让你在项目变大、多人协作、调用复杂库时少犯错，也能让未来的自己更快读懂代码。

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
feature_name: str = "登录 API"
retry_count: int = 3
latency_ms: float = 125.5
is_enabled: bool = True

# Python 不会强制检查类型注解
# 以下代码不会报错，但静态检查工具会警告
retry_count: int = "三次"  # 类型注解说是 int，实际赋了 str
```

:::note[类型注解只是"建议"]
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
estimated_hours: list[int] = [8, 12, 5]
task_hours: dict[str, int] = {"登录 API": 8, "RAG 演示": 12}
coordinates: tuple[float, float] = (3.14, 2.71)
unique_ids: set[int] = {1, 2, 3}

# Python 3.8 及更早：需要从 typing 导入
from typing import List, Dict, Tuple, Set

estimated_hours: List[int] = [8, 12, 5]
task_hours: Dict[str, int] = {"登录 API": 8, "RAG 演示": 12}
```

### Optional：可能为 None 的值

```python
from typing import Optional

def find_task(name: str) -> Optional[dict]:
    """查找任务，找不到返回 None"""
    tasks = {"登录 API": {"hours": 8}, "RAG 演示": {"hours": 12}}
    return tasks.get(name)

# Python 3.10+ 可以用更简洁的写法
def find_task(name: str) -> dict | None:
    tasks = {"登录 API": {"hours": 8}, "RAG 演示": {"hours": 12}}
    return tasks.get(name)
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
    if isinstance(data, list):
        return ", ".join(str(item) for item in data)
    return data
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
def analyze_latencies(
    latencies: list[float],
    endpoint: str = "未知",
    slow_line: float = 800.0
) -> dict[str, float | int | str]:
    """分析 API 延迟，返回统计信息"""
    if not latencies:
        return {"error": "延迟列表为空"}

    return {
        "endpoint": endpoint,
        "count": len(latencies),
        "average": sum(latencies) / len(latencies),
        "max": max(latencies),
        "min": min(latencies),
        "slow_count": sum(1 for ms in latencies if ms >= slow_line)
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
| **Ruff** | 实时代码检查，也可以负责格式化 |
| **Black Formatter** | 如果团队选择 Black 作为格式化器，就用它保存时自动格式化 |

新项目为了减少工具数量，可以让 Ruff 同时负责检查和格式化。在 VS Code 设置中添加：

```json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

如果团队已经统一使用 Black，就让 Black Formatter 做默认格式化器，Ruff 只负责 lint 和 import 清理。不要让 Ruff 和 Black 同时抢同一批 Python 文件的默认格式化职责。

---

## Python 代码风格指南（PEP 8）

PEP 8 是 Python 官方的代码风格指南，以下是最重要的几条：

### 命名规范

```python
# 变量和函数：小写加下划线（snake_case）
feature_name = "登录 API"
def calculate_average_latency(latencies):
    return sum(latencies) / len(latencies)

# 类：首字母大写（PascalCase）
class DataProcessor:
    def __init__(self, source: str):
        self.source = source

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
    return "function one"


def function_two():
    return "function two"


# 类之间空两行
class ClassOne:
    value = 1


class ClassTwo:
    value = 2

# 运算符周围加空格
x = 1 + 2       # ✅
x = 1+2          # ❌

# 逗号后面加空格
items = [1, 2, 3]     # ✅
items = [1,2,3]        # ❌

# 函数参数的默认值不加空格。
# 第二种写法语法上能运行，但不符合 PEP 8 推荐风格。
def func(x=10):       # ✅
    return x

def func_not_recommended(x = 10):  # ❌ 只是风格不推荐
    return x
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
    return {
        "param1": param1,
        "param2": param2,
        "param3": param3,
        "param4": param4,
    }
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
    total = sum(len(str(sample)) for sample in data)
    accuracy = min(0.99, 0.5 + total / 1000)
    return {"accuracy": accuracy, "loss": 1 - accuracy}
```

---

## 动手练习

### 练习 1：为旧代码添加类型注解

给以下代码添加完整的类型注解：

```python
def process_tasks(tasks, max_hours):
    results = []
    for task in tasks:
        if task["hours"] <= max_hours:
            results.append({
                "name": task["name"],
                "hours": task["hours"],
                "ready": True
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

<details>
<summary>参考实现与讲解</summary>

1. 旧代码应补全显式参数和返回类型，例如 `process_tasks(tasks: list[dict[str, int | str]], max_hours: int) -> list[dict[str, object]]` 和 `calculate_stats(numbers: Sequence[float]) -> dict[str, float] | None`。重点是让输入结构和空列表情况一眼可见。
2. `ruff` 的流程应是先 `ruff check`，再 `ruff format`，最后比较 diff。这样可以把 lint 和格式化分开，审阅更清楚。
3. 重写后的代码应使用清晰命名、类型注解、docstring 和 PEP 8 间距；求平均值的函数还要防止空输入导致除零。

</details>

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
模式：类、异常、文件 IO、函数式流水线、生成器或类型提示
代码产物：最小可运行示例和一个真实使用场景
输出：打印的对象状态、捕获的错误、保存的文件、yield 的值，或类型检查备注
失败检查：隐藏变异、吞掉异常、文件路径问题、懒迭代器混淆或误导性标注
期望产出：带调试说明的小型高级 Python 示例
```

## 小结

| 工具/概念 | 作用 | 推荐程度 |
|-----------|------|---------|
| **类型注解** | 标注参数和返回值类型 | 强烈推荐 |
| **PEP 8** | Python 代码风格规范 | 必须遵循 |
| **black / ruff format** | 自动格式化代码 | 强烈推荐 |
| **ruff** | 代码质量检查 | 强烈推荐 |
| **mypy** | 静态类型检查 | 推荐 |
| **docstring** | 文档字符串 | 公开函数必须有 |

:::tip[核心理解]
代码是写给人看的，顺便让机器执行。类型注解和代码规范不会让你的代码跑得更快，但会让你的代码**更容易被理解、被维护、被协作**。在 AI 项目中，一个人写的代码往往需要多人使用和修改——养成好习惯，从现在开始。
:::