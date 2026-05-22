---
title: "2 Python 编程基础"
description: "建立 AI 项目需要的 Python 基础：输入、数据结构、函数、文件、错误、API 和小项目。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Python入门, Python基础, Python教程, 编程入门, AI API"
---
![Python 编程基础主视觉](/img/course/ch02-python-foundation.webp)

第 2 章只解决一件事：把一个小想法写成**能运行、能保存数据、能处理错误、能讲清楚的 Python 程序**。

## 先看 Python 工作闭环

![Python AI 主线能力链](/img/course/ch02-python-ai-backbone.webp)

先看图。大多数入门 Python 程序都是这个闭环：

```text
输入 -> 数据结构 -> 函数 -> 文件/API/输出
```

Python 之所以是 AI 主线语言，是因为同一个闭环以后会变成数据清洗、模型训练、RAG 检索、API 封装和 Agent 工具。

## 学习顺序与任务表

下面这一张表同时作为本章学习指南和任务清单。

| 页面 | 跟着做 | 留下的证据 |
|---|---|---|
| [2.1.1 Python 介绍](ch01-basics/01-intro.md) 到 [2.1.5 流程控制](ch01-basics/05-control-flow.md) | 手敲变量、输入输出、条件和循环的小脚本 | 5 个改过并能输出结果的小脚本 |
| [2.1.6 数据结构](ch01-basics/06-data-structures.md) | 用列表、字典和 JSON 形状对象保存同一组任务 | 一段说明：为什么这个结构更合适 |
| [2.1.7 函数基础](ch01-basics/07-functions.md) 和 [2.1.8 模块与包](ch01-basics/08-modules.md) | 把重复逻辑拆成函数和模块 | 一个输入、返回值清楚的脚本 |
| [2.2.2 异常处理](ch02-advanced/02-exceptions.md) 和 [2.2.3 文件操作](ch02-advanced/03-file-io.md) | 保存数据、重新读取，并处理文件缺失或损坏 | 一个 JSON/文本文件和一条调试记录 |
| [2.2.1 面向对象](ch02-advanced/01-oop.md)、[2.2.5 迭代器](ch02-advanced/05-iterators-generators.md)、[2.2.6 类型提示](ch02-advanced/06-type-hints.md) | 先浏览，项目需要结构或清晰度时再回头用 | 一个重构过的函数或类 |
| [2.3.1 任务管理器](ch03-projects/01-todo-cli.md) 到 [2.3.4 AI API 体验](ch03-projects/04-ai-api-experience.md) | 做能保存数据、采集数据、提供 API、调用 AI API 的小项目 | 带 README 运行命令的项目文件夹 |
| [2.3.5 跟做工作坊](ch03-projects/05-hands-on-python-workshop.md) | 串起 CLI 命令、JSON 持久化、统计和报告导出 | `ch02_output/` 和终端输出 |

本章常见术语：

| 术语 | 含义 |
|---|---|
| `CLI` | Command-Line Interface，命令行界面：用文字命令操作的程序 |
| `I/O` | Input/Output，输入/输出：数据进入程序，结果从程序出来 |
| `JSON` | 适合保存任务、配置、API 返回结果的嵌套文本格式 |
| `API` | 一个程序调用另一个程序的入口 |
| `SDK` | 把 API 包装成更好调用函数的库 |

## 第一个可运行闭环

在一个空练习文件夹里运行下面代码。它不用第三方包，就能做一个极小的 JSON 任务管理器。

```python
import json
from pathlib import Path

DATA = Path("tasks.json")

def load_tasks():
    if not DATA.exists():
        return []
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

def save_tasks(tasks):
    DATA.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

tasks = load_tasks()
tasks.append({"title": "学习 Python 文件读写", "done": False})
save_tasks(tasks)
print(f"已保存 {len(tasks)} 个任务")
```

预期输出：

```text
已保存 1 个任务
```

连续运行两次。第二次应该输出 `已保存 2 个任务`，这说明程序已经能保存状态并重新读取。

### 如何读这个输出

- 第一次运行证明程序能创建数据文件。
- 第二次运行证明程序能读取旧状态，并追加新数据。
- `tasks.json` 才是真正的产物，打印文本只是快速确认。
- 如果第二次仍然是 `1`，优先检查当前目录和文件路径。

## 深度阶梯

| 层级 | 你能证明什么 |
|---|---|
| 最低通过 | 能写出表达式、条件、循环和函数，并得到预期输出。 |
| 项目可用 | 程序能持久化数据，能处理一条失败路径，并在 README 里说明输入和输出。 |
| 深度检查 | 能把核心逻辑和文件/API 边界分开，在有助理解的地方加类型提示，并在改代码前验证一个边界情况。 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
程序循环：输入、处理、输出，以及如有则保存的状态
代码文件：可重新运行的 Python 文件或 Notebook 单元
输出：打印结果、文件结果，或面向用户的行为
失败检查：语法、路径、类型、依赖或控制流问题
期望产出：一个可重复运行的 Python 产物，为数据和 AI 应用做准备
```

## 常见失败

| 现象 | 先检查什么 | 常见修复 |
|---|---|---|
| 语法错误 | 报错行和上一行 | 检查缩进、括号、引号和冒号 |
| 文件找不到 | 当前运行目录 | 打印 `Path.cwd()`，移动文件或改路径 |
| JSON 解析失败 | 文件是否为空或格式损坏 | 加 `try/except`，失败时回到空列表 |
| 函数看不懂 | 输入、返回值和隐藏的全局状态 | 拆成职责单一的小函数 |
| API 调用失败 | 参数、状态码和返回的错误内容 | 安全打印响应，并处理失败分支 |

## 通关检查

能回答下面五个问题，就可以进入第 3 章：

- 什么数据进入程序？什么结果离开程序？
- 什么时候字典比列表更合适？
- 文件路径是相对于哪个文件夹？
- `print` 和 `return` 有什么区别？
- 其他人能不能按 README 运行你的项目？

<details>
<summary>检查思路与讲解</summary>

1. 程序输入可以来自命令行文本、用户输入、文件或 API 响应；输出可以是打印文本、函数返回值、保存的文件，或发送给另一个程序的响应。
2. 当每个项目需要命名字段，或需要按键快速查找时，字典更合适；当你需要保存一组有顺序的相似项目时，列表更合适。
3. 相对路径以当前工作目录为基准，不一定是脚本所在目录。可以用 `Path.cwd()` 和 `Path(__file__).resolve()` 分别确认。
4. `print()` 是把信息显示给人看，返回值通常是 `None`；`return` 是把值交还给调用者，方便继续使用、测试或保存。
5. README 合格的标准是：一个新终端可以安装依赖、运行命令，并复现预期输出，不需要猜隐藏步骤。

</details>

需要打印式清单时，打开 [2.0 学习指南与任务单](./study-guide.md)。下一章会继续用 Python 处理 CSV、分析数据并连接数据库。
