---
title: "1.1 Python 简介"
sidebar_position: 1
description: "了解 Python 语言的特点、应用领域和开发环境"
---

# Python 简介

## 学习目标

- 了解 Python 是什么，为什么它这么受欢迎
- 理解 Python 在 AI 领域的核心地位
- 写出并运行你的第一个 Python 程序
- 理解 Python 代码的基本结构

---

## 为什么学 Python？

如果编程语言是工具，那 Python 就是**瑞士军刀**——什么都能干，而且上手简单。

先看几个数据：

| 维度 | 说明 |
|------|------|
| **流行度** | 连续多年蝉联 TIOBE 编程语言排行榜第一 |
| **AI 首选** | 几乎所有 AI/机器学习框架（PyTorch、TensorFlow）都以 Python 为主 |
| **就业市场** | 数据科学、AI 工程师、后端开发岗位的必备技能 |
| **学习曲线** | 语法接近自然语言，初学者最容易上手的语言之一 |

一句话总结：**如果你想做 AI，Python 是唯一的起点。**

---

## Python 到底是什么？

Python 是一门**高级编程语言**，由 Guido van Rossum（吉多·范罗苏姆）于 1991 年发布。

"高级"是什么意思？编程语言离硬件越远、越接近人类语言，就越"高级"。比较一下：

```
# 机器语言（二进制，计算机直接执行）
10110000 01100001

# C 语言（需要手动管理很多细节）
#include <stdio.h>
int main() {
    printf("Hello World\n");
    return 0;
}

# Python（简洁明了）
print("Hello World")
```

同样是打印一句话，Python 只需要 **1 行**，而 C 语言需要 5 行。这就是 Python 的设计哲学：**简洁优雅，让你专注于解决问题，而不是语法细节。**

### Python 的核心特点

| 特点 | 说明 | 对你的好处 |
|------|------|-----------|
| **语法简洁** | 用缩进代替大括号，代码像英语 | 学得快，写得少 |
| **解释型语言** | 写完直接运行，不需要编译 | 调试方便，马上看结果 |
| **动态类型** | 不需要声明变量类型 | 代码更简短灵活 |
| **生态丰富** | 超过 40 万个第三方库 | 别人造好的轮子，拿来就用 |
| **跨平台** | Windows、macOS、Linux 都能跑 | 一份代码，到处运行 |

---

## Python 能做什么？

Python 的应用范围非常广泛，以下是几个最重要的方向：

### 1. AI 和机器学习（这门课的核心）

```python
# 用 3 行代码训练一个机器学习模型
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
```

主流框架：PyTorch、TensorFlow、scikit-learn、Hugging Face Transformers

### 2. 数据分析和可视化

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv("sales.csv")

# 一行代码画图
data.plot(x="month", y="revenue", kind="bar")
plt.show()
```

主流库：pandas、NumPy、Matplotlib、Seaborn

### 3. Web 后端开发

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def say_hello():
    return {"message": "你好，世界！"}
```

主流框架：FastAPI、Django、Flask

### 4. 自动化脚本

```python
import os

# 批量重命名文件夹中的所有图片
for i, filename in enumerate(os.listdir("photos/")):
    new_name = f"photo_{i+1}.jpg"
    os.rename(f"photos/{filename}", f"photos/{new_name}")
```

### 5. 网络爬虫

```python
import requests
from bs4 import BeautifulSoup

page = requests.get("https://example.com")
soup = BeautifulSoup(page.text, "html.parser")
title = soup.find("h1").text
print(f"网页标题: {title}")
```

---

## 写你的第一个 Python 程序

### 方式一：在终端中使用 Python 交互模式

打开终端（在 stage0 里你已经学过了），输入：

```bash
python
```

你会看到类似这样的提示符：

```
Python 3.11.5 (main, Sep 11 2023, 08:31:25)
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

`>>>` 就是 Python 的交互提示符，表示它在等你输入命令。

试试这些：

```python
>>> print("Hello, World!")
Hello, World!

>>> 1 + 1
2

>>> "AI" * 3
'AIAIAI'

>>> len("Python")
6
```

输入 `exit()` 或按 `Ctrl+D` 退出交互模式。

:::tip 交互模式的用途
交互模式非常适合**快速实验**——比如你不确定一个函数怎么用，可以先在交互模式里试试，确认没问题再写到文件里。
:::

### 方式二：在 VS Code 中编写并运行

1. 打开 VS Code（在 stage0 里已经装好了）
2. 创建一个新文件 `hello.py`（注意后缀是 `.py`）
3. 输入以下代码：

```python
# 这是我的第一个 Python 程序
print("Hello, World!")
print("我正在学习 Python！")
print("1 + 1 =", 1 + 1)
```

4. 保存文件（`Ctrl+S` / `Cmd+S`）
5. 在终端中运行：

```bash
python hello.py
```

输出：

```
Hello, World!
我正在学习 Python！
1 + 1 = 2
```

恭喜你，你的第一个 Python 程序诞生了！

### 方式三：在 Jupyter Notebook 中运行

在 stage0 里你已经装了 Jupyter。打开它：

```bash
jupyter notebook
```

创建一个新 Notebook，在代码单元格里输入 `print("Hello from Jupyter!")` 然后按 `Shift+Enter` 运行。

:::info 三种方式怎么选？
- **交互模式**：快速测试一小段代码
- **VS Code + .py 文件**：写正式的项目代码
- **Jupyter Notebook**：数据分析、学习实验（我们这门课主要用这种）
:::

---

## Python 代码的基本规则

在深入学习之前，先了解几个最基本的规则：

### 1. 缩进很重要

Python 用**缩进**（通常是 4 个空格）来表示代码块，而不是像其他语言那样用大括号 `{}`。

```python
# 正确 ✅
if True:
    print("缩进了 4 个空格")
    print("同一个代码块")

# 错误 ❌ —— 会报 IndentationError
if True:
print("没有缩进，Python 会报错")
```

:::caution 注意
缩进错误是新手最常见的错误。VS Code 会帮你自动缩进，但如果你复制粘贴代码，要注意检查缩进是否正确。
:::

### 2. 注释用 `#`

```python
# 这是一行注释，Python 会忽略它
print("这行代码会执行")  # 行尾也可以写注释

# 多行注释就是多个 # 开头的行
# 第一行注释
# 第二行注释
```

注释是写给人看的，帮助你（和别人）理解代码。好的注释解释**为什么**这么做，而不是**做了什么**。

### 3. 大小写敏感

```python
name = "Alice"
Name = "Bob"
NAME = "Charlie"
# 这是三个不同的变量！

print(name)   # Alice
Print(name)   # 报错！Python 没有 Print，只有 print
```

### 4. 文件以 `.py` 结尾

Python 脚本文件的后缀是 `.py`，比如 `hello.py`、`train.py`、`model.py`。

---

## Python 2 还是 Python 3？

简短的回答：**用 Python 3，不要用 Python 2。**

Python 2 已经在 2020 年 1 月 1 日正式停止维护。所有新项目、所有现代库都只支持 Python 3。本课程使用 **Python 3.10+**。

确认你的 Python 版本：

```bash
python --version
# 应该输出 Python 3.10.x 或更高
```

如果输出是 `Python 2.x`，你需要使用 `python3` 命令，或者检查你在 stage0 中的 conda 环境是否正确激活。

---

## 动手练习

### 练习 1：Hello World 升级版

创建文件 `about_me.py`，让它输出你的个人介绍：

```python
print("=== 个人介绍 ===")
print("姓名：[你的名字]")
print("目标：成为 AI 工程师")
print("正在学习：Python 编程")
print("=" * 20)
```

运行它，看看输出。试着修改内容，加上更多信息。

### 练习 2：Python 当计算器

在 Python 交互模式中，尝试以下运算：

```python
>>> 100 + 200
>>> 10 * 3.14
>>> 2 ** 10        # ** 是幂运算，2的10次方
>>> 17 / 5         # 除法
>>> 17 // 5        # 整除（去掉小数部分）
>>> 17 % 5         # 取余数
```

记下每个运算的结果，想想为什么。

### 练习 3：探索 print()

试试以下代码，观察 `print()` 的不同用法：

```python
print("Hello")
print("Hello", "World")           # 多个参数用逗号分隔
print("Hello", "World", sep="-")  # 用 - 连接
print("Hello", end=" ")           # 不换行
print("World")
print("价格:", 99.9, "元")
```

---

## 小结

| 要点 | 说明 |
|------|------|
| Python 是 AI 开发的首选语言 | 几乎所有 AI 框架都基于 Python |
| 语法简洁，接近自然语言 | 降低学习门槛，让你专注于逻辑 |
| 生态丰富 | 40 万+ 第三方库，绝大部分需求都有现成方案 |
| 三种运行方式 | 交互模式、.py 文件、Jupyter Notebook |
| 缩进是 Python 的灵魂 | 用 4 个空格缩进，不要用 Tab |

:::tip 学习建议
编程是一门**手艺**，光看不练是学不会的。每一节课的练习都要动手敲一遍——不是复制粘贴，而是一个字一个字敲出来。打字的过程中你会犯错、会调试、会理解得更深。
:::
