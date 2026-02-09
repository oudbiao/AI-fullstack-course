---
title: "Jupyter Notebook"
sidebar_position: 3
description: "掌握数据分析和 AI 实验的标配工具"
---

# Jupyter Notebook

## 学习目标

- 理解 Jupyter Notebook 是什么、适合做什么
- 安装和启动 Jupyter Notebook
- 掌握 Cell 类型和基本操作
- 熟悉最常用的快捷键
- 学会使用魔法命令
- 理解 Notebook 和 `.py` 文件的区别

---

## Jupyter Notebook 是什么？

**Jupyter Notebook 是一种交互式编程环境**——你可以写一段代码、立刻运行、看到结果，然后再写下一段。代码、输出、图表、文字说明全部混在一个文件里。

### 它长什么样？

想象一个笔记本，每一页（叫做 **Cell**）可以是：
- 一段可以运行的代码
- 一段 Markdown 文字（标题、说明、公式）
- 代码运行后的输出（数字、表格、图表）

它们按顺序排列在一起，形成一个"可运行的文档"。

### 什么场景最适合用 Jupyter？

| 场景 | 用 Jupyter | 用 .py 文件 |
|------|:---------:|:----------:|
| 探索性数据分析（EDA） | ✅ 最佳 | ❌ |
| 画图和可视化 | ✅ 图直接显示在下方 | ❌ 需要弹窗 |
| 学习和实验 | ✅ 逐步运行，边学边试 | ❌ |
| 展示成果（给老板看） | ✅ 代码+图+文字一体 | ❌ |
| 正式项目代码 | ❌ | ✅ 更好维护 |
| 调试复杂程序 | ❌ | ✅ |
| 团队协作 | ❌ 合并冲突多 | ✅ |

一句话：**学习和实验用 Jupyter，写正式代码用 .py 文件。** 本课程的前几个阶段会大量使用 Jupyter。

---

## 安装和启动

### 安装

确保你在正确的 conda 环境里：

```bash
conda activate ai-course

# 安装 Jupyter Notebook
pip install jupyter

# （可选）安装 JupyterLab——Jupyter 的增强版，界面更现代
pip install jupyterlab
```

### 启动

```bash
# 启动 Jupyter Notebook（经典版）
jupyter notebook

# 或启动 JupyterLab（推荐）
jupyter lab
```

运行后，终端会输出类似这样的信息：

```
[I 10:00:00 NotebookApp] Serving notebooks from local directory: /Users/zhangsan
[I 10:00:00 NotebookApp] http://localhost:8888/?token=abc123...
```

浏览器会自动打开，你就能看到 Jupyter 的界面了。

:::tip 在 VS Code 里用 Jupyter
如果你安装了 VS Code 的 Jupyter 扩展，可以直接在 VS Code 里创建和运行 `.ipynb` 文件，不需要启动浏览器版本。新建一个 `.ipynb` 文件就行。后续课程中两种方式都可以用。
:::

### 创建新 Notebook

在 Jupyter 界面里：
1. 点击右上角的 **New → Python 3**（经典版）
2. 或点击左侧的 **+** 号，选择 **Python 3 Notebook**（JupyterLab）

一个新的空白 Notebook 就创建好了。

---

## Cell（单元格）基础

Notebook 由一个一个的 **Cell** 组成。每个 Cell 有两种类型：

### Code Cell（代码单元格）

用来写和运行 Python 代码：

```python
# Cell 1：定义变量
name = "AI 全栈学习"
year = 2026
```

按 `Shift + Enter` 运行。

```python
# Cell 2：使用上面定义的变量
print(f"欢迎来到 {name} 课程！现在是 {year} 年。")
```

输出：

```
欢迎来到 AI 全栈学习教程！现在是 2026 年。
```

**重要特性：** Cell 之间共享变量。你在 Cell 1 里定义的 `name`，在 Cell 2 里可以直接用。

### Markdown Cell（文字单元格）

用来写文字说明、标题、列表、公式等。切换方法：
- 选中 Cell，按 `M` 键切换为 Markdown
- 按 `Y` 键切换回 Code

Markdown Cell 里可以写：

```markdown
## 第一步：加载数据

我们使用 **Iris 数据集**进行探索性分析。

- 150 个样本
- 4 个特征
- 3 个类别

数学公式：$y = wx + b$
```

运行后会渲染成漂亮的格式化文字。

### 案例：一个典型的数据分析 Notebook 结构

```
[Markdown]  # Iris 数据集探索性分析
[Markdown]  ## 1. 导入库
[Code]      import numpy as np
            import pandas as pd
            import matplotlib.pyplot as plt

[Markdown]  ## 2. 加载数据
[Code]      from sklearn.datasets import load_iris
            iris = load_iris()
            df = pd.DataFrame(iris.data, columns=iris.feature_names)
            df['species'] = iris.target
            df.head()

[Output]    （显示一个表格）

[Markdown]  ## 3. 数据概览
[Code]      df.describe()

[Output]    （显示统计摘要表格）

[Markdown]  ## 4. 可视化
[Code]      plt.figure(figsize=(10, 6))
            plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['species'])
            plt.xlabel('sepal length')
            plt.ylabel('sepal width')
            plt.title('Iris Dataset')
            plt.show()

[Output]    （直接显示散点图）

[Markdown]  ## 5. 结论
            花瓣长度和花瓣宽度是区分三个品种最有效的特征。
```

代码、图表、文字解释，全在一个文件里。这就是 Jupyter 的魅力。

---

## 快捷键

Jupyter 有两种模式：

- **命令模式**（Cell 外框是蓝色）：按 `Esc` 进入，用于管理 Cell
- **编辑模式**（Cell 外框是绿色）：按 `Enter` 进入，用于编辑内容

### 命令模式快捷键（按 Esc 后使用）

| 快捷键 | 操作 |
|:---:|------|
| `Shift + Enter` | 运行当前 Cell 并跳到下一个（最最最常用） |
| `Ctrl + Enter` | 运行当前 Cell 但不跳转 |
| `A` | 在上方插入新 Cell |
| `B` | 在下方插入新 Cell |
| `DD`（连按两次 D） | 删除当前 Cell |
| `M` | 把当前 Cell 改为 Markdown |
| `Y` | 把当前 Cell 改为 Code |
| `Z` | 撤销删除 Cell |
| `↑` / `↓` | 上下移动选中不同的 Cell |

### 编辑模式快捷键（按 Enter 后使用）

| 快捷键 | 操作 |
|:---:|------|
| `Shift + Enter` | 运行并跳到下一个 |
| `Tab` | 代码补全 |
| `Shift + Tab` | 显示函数文档 |
| `Ctrl + /` | 注释/取消注释 |
| `Ctrl + Z` | 撤销 |

### 实操：练习快捷键

创建一个新 Notebook，然后：

1. 在第一个 Cell 里输入 `print("Cell 1")`，按 `Shift + Enter` 运行
2. 按 `B` 在下方新建一个 Cell
3. 输入 `print("Cell 2")`，按 `Ctrl + Enter` 运行（注意光标不跳转）
4. 按 `Esc` 回到命令模式
5. 按 `A` 在上方插入 Cell
6. 按 `M` 切换为 Markdown，输入 `# 我的标题`，按 `Shift + Enter` 渲染
7. 选中一个不需要的 Cell，按 `DD` 删除

反复练习几次，很快就能形成肌肉记忆。

---

## 魔法命令

Jupyter 提供了一些以 `%` 或 `!` 开头的特殊命令，叫做"魔法命令"。它们可以做到普通 Python 代码做不到的事。

### `!` 命令：在 Cell 里执行终端命令

```python
# 安装包（不需要切到终端）
!pip install seaborn

# 查看当前目录
!ls

# 查看 Python 版本
!python --version

# 下载文件
!wget https://example.com/data.csv
```

### `%timeit`：测量代码运行时间

```python
import numpy as np

# 测量一行代码的运行时间
%timeit np.random.rand(1000, 1000)
# 输出: 5.23 ms ± 128 µs per loop
```

```python
%%timeit
# 测量整个 Cell 的运行时间（注意是两个 %）
data = np.random.rand(1000, 1000)
result = np.dot(data, data.T)
# 输出: 15.6 ms ± 1.2 ms per loop
```

### `%matplotlib inline`：让图表显示在 Notebook 里

```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
plt.plot(x, np.sin(x))
plt.title("正弦函数")
plt.show()
# 图表直接显示在 Cell 下方
```

:::info
在较新的 Jupyter 版本中，`%matplotlib inline` 通常是默认行为，可以省略。但写上也不会错。
:::

### `%who`：查看当前定义的变量

```python
name = "张三"
age = 25
scores = [90, 85, 92]

%who
# 输出: age   name   scores

%whos
# 输出变量的详细信息（类型、值）
```

### 常用魔法命令速查

| 命令 | 用途 |
|------|------|
| `!命令` | 执行终端命令 |
| `%timeit` | 测量一行代码运行时间 |
| `%%timeit` | 测量整个 Cell 运行时间 |
| `%matplotlib inline` | 图表内联显示 |
| `%who` / `%whos` | 查看当前变量 |
| `%reset` | 清除所有变量（重新开始） |
| `%pwd` | 显示当前目录 |
| `%history` | 显示输入历史 |

---

## Notebook vs .py 文件

### 什么时候用 Notebook？

- 数据分析、EDA
- 学习新库、做实验
- 画图和可视化
- 给别人展示（如 Kaggle Notebook）
- 写教程

### 什么时候用 .py 文件？

- 正式项目代码（模型定义、训练脚本、API 服务）
- 需要被其他文件 import 的模块
- 需要在命令行用参数运行的脚本
- 团队协作的代码

### 一个典型的 AI 项目，两者配合使用

```
my-ai-project/
├── notebooks/
│   ├── 01_eda.ipynb          # 探索数据（Notebook）
│   ├── 02_experiment.ipynb   # 实验不同模型（Notebook）
│   └── 03_analysis.ipynb     # 分析结果（Notebook）
├── src/
│   ├── model.py              # 模型定义（.py）
│   ├── train.py              # 训练脚本（.py）
│   ├── evaluate.py           # 评估脚本（.py）
│   └── utils.py              # 工具函数（.py）
├── data/
├── models/
├── requirements.txt
└── README.md
```

先在 Notebook 里做实验，确定方案后把代码整理到 `.py` 文件中——这是 AI 工程师的标准工作流。

### 在 Notebook 里调用 .py 文件的代码

```python
# 在 Notebook 中 import 自己写的模块
import sys
sys.path.append('../src')  # 把 src 目录加入路径

from model import SimpleCNN
from utils import accuracy

model = SimpleCNN()
print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")
```

---

## 实操练习

创建一个 Notebook，完成以下练习：

**Cell 1（Markdown）：**
```markdown
# 我的第一个 Jupyter Notebook
今天的日期：2026 年 X 月 X 日
```

**Cell 2（Code）：**
```python
# 基本运算
import math
print(f"圆周率: {math.pi:.10f}")
print(f"自然对数底: {math.e:.10f}")
print(f"10! = {math.factorial(10)}")
```

**Cell 3（Code）：**
```python
# 列表操作
fruits = ["苹果", "香蕉", "橙子", "葡萄", "西瓜"]
for i, fruit in enumerate(fruits, 1):
    print(f"第{i}种水果: {fruit}")
```

**Cell 4（Code）：**
```python
# 简单可视化
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, np.sin(x), color='blue')
axes[0].set_title('sin(x)')

axes[1].plot(x, np.cos(x), color='red')
axes[1].set_title('cos(x)')

plt.tight_layout()
plt.show()
```

**Cell 5（Code）：**
```python
# 测量性能
%timeit sum(range(100000))
%timeit np.sum(np.arange(100000))
# 对比 Python 原生 sum 和 NumPy sum 的速度差异
```

**Cell 6（Markdown）：**
```markdown
## 小结
- 学会了创建和运行 Cell
- 学会了在 Notebook 里画图
- 发现 NumPy 比原生 Python 快很多（这就是第二阶段要学 NumPy 的原因！）
```

---

## 第零阶段自检

恭喜你完成了整个第零阶段！回顾一下你学会了什么：

- [ ] **终端：** 能用命令行导航、操作文件、使用管道和重定向
- [ ] **Git：** 能创建仓库、提交代码、推送到 GitHub、使用分支
- [ ] **Python 环境：** 能用 Miniconda 创建和管理虚拟环境
- [ ] **VS Code：** 能用 VS Code 写代码、调试、使用快捷键
- [ ] **Jupyter：** 能创建 Notebook、运行代码、画图、写文档

:::tip 全部打勾了？
你已经拥有了一个专业的 AI 开发环境。这些工具会陪伴你走过整个学习旅程。接下来，正式开始学习 Python 编程！
:::
