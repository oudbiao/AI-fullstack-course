---
title: "1 开发者工具基础"
sidebar_position: 0
description: "搭建后续 AI 项目所需的最小终端、Git、编辑器、Python 环境和 Notebook 工作流。"
keywords: [终端, 命令行, Git, VS Code, 开发环境, Python 环境配置]
---

# 1 开发者工具基础

![开发者工具基础主视觉](/img/course/ch01-tools-foundation.png)

本章只解决一件事：你能不能**创建代码、运行代码、保存代码，并说明别人怎样重新运行**。

## 1.0.1 先看工作台

![开发者工具 AI 工作站漫画指导图](/img/course/ch01-ai-workstation-comic.png)

先看图。本章的完整闭环就是：

```text
终端 -> 项目文件夹 -> Python 环境 -> 编辑器/Notebook -> Git 历史
```

现在不需要掌握所有工具。先搭一个稳定工作台，后面的 AI 项目都会反复用到它。

## 1.0.2 学习顺序与任务表

下面这一张表同时作为学习指南和任务清单。

| 页面 | 跟着做 | 留下的证据 |
|---|---|---|
| [1.1.1 终端与命令行](ch01-terminal/01-why-cli.md) | 打开终端，运行 `pwd`、`ls`、`cd` | 一小段命令记录 |
| [1.1.2 基础终端操作](ch01-terminal/02-basic-operations.md) | 在练习文件夹里创建、移动、查看、删除文件 | 文件夹截图或终端输出 |
| [1.1.3 包管理器](ch01-terminal/03-package-managers.md) | 检查你的系统怎样安装工具 | 工具版本记录 |
| [1.2.1 Git 基础](ch02-git/01-git-basics.md) 和 [1.2.2 Git 核心操作](ch02-git/02-core-operations.md) | 保存第一次本地项目快照 | 一次干净的 Git 提交 |
| [1.3.1 Python 环境](ch03-devenv/01-python-env.md) | 创建虚拟环境，并在其中运行 Python | Python 版本和环境命令 |
| [1.3.2 VS Code](ch03-devenv/02-vscode.md) 和 [1.3.3 Jupyter](ch03-devenv/03-jupyter.md) | 用编辑器写代码，用 Notebook 做探索 | 编辑器/Notebook 可用记录 |
| [1.4.1 跟做工作坊](ch04-workshop/01-hands-on-tools-workshop.md) | 串起终端、Python、编辑器、Notebook 和 Git | 可复现的 `ai-learning-lab` README |

工作坊放在最后，因为它是综合实操：先学零件，再把零件装起来。

## 1.0.3 第一个可运行闭环

在练习文件夹里运行下面命令。它会创建一个小项目、运行代码、写说明并提交 Git。

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
printf 'print("AI learning lab is ready")\n' > hello_ai.py
printf '# AI Learning Lab\n\nRun with: python hello_ai.py\n' > README.md
python hello_ai.py
git init
git add README.md hello_ai.py
git commit -m "init learning lab"
```

预期输出：

```text
AI learning lab is ready
```

如果失败，不要直接清空错误。保存命令、完整输出、操作系统、Python 版本和当前目录，这些都是有价值的项目证据。

## 1.0.4 常见失败

| 现象 | 先检查什么 | 常见修复 |
|---|---|---|
| command not found | 工具是否已安装，PATH 是否生效 | 重新打开终端，或重新安装工具 |
| Python import 失败 | `python` 和 `pip` 是否来自同一个环境 | 用 `python -m pip install ...` 安装 |
| 找不到文件 | 当前目录是否正确 | 运行 `pwd` 和 `ls`，再进入项目目录 |
| Git commit 失败 | 是否初始化、是否暂存、是否配置身份 | 运行 `git status`，必要时配置用户名和邮箱 |
| README 命令跑不通 | README 是否写全每一步 | 从新终端重跑，并更新 README |

## 1.0.5 通关检查

能回答下面五个问题，就可以进入第 2 章：

- 当前终端正在使用哪个目录？
- 运行脚本的是哪个 Python 解释器？
- 上一次 Git 提交后，项目发生了什么变化？
- 从新终端重新运行项目的命令是什么？
- 第一次错误和修复过程记录在哪里？

目标不是把工具学到完美，而是为后面的课程准备一个稳定工作台。
