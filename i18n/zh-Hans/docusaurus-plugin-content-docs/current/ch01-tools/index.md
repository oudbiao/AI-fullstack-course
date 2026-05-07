---
title: "1 开发者工具基础"
sidebar_position: 0
description: "搭建后续 AI 项目所需的最小终端、Git、编辑器、Python 环境和 Notebook 工作流。"
keywords: [终端, 命令行, Git, VS Code, 开发环境, Python 环境配置]
---

# 1 开发者工具基础

![开发者工具基础主视觉](/img/course/ch01-tools-foundation.png)

本章只解决一件事：你能不能**创建代码、运行代码、保存代码，并说明别人怎样重新运行**。如果这个工作台不稳定，后面每个 AI 主题都会被环境问题放大。

## 先看工作台

![开发者工具 AI 工作站漫画指导图](/img/course/ch01-ai-workstation-comic.png)

把图当成一个工作流来读：

```text
终端 -> 项目文件夹 -> Python 环境 -> 编辑器/Notebook -> Git 历史
```

不用背下所有命令，先把一个小闭环跑稳。

## 阶段目标

| 项目 | 目标 |
|---|---|
| 适合对象 | 刚入门，或开发环境经常不稳定的学习者 |
| 预估学时 | 8-12 小时 |
| 最小产出 | 一个能运行的 `ai-learning-lab` 文件夹、一个 Python 文件、一次 Git 提交 |
| 作品集产出 | README、环境说明、截图/日志、清晰的 Git 历史 |

## 推荐学习顺序

| 步骤 | 页面 | 要做什么 |
|---|---|---|
| 1.1 | [1.1.1 终端与命令行](ch01-terminal/01-why-cli.md) | 打开终端、切换目录、查看文件、运行命令 |
| 1.2 | [1.1.2 基础终端操作](ch01-terminal/02-basic-operations.md) | 在练习文件夹里创建、移动、查看和删除文件 |
| 1.3 | [1.1.3 包管理器](ch01-terminal/03-package-managers.md) | 理解工具怎样安装、怎样检查 |
| 1.4 | [1.2.1 Git 基础](ch02-git/01-git-basics.md) | 用 Git 保存第一次项目快照 |
| 1.5 | [1.2.2 Git 核心操作](ch02-git/02-core-operations.md) | 使用 `status`、`add`、`commit`、`log`、`diff` |
| 1.6 | [1.3.1 Python 环境](ch03-devenv/01-python-env.md) | 创建隔离环境，正确安装依赖 |
| 1.7 | [1.3.2 VS Code](ch03-devenv/02-vscode.md) 与 [1.3.3 Jupyter](ch03-devenv/03-jupyter.md) | 用编辑器写项目，用 Notebook 做探索 |
| 1.8 | [1.4.1 跟做工作坊](ch04-workshop/01-hands-on-tools-workshop.md) | 把所有工具串成一个可复现小项目 |

工作坊放在最后，因为它是综合实操。先学零件，再把零件装起来。

## 本阶段必须完成的任务

| 任务 | 产出物 | 完成检查 |
|---|---|---|
| 安全使用终端 | 命令练习记录 | 能说明 `pwd`、`ls`、`cd` 正在操作哪里 |
| 运行 Python | `hello_ai.py` | `python hello_ai.py` 输出预期内容 |
| 隔离环境 | `.venv` 或 Conda 环境说明 | 知道当前正在用哪个 Python |
| 使用编辑器 | 用 VS Code 或等价工具打开项目 | 能编辑、运行并查看终端输出 |
| 用 Git 保存 | 至少一次本地 commit | commit 后 `git status` 是干净的 |
| 完成工作坊 | 带 README 和日志的 `ai-learning-lab` | 其他人能跟着 README 重新运行 |

## 最小可运行实验

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

## 常见失败

| 现象 | 先检查什么 | 常见修复 |
|---|---|---|
| command not found | 工具是否已安装，PATH 是否生效 | 重新打开终端，或重新安装工具 |
| Python import 失败 | `python` 和 `pip` 是否来自同一个环境 | 用 `python -m pip install ...` 安装 |
| 找不到文件 | 当前目录是否正确 | 运行 `pwd` 和 `ls`，再进入项目目录 |
| Git commit 失败 | 是否初始化、是否暂存、是否配置身份 | 运行 `git status`，必要时配置用户名和邮箱 |
| README 命令跑不通 | README 是否写全每一步 | 从新终端重跑，并更新 README |

## 阶段交付物

| 交付物 | 最小版本 | 作品集版本 |
|---|---|---|
| 学习仓库 | `ai-learning-lab` 存在，并能运行一个 Python 文件 | 目录清晰，有 README、截图/日志和提交历史 |
| 环境说明 | 记录 Python 版本和安装命令 | 包含虚拟环境步骤和依赖文件 |
| 命令日志 | 保存 5-10 条常用命令 | 每条命令有用途、输出和失败处理 |
| Git 记录 | 至少一次本地提交 | commit message 能体现小步推进 |
| README | 说明如何运行 `hello_ai.py` | 说明目标、环境、运行命令、示例输出和下一步 |

## 阶段通关标准

| 等级 | 满足这个条件就可以继续 |
|---|---|
| 最小通关 | 能打开终端、运行 Python，并完成一次 Git 提交 |
| 推荐通关 | 能创建虚拟环境、安装依赖，并写清楚 README |
| 作品集通关 | 能完成工作坊，并留下别人可复现的证据 |

## 阶段通关问题

- 当前终端正在使用哪个目录？
- 运行脚本的是哪个 Python 解释器？
- 上一次 Git 提交后，项目发生了什么变化？
- 从新终端重新运行项目的命令是什么？
- 你把第一次错误和修复过程记录在哪里？

完成本章后继续1.2 节。目标不是把工具学到完美，而是为后面的课程准备一个稳定工作台。
