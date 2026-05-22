---
title: "1 开发者工具基础"
description: "搭建后续 AI 项目所需的最小终端、Git、编辑器、Python 环境和 Notebook 工作流。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "终端, 命令行, Git, VS Code, 开发环境, Python 环境配置"
---
![开发者工具基础主视觉](/img/course/ch01-tools-foundation.webp)

本章只解决一件事：你能不能**创建代码、运行代码、保存代码，并说明别人怎样重新运行**。

## 先看工作台

![开发者工具 AI 工作站漫画指导图](/img/course/ch01-ai-workstation-comic.webp)

先看图。本章的完整闭环就是：

```text
终端 -> 项目文件夹 -> Python 环境 -> 编辑器/Notebook -> Git 历史
```

现在不需要掌握所有工具。先搭一个稳定工作台，后面的 AI 项目都会反复用到它。

## 学习顺序与任务表

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

## 第一个可运行闭环

在练习文件夹里运行下面命令。它会创建一个小项目、运行代码、写说明并提交 Git。

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
. .venv/bin/activate
python -c "import sys; print(sys.executable)"
printf '.venv/\n__pycache__/\n' > .gitignore
printf 'print("AI 学习实验室已准备就绪")\n' > hello_ai.py
printf '# AI 学习实验室\n\n激活环境：. .venv/bin/activate\n运行方式：python hello_ai.py\n' > README.md
python hello_ai.py
git init
git add .gitignore README.md hello_ai.py
git commit -m "init learning lab"
```

预期输出：

```text
AI 学习实验室已准备就绪
```

如果失败，不要直接清空错误。保存命令、完整输出、操作系统、Python 版本和当前目录，这些都是有价值的项目证据。

如果使用 Windows PowerShell，把 `. .venv/bin/activate` 换成 `.venv\Scripts\Activate.ps1`。如果你的系统使用 `python3`，就把命令和 README 里的 `python` 统一换成 `python3`。

### 如何读这个输出

- `AI 学习实验室已准备就绪` 证明脚本确实在项目文件夹里跑起来了。
- `python -c "import sys; print(sys.executable)"` 证明当前到底是哪一个解释器在运行。
- Git commit 证明项目可以被保存、回看和复现。
- 如果某条命令失败，命令和完整错误输出也是证据，不是噪音。

## 深度阶梯

| 层级 | 你能证明什么 |
|---|---|
| 最低通过 | 能创建文件夹、运行脚本，并说清当前目录和 Python 解释器。 |
| 项目可用 | 从新终端能按 README 重跑，`.venv/` 已被忽略，`git status` 只显示有意修改。 |
| 深度检查 | 能解释 PATH、工作目录、shell 和解释器选择为什么会让同一命令在不同机器上表现不同。 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
工作区：终端、Git 仓库、编辑器、Python 环境和 Notebook 都已验证
工件：简短命令日志、提交历史、脚本输出或 notebook 单元结果
调试说明：一个设置问题以及你的诊断方式
失败检查：路径混淆、环境不匹配、Git 状态异常或缺少依赖
期望产出：一套可直接开始学习的工作站证据包
```

## 常见失败

| 现象 | 先检查什么 | 常见修复 |
|---|---|---|
| command not found | 工具是否已安装，PATH 是否生效 | 重新打开终端，或重新安装工具 |
| Python import 失败 | `python` 和 `pip` 是否来自同一个环境 | 用 `python -m pip install ...` 安装 |
| 找不到文件 | 当前目录是否正确 | 运行 `pwd` 和 `ls`，再进入项目目录 |
| Git commit 失败 | 是否初始化、是否暂存、是否配置身份 | 运行 `git status`，必要时配置用户名和邮箱 |
| README 命令跑不通 | README 是否写全每一步 | 从新终端重跑，并更新 README |

## 通关检查

能回答下面五个问题，就可以进入第 2 章：

- 当前终端正在使用哪个目录？
- 运行脚本的是哪个 Python 解释器？
- 上一次 Git 提交后，项目发生了什么变化？
- 从新终端重新运行项目的命令是什么？
- 第一次错误和修复过程记录在哪里？

<details>
<summary>检查思路与讲解</summary>

1. 当前目录用 `pwd` 确认，应该是项目根目录，或命令里明确提到的目标文件夹。
2. Python 解释器用 `which python` 或 `python -c "import sys; print(sys.executable)"` 确认，应该指向课程环境。
3. Git 变化先用 `git status --short` 看概要，再用 `git diff` 解释具体改了什么。
4. 从新终端重跑项目的命令，应该包含激活环境、必要的安装或检查步骤，以及精确脚本命令。
5. 错误记录至少要有现象、命令、可能原因和修复方式。只有截图还不够。

</details>

目标不是把工具学到完美，而是为后面的课程准备一个稳定工作台。
