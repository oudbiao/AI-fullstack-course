---
sidebar_position: 1
title: "0.2 环境准备"
description: "准备第一周需要的最小可复现 AI 工程环境：浏览器、Python、Git 和一个项目文件夹。"
keywords: [AI 工程环境, AI 环境准备, Python 环境, VS Code, Git, 快速开始]
---

# 0.2 环境准备

![AI 课程最小环境工具包](/img/course/intro-minimal-setup-kit.webp)

先少装。目标只有一个：进入一个文件夹，能运行 Python，用 Git 保存代码，并留下足够证据，让别人也能复现你的工作。

在面向工作的 AI 项目里，环境准备不是杂事。它是第一份证明：你的项目可以从自己的电脑，迁移到审阅者、队友或部署环境里。

## 现在只装这些

| 工具 | 用途 |
|---|---|
| 浏览器 | 打开课程、Colab、GitHub、AI 工具 |
| VS Code | 编辑文件 |
| Python 3.11 | 运行示例 |
| Git | 保存检查点 |

Docker、CUDA、向量数据库和大型框架以后再装。太早装太多东西，会让新手更难定位错误。

## 选择一个 Python 命令

不同机器启动 Python 的命令可能不同。先找到第一个能用的命令，然后在笔记里固定使用它。

| 系统 | 先试 | 不行再试 |
|---|---|---|
| macOS / Linux | `python3 --version` | `python --version` |
| Windows PowerShell | `py -3.11 --version` | `python --version` |
| Colab | 不需要本地安装 | 使用 Notebook runtime |

后面课程如果写 `python`，你可以替换成自己机器上能用的命令。

## 五分钟验证

```bash
python3 --version
git --version
mkdir ai-learning-lab
cd ai-learning-lab
python3 -m venv .venv
source .venv/bin/activate
python -c "print('AI course environment is ready')"
git init
git status
```

Windows PowerShell 激活虚拟环境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

看到类似输出即可：

```text
AI course environment is ready
Initialized empty Git repository ...
```

`git status` 应该能说明你已经在一个仓库里。现在不需要提交代码，重点是确认这个文件夹已经可以追踪工作。

## 如果验证失败

| 现象 | 先做什么 | 留下什么证据 |
|---|---|---|
| 找不到 `python3` | 试上面的命令表，再安装 Python 3.11 | 命令和完整报错 |
| 虚拟环境激活失败 | 检查 shell：zsh/bash 用 `source`，PowerShell 用 `Activate.ps1` | shell 名称和激活命令 |
| 找不到 `git` | 安装 Git，重开终端，再试 `git --version` | 版本输出或报错 |
| 权限错误 | 把项目放在用户目录，不要放系统保护目录 | `pwd` 显示的当前目录 |

如果仍然失败，先用 Colab 继续学，学完第 1 章再回来。达标线很简单：进入文件夹、运行 Python、初始化 Git。

## 有经验的人要检查什么

如果你已经有环境，也不要完全跳过。确认你能解释：

- 这个课程项目到底用哪个解释器运行。
- 依赖会安装到哪里。
- 怎样在另一台机器上重建环境。
- 哪些文件应该提交，哪些应该留在本地。

环境也是课程产出的一部分。一个项目如果只能在你电脑上运行，还不算完成。更强的项目会有简短环境说明、明确 Python 版本、依赖方案，以及一个证明基础运行时可用的命令。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
机器状态：OS、Python/Node 版本、编辑器、终端和包管理器
验证记录：运行的命令、打印的版本和第一段脚本输出
调试说明：安装错误、路径问题、权限问题或环境不匹配
恢复计划：在继续前重试的精确命令或文档页面
期望产出：一个可复现项目文件夹、成功命令输出和一个已知回退方案
```
