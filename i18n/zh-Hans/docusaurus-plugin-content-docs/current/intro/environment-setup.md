---
sidebar_position: 5
title: "环境准备"
description: "准备 AI 全栈课程第一周所需的最小工具，并用一个小命令验证 Python、Git 和项目文件夹能正常工作。"
keywords: [AI 环境准备, Python 环境, VS Code, Git, Miniconda, 快速开始]
---

# 环境准备

![AI 课程最小环境工具包](/img/course/intro-minimal-setup-kit.png)

**目标：**只安装第一周需要的工具，然后证明电脑能运行 Python，并能用 Git 保存代码。

如果环境问题卡住超过 20 分钟，先用 [Google Colab](https://colab.research.google.com) 继续学，之后再回来修。本地环境报错是正常工程任务，不是你不适合学 AI。

## 1. 先安装这些

| 安装 | 为什么现在需要 |
|---|---|
| 现代浏览器 | 打开课程、Colab、GitHub 和 AI 工具 |
| VS Code | 编辑代码、浏览项目文件夹 |
| Python 3.11 | 运行前期示例 |
| Git | 保存项目检查点 |
| Miniconda 或 `venv` | 让每个项目的依赖分开 |

GPU 驱动、CUDA、Docker、向量数据库和大型 AI 框架先不要急着装。等章节真正用到时再装。

## 2. 五分钟验证

检查版本：

```bash
python --version
git --version
```

如果 macOS 或 Linux 上找不到 `python`，试试：

```bash
python3 --version
```

创建第一个项目文件夹：

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -c "print('AI course environment is ready')"
git init
```

Windows PowerShell 使用这组命令：

```powershell
mkdir ai-learning-lab
cd ai-learning-lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -c "print('AI course environment is ready')"
git init
```

看到类似输出即可：

```text
AI course environment is ready
Initialized empty Git repository ...
```

## 3. 先认识这些词

| 术语 | 意思 |
|---|---|
| 终端 | 输入命令的地方 |
| 解释器 | 运行 Python 的程序 |
| 虚拟环境 | 一个项目专用的依赖隔离空间 |
| 包 | 用 `pip` 或 `conda` 安装的可复用代码 |
| 仓库 | 被 Git 跟踪的项目文件夹 |
| API Key | 调用在线 AI 服务的私密密码 |

## 4. 如果失败

| 现象 | 先这样处理 |
|---|---|
| 找不到 `python` | 试 `python3` 或 `py -3.11`；重新安装 Python 3.11 |
| 找不到 `git` | 安装 Git 后重新打开终端 |
| Windows 上 `source` 失败 | 使用上面的 PowerShell 命令 |
| `pip install` 很慢 | 暂时用 Colab，或换区域镜像 |
| 感觉太复杂 | 先用 Colab 继续，学完第 1 章再回来 |

达标线很简单：能进入一个文件夹、运行 Python、初始化 Git。
