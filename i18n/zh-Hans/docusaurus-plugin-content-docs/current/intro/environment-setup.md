---
sidebar_position: 5
title: "环境准备"
description: "准备 AI 全栈课程开头真正需要的最小工具，并验证 Python、Git 和项目文件夹可以正常工作。"
keywords: [AI 环境配置, Python 环境, VS Code, Git, Miniconda, 快速开始]
---

# 环境准备

![AI 课程最小准备清单](/img/course/intro-minimal-setup-kit.png)

**目标：** 只准备前几章真正需要的工具，然后确认你的电脑能跑一个最小项目。

如果本地环境暂时卡住，可以先用 [Google Colab](https://colab.research.google.com) 继续学习，后面再回来修。本地环境出问题很常见，不代表你不适合学 AI。

## 先只安装这些

| 工具 | 它是什么 | 现在为什么需要 |
| --- | --- | --- |
| 现代浏览器 | Chrome、Edge、Safari 或 Firefox | 打开课程、Colab、GitHub 和 AI 工具 |
| VS Code | 代码编辑器 | 写代码、看项目文件 |
| Python 3.11 | 编程语言 | 运行课程示例 |
| Git | 版本管理工具 | 保存学习检查点，后面上传项目 |
| Miniconda | Python 环境管理工具 | 让不同项目的依赖互不干扰 |

第 1 章之前，你**不需要**提前安装 GPU 驱动、CUDA、Docker、向量数据库或所有 AI 框架。课程用到时再装，压力会小很多。

## 5 分钟验证

安装基础工具后，打开终端检查：

```bash
python --version
git --version
```

有些 macOS 或 Linux 机器需要用 `python3 --version`，如果 `python` 找不到就试这个。

然后创建一个最小项目：

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell 的激活命令不同：

```powershell
mkdir ai-learning-lab
cd ai-learning-lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

再做一次运行检查：

```bash
python -c "print('AI course environment is ready')"
git init
```

预期输出大致是：

```text
AI course environment is ready
Initialized empty Git repository ...
```

看到这个，就可以进入第 1 章了。

## 先认识这些概念即可

| 术语 | 简单理解 |
| --- | --- |
| 终端 | 输入和运行命令的地方 |
| 编辑器 | 写代码、整理文件的地方 |
| Python 解释器 | 真正执行 Python 代码的程序 |
| 虚拟环境 | 给某个项目单独准备的小房间，里面放这个项目的包 |
| 包 | 用 `pip` 或 `conda` 安装的可复用代码 |
| 仓库 | 被 Git 记录版本的项目文件夹 |
| API key | 调用在线 AI 服务时使用的私人密码 |
| GPU | 加速深度学习的硬件；后面有用，现在不必准备 |

## 这些以后再装

| 后续工具 | 什么时候有用 |
| --- | --- |
| Jupyter Notebook | 第 3 章数据分析 |
| PyTorch | 第 6 章深度学习 |
| Hugging Face `transformers` | 第 7 章和第 11 章 |
| OpenAI 兼容 SDK | 第 8 章 LLM 应用 |
| Docker | 第 8 章部署与可复现 |
| 向量数据库 | 第 8 章 RAG |
| GPU 或云 GPU | 第 6 章之后的模型实验 |

按章节要求逐步安装，第一周会轻很多，也更不容易出现依赖冲突。

## 如果遇到问题

| 现象 | 先检查什么 |
| --- | --- |
| 找不到 `python` | 试试 `python3` 或 `py -3.11`，并确认已安装 Python 3.11 |
| 找不到 `git` | 安装 Git，然后重新打开终端 |
| Windows 里 `source` 不能用 | 使用上面的 PowerShell 激活命令 |
| `pip install` 很慢 | 可以换国内 PyPI 镜像，或先用 Colab 继续 |
| 本地环境让你很烦 | 先用 Colab 学，第 1 章后再回来整理 |

现在的通关线很简单：你能打开终端、进入项目文件夹、运行 Python，并初始化 Git。
