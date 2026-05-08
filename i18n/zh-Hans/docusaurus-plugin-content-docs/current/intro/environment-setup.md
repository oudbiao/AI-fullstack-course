---
sidebar_position: 1
title: "0.2 环境准备"
description: "只准备第一周需要的最小工具：浏览器、Python、Git 和一个项目文件夹。"
keywords: [AI 环境准备, Python 环境, VS Code, Git, Miniconda, 快速开始]
---

# 0.2 环境准备

![AI 课程最小环境工具包](/img/course/intro-minimal-setup-kit.webp)

先少装。目标只有一个：能运行 Python、用 Git 保存代码、保留一个项目文件夹。

## 现在只装这些

| 工具 | 用途 |
|---|---|
| 浏览器 | 打开课程、Colab、GitHub、AI 工具 |
| VS Code | 编辑文件 |
| Python 3.11 | 运行示例 |
| Git | 保存检查点 |

Docker、CUDA、向量数据库和大型框架以后再装。

## 五分钟验证

```bash
python --version
git --version
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate
python -c "print('AI course environment is ready')"
git init
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

如果失败，先用 Colab 继续学，学完第 1 章再回来。达标线很简单：进入文件夹、运行 Python、初始化 Git。
