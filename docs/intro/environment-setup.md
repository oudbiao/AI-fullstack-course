---
sidebar_position: 5
title: "Environment Setup"
description: "Prepare only the minimum tools needed for the first week: browser, Python, Git, and one project folder."
keywords: [AI environment setup, Python environment, VS Code, Git, Miniconda, quick start]
---

# Environment Setup

![Minimal AI course setup kit](/img/course/intro-minimal-setup-kit-en.png)

Install less first. The goal is only: run Python, save code with Git, and keep one project folder.

## Install Now

| Tool | Use |
|---|---|
| Browser | Course, Colab, GitHub, AI tools |
| VS Code | Edit files |
| Python 3.11 | Run examples |
| Git | Save checkpoints |

Install Docker, CUDA, vector databases, and large frameworks later.

## Five-Minute Check

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

Windows PowerShell activation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Expected signal:

```text
AI course environment is ready
Initialized empty Git repository ...
```

If this fails, use Colab for now and return after Chapter 1. The pass line is simple: enter a folder, run Python, initialize Git.
