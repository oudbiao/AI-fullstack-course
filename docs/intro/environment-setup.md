---
sidebar_position: 5
title: "Environment Setup"
description: "Prepare the minimum tools needed to start the AI full-stack course, then verify Python, Git, and a project folder with one tiny run check."
keywords: [AI environment setup, Python environment, VS Code, Git, Miniconda, quick start]
---

# Environment Setup

![Minimal AI course setup kit](/img/course/intro-minimal-setup-kit-en.png)

**Goal:** install only the tools needed for the first week, then prove the machine can run Python and save code with Git.

If setup blocks you for more than 20 minutes, use [Google Colab](https://colab.research.google.com) for now and return later. Environment trouble is a normal engineering task.

## 1. Install First

| Install | Why now |
|---|---|
| Modern browser | Open the course, Colab, GitHub, and AI tools |
| VS Code | Edit code and browse project folders |
| Python 3.11 | Run the early examples |
| Git | Save project checkpoints |
| Miniconda or `venv` | Keep dependencies separate per project |

Wait on GPU drivers, CUDA, Docker, vector databases, and large AI frameworks. Install them when the relevant chapter uses them.

## 2. Five-Minute Check

Check versions:

```bash
python --version
git --version
```

If `python` is not found on macOS or Linux, try:

```bash
python3 --version
```

Create the first project folder:

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -c "print('AI course environment is ready')"
git init
```

Windows PowerShell uses this activation step:

```powershell
mkdir ai-learning-lab
cd ai-learning-lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -c "print('AI course environment is ready')"
git init
```

Expected signal:

```text
AI course environment is ready
Initialized empty Git repository ...
```

## 3. Know These Words

| Term | Meaning |
|---|---|
| Terminal | Place to run commands |
| Interpreter | Program that runs Python |
| Virtual environment | Isolated package room for one project |
| Package | Reusable code installed with `pip` or `conda` |
| Repository | Project folder tracked by Git |
| API key | Private password for online AI services |

## 4. If It Fails

| Symptom | First fix |
|---|---|
| `python` not found | Try `python3` or `py -3.11`; reinstall Python 3.11 |
| `git` not found | Install Git and reopen the terminal |
| `source` fails on Windows | Use the PowerShell command above |
| `pip install` is slow | Use Colab temporarily or a regional mirror |
| Everything feels too much | Continue with Colab and return after Chapter 1 |

Pass line: you can enter a folder, run Python, and initialize Git.
