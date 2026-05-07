---
sidebar_position: 5
title: "Environment Setup"
description: "Prepare the minimum tools needed to start the AI full-stack course, then verify that Python, Git, and a project folder work."
keywords: [AI environment setup, Python environment, VS Code, Git, Miniconda, quick start]
---

# Environment Setup

![Minimal AI course setup kit](/img/course/intro-minimal-setup-kit-en.png)

**Goal:** prepare only what you need for the first chapters, then prove that your computer can run a tiny project.

If your local setup gets stuck, keep learning with [Google Colab](https://colab.research.google.com) and come back later. Environment problems are normal; they are not a sign that you are bad at AI.

## Install Only These First

| Tool | What it is | Why you need it now |
| --- | --- | --- |
| A modern browser | Chrome, Edge, Safari, or Firefox | Open the course, Colab, GitHub, and AI tools |
| VS Code | Code editor | Edit files and browse projects |
| Python 3.11 | Programming language | Run course examples |
| Git | Version control | Save checkpoints and upload projects later |
| Miniconda | Python environment manager | Keep project dependencies separate |

You do **not** need to install GPU drivers, CUDA, Docker, vector databases, or every AI framework before Chapter 1. Those tools appear later when the course actually uses them.

## Five-Minute Check

After installing the basics, open a terminal and check:

```bash
python --version
git --version
```

On some macOS or Linux machines, use `python3 --version` if `python` is not found.

Then create a tiny project:

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell uses a different activation command:

```powershell
mkdir ai-learning-lab
cd ai-learning-lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Create a quick run check:

```bash
python -c "print('AI course environment is ready')"
git init
```

Expected output:

```text
AI course environment is ready
Initialized empty Git repository ...
```

If you see this, you are ready to enter Chapter 1.

## Concepts You Only Need to Recognize

| Term | Simple meaning |
| --- | --- |
| Terminal | A place to run commands |
| Editor | A place to write and organize code |
| Python interpreter | The program that runs Python code |
| Virtual environment | A small isolated room for one project's packages |
| Package | Reusable code installed with `pip` or `conda` |
| Repository | A project folder tracked by Git |
| API key | A private password for calling an online AI service |
| GPU | Hardware that speeds up deep learning; useful later, not required now |

## Install Later, Not Now

| Later tool | When it becomes useful |
| --- | --- |
| Jupyter Notebook | Chapter 3 data analysis |
| PyTorch | Chapter 6 deep learning |
| Hugging Face `transformers` | Chapters 7 and 11 |
| OpenAI-compatible SDKs | Chapter 8 LLM applications |
| Docker | Chapter 8 deployment and reproducibility |
| Vector database | Chapter 8 RAG |
| GPU or cloud GPU | Chapter 6+ model experiments |

Install tools when a chapter asks for them. This keeps the first week light and avoids dependency conflicts.

## If Something Fails

| Symptom | Check first |
| --- | --- |
| `python` is not found | Try `python3` or `py -3.11`; confirm Python 3.11 is installed |
| `git` is not found | Install Git, then reopen the terminal |
| `source` does not work on Windows | Use the PowerShell activation command shown above |
| `pip install` is very slow | Try a regional PyPI mirror or use Colab temporarily |
| The local setup feels overwhelming | Continue with Colab and return after Chapter 1 |

For now, the pass line is simple: you can open a terminal, enter a project folder, run Python, and initialize Git.
