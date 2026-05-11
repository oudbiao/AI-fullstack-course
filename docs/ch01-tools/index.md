---
title: "1 Developer Tools Fundamentals"
sidebar_position: 0
description: "Build the minimum terminal, Git, editor, Python environment, and notebook workflow needed for later AI projects."
keywords: [Terminal, command line, Git, VS Code, development environment, Python environment setup]
---

# 1 Developer Tools Fundamentals

![Main visual for Developer Tools Fundamentals](/img/course/ch01-tools-foundation-en.webp)

Chapter 1 has one job: make sure you can **create code, run code, save code, and explain how to rerun it**.

## See The Workstation

![AI workstation comic guide for developer tools](/img/course/ch01-ai-workstation-comic-en.webp)

Read the picture first. The whole chapter is this loop:

```text
terminal -> project folder -> Python environment -> editor/notebook -> Git history
```

Do not try to master every tool now. Build one stable workstation, then reuse it in later AI projects.

## Learning Order And Task List

Use this table as both the guide and the task list.

| Page | Follow-along action | Evidence to keep |
|---|---|---|
| [1.1.1 Terminal and command line](ch01-terminal/01-why-cli.md) | Open the terminal and run `pwd`, `ls`, `cd` | A short command log |
| [1.1.2 Basic terminal operations](ch01-terminal/02-basic-operations.md) | Create, move, inspect, and remove files in a practice folder | Folder screenshot or terminal output |
| [1.1.3 Package managers](ch01-terminal/03-package-managers.md) | Check how your system installs tools | Tool version notes |
| [1.2.1 Git basics](ch02-git/01-git-basics.md) and [1.2.2 Git core operations](ch02-git/02-core-operations.md) | Save a first local project snapshot | One clean Git commit |
| [1.3.1 Python environment](ch03-devenv/01-python-env.md) | Create a virtual environment and run Python inside it | Python version and environment command |
| [1.3.2 VS Code](ch03-devenv/02-vscode.md) and [1.3.3 Jupyter](ch03-devenv/03-jupyter.md) | Edit code in an editor and explore in a notebook | Working editor/notebook notes |
| [1.4.1 Follow-along workshop](ch04-workshop/01-hands-on-tools-workshop.md) | Combine terminal, Python, editor, notebook, and Git | Reproducible `ai-learning-lab` README |

The workshop stays at the end because it is the integration step: learn the pieces first, then connect them.

## First Runnable Loop

Run this in a practice folder. It creates a tiny project, runs it, documents it, and commits it.

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
. .venv/bin/activate
python -c "import sys; print(sys.executable)"
printf '.venv/\n__pycache__/\n' > .gitignore
printf 'print("AI learning lab is ready")\n' > hello_ai.py
printf '# AI Learning Lab\n\nActivate env: . .venv/bin/activate\nRun with: python hello_ai.py\n' > README.md
python hello_ai.py
git init
git add .gitignore README.md hello_ai.py
git commit -m "init learning lab"
```

Expected output:

```text
AI learning lab is ready
```

If the command fails, do not erase the error. Save the command, full output, operating system, Python version, and current directory. That record is useful project evidence.

On Windows PowerShell, use `.venv\Scripts\Activate.ps1` instead of `. .venv/bin/activate`. If your system uses `python3`, replace `python` with `python3` consistently in the commands and README.

## Depth Ladder

| Level | What you can prove |
|---|---|
| Minimum pass | You can create a folder, run a script, and identify the current directory and Python interpreter. |
| Project-ready | A fresh terminal can follow your README, `.venv/` is ignored, and `git status` only shows intentional changes. |
| Deeper check | You can explain why PATH, working directory, shell, and interpreter choice change results across machines. |

## Common Failures

| Symptom | First thing to check | Usual fix |
|---|---|---|
| Command not found | Is the tool installed and available in PATH? | Reopen the terminal or reinstall the tool |
| Python import fails | Are `python` and `pip` from the same environment? | Install with `python -m pip install ...` |
| File not found | Are you in the correct directory? | Run `pwd` and `ls`, then move to the project folder |
| Git commit fails | Is Git initialized, staged, and configured? | Run `git status` and set username/email if needed |
| README command fails | Did the README include every required step? | Test from a fresh terminal and update the README |

## Pass Check

Move to Chapter 2 when you can answer these five questions:

- Which directory is the terminal using?
- Which Python interpreter is running your script?
- What changed since the last Git commit?
- What command reruns the project from a fresh terminal?
- Where did you record your first error and fix?

The goal is not tool perfection. The goal is a workstation stable enough for the rest of the course.
