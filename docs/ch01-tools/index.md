---
title: "1 Developer Tools Fundamentals"
sidebar_position: 0
description: "Build the minimum terminal, Git, editor, Python environment, and notebook workflow needed for later AI projects."
keywords: [Terminal, command line, Git, VS Code, development environment, Python environment setup]
---

# 1 Developer Tools Fundamentals

![Main visual for Developer Tools Fundamentals](/img/course/ch01-tools-foundation-en.png)

Chapter 1 has one job: make sure you can **create code, run code, save code, and explain how to rerun it**. If this workstation is unstable, every later AI topic will feel harder than it really is.

## See the workstation first

![AI workstation comic guide for developer tools](/img/course/ch01-ai-workstation-comic-en.png)

Read the picture as one workflow:

```text
terminal -> project folder -> Python environment -> editor/notebook -> Git history
```

You do not need to memorize every command. You need a small repeatable loop that works.

## Stage goal

| Item | Target |
|---|---|
| Suitable for | Beginners, or learners whose development environment is unstable |
| Estimated time | 8-12 hours |
| Minimum output | A runnable `ai-learning-lab` folder with one Python file and one Git commit |
| Portfolio output | README, environment notes, screenshots/logs, and clear Git history |

## Recommended learning order

| Step | Page | What to do |
|---|---|---|
| 1.1 | [1.1.1 Terminal and command line](ch01-terminal/01-why-cli.md) | Open a terminal, move between folders, list files, and run commands |
| 1.2 | [1.1.2 Basic terminal operations](ch01-terminal/02-basic-operations.md) | Create, move, inspect, and remove files inside a practice folder |
| 1.3 | [1.1.3 Package managers](ch01-terminal/03-package-managers.md) | Understand how tools are installed and checked |
| 1.4 | [1.2.1 Git basics](ch02-git/01-git-basics.md) | Save a first project snapshot with Git |
| 1.5 | [1.2.2 Git core operations](ch02-git/02-core-operations.md) | Use `status`, `add`, `commit`, `log`, and `diff` |
| 1.6 | [1.3.1 Python environment](ch03-devenv/01-python-env.md) | Create an isolated Python environment and install packages correctly |
| 1.7 | [1.3.2 VS Code](ch03-devenv/02-vscode.md) and [1.3.3 Jupyter](ch03-devenv/03-jupyter.md) | Use an editor for projects and a notebook for exploration |
| 1.8 | [1.4.1 Follow-along workshop](ch04-workshop/01-hands-on-tools-workshop.md) | Connect everything into one reproducible mini-project |

The workshop stays at the end because it is the integration step. First learn the pieces, then use them together.

## Tasks you must complete in this stage

| Task | Deliverable | Completion check |
|---|---|---|
| Use terminal safely | Command practice log | You can explain where `pwd`, `ls`, and `cd` are operating |
| Run Python | `hello_ai.py` | `python hello_ai.py` prints the expected output |
| Isolate an environment | `.venv` or Conda environment note | You know which Python interpreter is active |
| Use an editor | Project opened in VS Code or equivalent | You can edit, run, and inspect terminal output |
| Save with Git | At least one local commit | `git status` is clean after the commit |
| Finish the workshop | `ai-learning-lab` with README and logs | Another person can follow your README and rerun it |

## Minimum runnable experiment

Run this in a practice folder. It creates a tiny project, runs it, documents it, and commits it.

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

Expected output:

```text
AI learning lab is ready
```

If the command fails, do not erase the error. Save the command, full output, operating system, Python version, and current directory. That record is useful project evidence.

## Common failures

| Symptom | First thing to check | Usual fix |
|---|---|---|
| Command not found | Is the tool installed and available in PATH? | Reopen the terminal or reinstall the tool |
| Python import fails | Are `python` and `pip` from the same environment? | Install with `python -m pip install ...` |
| File not found | Are you in the correct directory? | Run `pwd` and `ls`, then move to the project folder |
| Git commit fails | Is Git initialized, staged, and configured? | Run `git status` and set username/email if needed |
| README command fails | Did the README include every required step? | Test from a fresh terminal and update the README |

## Stage deliverables

| Deliverable | Minimum version | Portfolio version |
|---|---|---|
| Learning repository | `ai-learning-lab` exists and runs one Python file | Clear folders, README, screenshots/logs, and commit history |
| Environment notes | Python version and install commands are recorded | Virtual environment steps and dependency files are included |
| Command log | 5-10 useful commands are saved | Each command includes purpose, output, and failure handling |
| Git record | One local commit exists | Commit messages show small, meaningful progress |
| README | Explains how to run `hello_ai.py` | Explains goal, setup, run command, sample output, and next step |

## Stage completion criteria

| Level | You can move on when... |
|---|---|
| Minimum pass | You can open the terminal, run Python, and make one Git commit |
| Recommended pass | You can create a virtual environment, install a dependency, and write a clear README |
| Portfolio pass | You can complete the workshop and leave enough evidence for another person to rerun it |

## Stage completion questions

- Which directory is the terminal currently using?
- Which Python interpreter is running your script?
- What changed since the last Git commit?
- What command reruns your project from a fresh terminal?
- Where did you record the first error you met and how you fixed it?

After this chapter, continue to Chapter 2. The goal is not tool perfection; it is a stable workstation for the rest of the course.
