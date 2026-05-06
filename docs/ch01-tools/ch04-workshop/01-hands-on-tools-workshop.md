---
title: "1.4 Follow-Along Workshop: Build a Reproducible AI Learning Lab"
sidebar_position: 1
description: "A step-by-step Chapter 1 workshop that connects the terminal, Python environment checks, VS Code, Jupyter, Git, and portfolio evidence into one runnable learning repository."
keywords: [developer tools workshop, terminal, Git, Python environment, VS Code, Jupyter, reproducible project]
---

# Follow-Along Workshop: Build a Reproducible AI Learning Lab

![Chapter 1 hands-on workstation route](/img/course/ch01-hands-on-workstation-route-en.png)

:::tip Workshop goal
This page is the practical bridge for Chapter 1. You will build a small repository named `ai-learning-lab`, run a Python environment check, save reports, make Git commits, and leave portfolio evidence that proves your workstation is usable.
:::

## What you will build

You will create a local project that answers one simple but important question: "Can I create, run, inspect, save, and explain a project on this computer?"

The final project will contain:

| File or folder | Purpose |
|---|---|
| `README.md` | Explains the project goal, run command, and expected output |
| `src/workstation_check.py` | Runnable Python script that checks the current toolchain |
| `notes/learning-log.md` | Your daily command and troubleshooting notes |
| `reports/workstation-check.json` | Machine-readable environment report |
| `reports/workstation-report.md` | Human-readable portfolio evidence |
| `.gitignore` | Prevents cache, secrets, and local environments from being committed |

This workshop uses only the Python standard library. You do not need a third-party SDK, a cloud account, or a paid service.

## Step 0: Create a clean practice folder

Open a terminal and run:

```bash
mkdir ai-learning-lab
cd ai-learning-lab
pwd
python3 --version
```

Expected output will be similar to this:

```text
/Users/zhangsan/ai-learning-lab
Python 3.12.3
```

The exact path and Python version can be different. For this workshop, Python 3.10 or newer is enough.

:::info Windows note
In PowerShell, use `python --version` if `python3 --version` is not recognized. Keep using the same command consistently in the rest of the workshop.
:::

## Step 1: Understand the complete route first

![Terminal, Python, and Git execution loop](/img/course/ch01-hands-on-terminal-git-loop-en.png)

Do not treat the tools as separate topics. In real work, they form one loop:

| Step | Tool | What you do |
|---|---|---|
| 1 | Terminal | Move into the project folder and run commands |
| 2 | Python | Run the check script and generate evidence |
| 3 | Editor | Read and improve the files |
| 4 | Git | Save the stable state as a commit |
| 5 | Report | Keep output that proves the project can be rerun |

When something fails later, you will usually debug one of these links: current folder, Python interpreter, dependency location, file path, or Git state.

## Step 2: Create the project skeleton

Run these commands inside `ai-learning-lab`:

```bash
mkdir -p src notes reports notebooks screenshots
touch requirements.txt
```

Create a `.gitignore` file:

```bash
cat > .gitignore << 'EOF'
__pycache__/
*.pyc
.venv/
venv/
.env
.env.local
.ipynb_checkpoints/
.DS_Store
EOF
```

Create the first README:

````bash
cat > README.md << 'EOF'
# AI Learning Lab

This repository is my reproducible learning workspace for the AI full-stack course.

## Run

```bash
python3 src/workstation_check.py
```

## Expected output

The script prints the current project root, Python executable, Git branch, and report file paths.
EOF
````

Create the first learning log:

```bash
cat > notes/learning-log.md << 'EOF'
# Learning Log

| Time | Command or action | Result | Note |
|---|---|---|---|
EOF
```

Expected structure:

```text
ai-learning-lab/
  README.md
  requirements.txt
  src/
  notes/
  reports/
  notebooks/
  screenshots/
```

If you installed `tree`, you can check it with `tree -a -L 2`. If not, `find . -maxdepth 2 -type f` is enough.

## Step 3: Add the runnable environment check script

Create `src/workstation_check.py` and paste the complete code below.

```python
from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "reports"
NOTES_DIR = PROJECT_ROOT / "notes"
JSON_REPORT = REPORTS_DIR / "workstation-check.json"
MARKDOWN_REPORT = REPORTS_DIR / "workstation-report.md"
LEARNING_LOG = NOTES_DIR / "learning-log.md"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def run_command(command: list[str]) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        return {
            "command": " ".join(command),
            "returncode": 127,
            "stdout": "",
            "stderr": f"{command[0]} was not found",
        }
    return {
        "command": " ".join(command),
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def visible_project_files() -> list[str]:
    files: list[str] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if ".git" in path.parts or path.is_dir():
            continue
        files.append(str(path.relative_to(PROJECT_ROOT)))
    return files


def ensure_workspace_files() -> None:
    REPORTS_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)
    if not LEARNING_LOG.exists():
        LEARNING_LOG.write_text(
            "# Learning Log\n\n| Time | Command or action | Result | Note |\n|---|---|---|---|\n",
            encoding="utf-8",
        )


def build_report() -> dict[str, Any]:
    return {
        "generated_at": utc_now(),
        "project_root": str(PROJECT_ROOT),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "git_branch": run_command(["git", "branch", "--show-current"]),
        "git_status": run_command(["git", "status", "--short"]),
        "project_files": visible_project_files(),
    }


def write_reports(report: dict[str, Any]) -> None:
    JSON_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    branch = report["git_branch"]["stdout"] or "(no branch yet)"
    status = report["git_status"]["stdout"] or "working tree clean"
    lines = [
        "# Workstation Report",
        "",
        f"- Generated at: {report['generated_at']}",
        f"- Project root: `{report['project_root']}`",
        f"- Python version: `{report['python_version']}`",
        f"- Python executable: `{report['python_executable']}`",
        f"- Git branch: `{branch}`",
        "",
        "## Git status",
        "",
        "```text",
        status,
        "```",
        "",
        "## Project files",
        "",
    ]
    lines.extend(f"- `{file}`" for file in report["project_files"])
    MARKDOWN_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def append_learning_log(report: dict[str, Any]) -> None:
    branch = report["git_branch"]["stdout"] or "no branch"
    LEARNING_LOG.write_text(
        LEARNING_LOG.read_text(encoding="utf-8")
        + f"| {report['generated_at']} | python3 src/workstation_check.py | ok | branch: {branch} |\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_workspace_files()
    report = build_report()
    write_reports(report)
    append_learning_log(report)

    branch = report["git_branch"]["stdout"] or "(no branch yet)"
    print(f"[ok] project root: {PROJECT_ROOT}")
    print(f"[ok] python: {report['python_version']} at {report['python_executable']}")
    print(f"[ok] git branch: {branch}")
    print(f"[ok] wrote {JSON_REPORT.relative_to(PROJECT_ROOT)}")
    print(f"[ok] wrote {MARKDOWN_REPORT.relative_to(PROJECT_ROOT)}")
    print("[next] run git status, then commit the files when the output looks right")


if __name__ == "__main__":
    main()
```

Important beginner detail: the script uses `Path(__file__).resolve().parents[1]` to find the project root. That means it still works even if you run it from the project folder with `python3 src/workstation_check.py`.

## Step 4: Initialize Git and run the script

Run the commands below:

```bash
git init
git branch -M main
git config user.name "AI Learner"
git config user.email "learner@example.com"
python3 src/workstation_check.py
```

Expected output:

```text
[ok] project root: /Users/zhangsan/ai-learning-lab
[ok] python: 3.12.3 at /usr/local/bin/python3
[ok] git branch: main
[ok] wrote reports/workstation-check.json
[ok] wrote reports/workstation-report.md
[next] run git status, then commit the files when the output looks right
```

Your Python path can be different. What matters is that the script runs, and both report files are created.

Check the generated evidence:

```bash
cat reports/workstation-report.md
git status --short
```

Expected `git status --short` output will look similar to this:

```text
?? .gitignore
?? README.md
?? notes/
?? reports/
?? requirements.txt
?? src/
```

`??` means Git can see the files, but it has not started tracking them yet.

## Step 5: Make your first clean commit

```bash
git add .gitignore README.md requirements.txt src notes reports
git status --short
git commit -m "Initialize AI learning lab workstation"
git log --oneline
```

Expected output:

```text
abc1234 Initialize AI learning lab workstation
```

You have now created the first stable checkpoint of your workstation.

## Step 6: Practice a branch without breaking main

Create a small branch, add one learning note, rerun the script, then merge it back.

```bash
git checkout -b practice/add-daily-note
printf "\n- Practiced terminal, Python, and Git together.\n" >> notes/learning-log.md
python3 src/workstation_check.py
git diff -- notes/learning-log.md
git add notes/learning-log.md reports/workstation-check.json reports/workstation-report.md
git commit -m "Add daily tool practice note"
git checkout main
git merge practice/add-daily-note
git log --oneline --graph --all
```

PowerShell users can replace the `printf` line with:

```powershell
Add-Content notes/learning-log.md "- Practiced terminal, Python, and Git together."
```

This exercise is small on purpose. The goal is not to learn a complex branching model yet; it is to feel that `main` is the stable line, and a practice branch is a safe place to try changes.

## Step 7: Use VS Code and Jupyter as two work panels

![Environment, editor, and notebook flow](/img/course/ch01-hands-on-env-editor-notebook-flow-en.png)

Open the project in VS Code:

```bash
code .
```

Then check these items:

| Check | Where to look |
|---|---|
| The folder name is `ai-learning-lab` | VS Code Explorer |
| The selected interpreter is the one you expect | Command Palette -> `Python: Select Interpreter` |
| The script runs | VS Code terminal: `python3 src/workstation_check.py` |
| Git changes are visible | Source Control panel |

If you use Jupyter, create `notebooks/01-workstation-review.ipynb` and run a cell like this:

```python
import json
from pathlib import Path

report = json.loads(Path("../reports/workstation-check.json").read_text(encoding="utf-8"))
print(report["python_version"])
print(report["git_branch"]["stdout"])
print(len(report["project_files"]))
```

Expected output:

```text
3.12.3
main
7
```

The file count can be different. The important point is that the Notebook reads the same report generated by your script. This connects exploration (`notebooks/`) with project evidence (`reports/`).

## Step 8: Troubleshoot by locating the broken link

![Chapter 1 workstation troubleshooting map](/img/course/ch01-hands-on-debug-map-en.png)

When an error appears, slow down and locate which link is broken:

| Symptom | First command to run | Likely cause | Fix |
|---|---|---|---|
| `python3: command not found` | `python --version` | Your system uses `python` instead of `python3` | Use `python` consistently, or configure PATH |
| `No such file or directory` | `pwd` and `ls` | You are in the wrong folder | `cd` into `ai-learning-lab` |
| `ModuleNotFoundError` | `which python` and `python -m pip --version` | The package was installed into another environment | Activate the intended environment, then install with `python -m pip install ...` |
| `fatal: not a git repository` | `git status` | You are outside the repository or forgot `git init` | Move into the project folder or run `git init` |
| VS Code runs a different Python | `python3 -c "import sys; print(sys.executable)"` | VS Code interpreter and terminal interpreter differ | Use `Python: Select Interpreter` |
| Jupyter cannot find the report | `Path.cwd()` in a Notebook cell | Notebook path is different from the project root | Use `../reports/...` or move the Notebook |

Do not only copy the last error line. Save the full command, the full output, and what you tried next in `notes/learning-log.md`.

## Step 9: Package the portfolio evidence

![Chapter 1 portfolio evidence pack](/img/course/ch01-hands-on-portfolio-pack-en.png)

Before leaving Chapter 1, your evidence pack should show both result and process:

| Evidence | Minimum acceptable version | Stronger portfolio version |
|---|---|---|
| Run command | `python3 src/workstation_check.py` | README includes command, output, and troubleshooting note |
| Environment proof | Python version in terminal output | `reports/workstation-check.json` records executable and platform |
| Git proof | One commit | Multiple small commits with meaningful messages |
| Editor proof | Project opens in VS Code | Interpreter selected and run output recorded |
| Notebook proof | Optional | Notebook reads the generated report and explains it |
| Debug proof | One error note | A short "symptom -> cause -> fix" table |

## Mini exercises

1. Add a `docs/commands.md` file that records 10 commands you used in this chapter. Commit it with `git commit -m "Add command practice notes"`.
2. Add a `reports/terminal-transcript.txt` file and paste the output of one successful run plus one mistake you fixed.
3. Add a second script named `src/path_check.py` that prints `Path.cwd()` and `Path(__file__).resolve()`.
4. Create a branch named `practice/readme-update`, improve the README, then merge it back into `main`.

## Final self-check

- [ ] I can explain the difference between current folder, project root, and Python file path.
- [ ] I can run the same script from the terminal and from VS Code.
- [ ] I can inspect Git state with `git status --short`.
- [ ] I can make a small branch, commit, and merge it back.
- [ ] I have a report file that proves my workstation can run code.

Once these are checked, Chapter 1 is no longer just a list of tools. It has become a working foundation you can reuse in every later chapter.
