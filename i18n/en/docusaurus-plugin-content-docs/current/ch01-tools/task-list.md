---
title: "Stage Learning Task Sheet"
description: "Break the developer tools basics stage into actionable learning tasks, practice deliverables, and completion criteria."
keywords: [developer tools, learning task sheet, Git, command line, development environment]
---

# Stage Learning Task Sheet: Developer Tools Basics

The goal of this stage is not to memorize every command, but to give you the basic development workflow needed for later AI project learning. After finishing this stage, you should be able to create projects, manage files, install dependencies, use Git to save versions, and record execution results.

![Developer Tools Stage Task Chain](/img/course/ch01-task-list-workflow.png)

## Tasks you must complete in this stage

| Task | Deliverable | Completion Criteria |
| --- | --- | --- |
| Configure the terminal and common commands | A command practice log | You can create, move, view, and delete files with commands |
| Set up a Python development environment | A runnable Python project directory | You can run `python --version`, install dependencies, and execute scripts |
| Configure VS Code or an equivalent editor | A project screenshot or setup notes | You can open the project, run files, and view terminal output |
| Master the basic Git workflow | At least 3 commit records | You can explain the role of add, commit, status, and log |
| Connect to a remote repository | A remote repository URL or screenshot | You can push and confirm the file exists remotely |

## Recommended learning order

First learn the terminal and file operations, then configure the Python environment, then learn the editor, and finally learn Git. Do not get stuck on complex branching models at the beginning. First make sure you can follow the main workflow of “edit files locally, commit, and push.”

If you run into environment issues, prioritize recording the original error message, operating system, Python version, execution directory, and the full command. In later learning on RAG, Agent, and deployment, many problems still come down to environment, paths, and dependencies.

## Relationship to the AI Learning Assistant project

This stage corresponds to the v0.1 project skeleton of the AI Learning Assistant. You need to create a project directory, prepare a README, set up a Git repository, and write the simplest command-line entry point. This version does not need AI capabilities; it only needs to give the project a sustainable engineering foundation for continuous iteration.

Recommended directory structure:

```text
ai-learning-assistant/
  README.md
  src/
    main.py
  notes/
    learning-log.md
  requirements.txt
```

`main.py` can first just print a line of welcome text. The key point is not functionality, but whether you can run it with a fixed command and write the result into the README.

## Common sticking points

The most common problems include running commands in the wrong directory, mixed-up Python versions, installing dependencies into the wrong environment, Git not being configured with a username, and remote repository permission failures. When solving these problems, do not only copy the last line of the error. Keep the full command and the full output.


## Easy / Standard / Challenge tasks

| Difficulty | What you need to complete | Who it is for |
|---|---|---|
| Easy | Run a Python file and complete one Git commit | First-time learners, learners with limited time, or beginners |
| Standard | Add a README, virtual environment notes, and command logs | Learners who want to include this stage in their portfolio |
| Challenge | Intentionally create path or command errors and write troubleshooting notes | Learners with a foundation who want stronger project evidence |

## Stage badge and boss battle

| Type | Content |
|---|---|
| Boss battle | Workbench Guardian |
| Unlockable badges | Terminal Survivor, Git Archivist |
| Minimum completion motto | Get it running first, then explain it, then record the failures |
| Evidence storage suggestion | Save screenshots, logs, failure samples, or evaluation tables in `reports/`, `evals/`, or `logs/` |

Completing the Easy version is enough to move on; completing the Standard version is what we recommend including in your portfolio; only do the Challenge version if you have extra bandwidth.

## Stage portfolio deliverables

If you want to preserve the results of this stage in your portfolio, we recommend keeping at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `README.md` | Project goals, environment requirements, run commands, and example output |
| `requirements.txt` | Current project dependencies; you can keep it even if it is initially empty |
| `notes/learning-log.md` | Record what you did each day and what environment issues you encountered |
| `screenshots/` | Save screenshots of terminal runs, editor usage, and Git commits |
| Git commit history | At least 3 small commits that show the project iteration process |

These materials will prove that you are not just following tutorials, but already have the basic engineering habits needed for continuous AI project development.

## Stage completion questions

After finishing this stage, you should be able to answer these questions: which directory the terminal is currently in, which interpreter the Python script is running with, where the project dependency record is stored, what uncommitted changes Git currently has, and why you should make a commit every time you finish a small feature.

## Completion checklist

- [ ] I can enter the project directory in the terminal and explain where the current command is being executed.
- [ ] I can create, run, and modify a Python file.
- [ ] I can use Git to check status, commit changes, and understand the meaning of commit records.
- [ ] I have already created the project skeleton, README, and run screenshots for the AI Learning Assistant.
- [ ] I have recorded at least one environment or path issue, along with my troubleshooting process.
