---
title: "2 Python Programming Fundamentals"
sidebar_position: 0
description: "Build the Python skills needed for AI work: input, data structures, functions, files, errors, APIs, and small projects."
keywords: [Python Basics, Python Fundamentals, Python Tutorial, Programming Basics, AI API]
---

# 2 Python Programming Fundamentals

![Main visual for Python programming fundamentals](/img/course/ch02-python-foundation-en.png)

Chapter 2 has one job: help you turn a small idea into **Python code that runs, saves data, handles errors, and can be explained**.

## See The Python Work Loop

![Python AI backbone capability chain](/img/course/ch02-python-ai-backbone-en.png)

Read the picture first. Most beginner Python programs are this loop:

```text
input -> data structure -> function -> file/API/output
```

Python matters in AI because the same loop later becomes data cleaning, model training, RAG retrieval, API wrapping, and Agent tools.

## Learning Order And Task List

Use this table as both the chapter guide and the task sheet.

| Page | Follow-along action | Evidence to keep |
|---|---|---|
| [2.1.1 Python Introduction](ch01-basics/01-intro.md) to [2.1.5 Flow Control](ch01-basics/05-control-flow.md) | Type small scripts with variables, input/output, conditions, and loops | 5 changed scripts with expected output |
| [2.1.6 Data Structures](ch01-basics/06-data-structures.md) | Store the same task list with a list, dict, and JSON-shaped object | A note explaining why one structure fits best |
| [2.1.7 Function Basics](ch01-basics/07-functions.md) and [2.1.8 Modules and Packages](ch01-basics/08-modules.md) | Split repeated logic into functions and a module | A script with clear inputs and return values |
| [2.2.2 Exception Handling](ch02-advanced/02-exceptions.md) and [2.2.3 File Operations](ch02-advanced/03-file-io.md) | Save data, read it back, and handle a missing or broken file | A JSON/text file plus one debug note |
| [2.2.1 OOP](ch02-advanced/01-oop.md), [2.2.5 Iterators](ch02-advanced/05-iterators-generators.md), and [2.2.6 Type Hints](ch02-advanced/06-type-hints.md) | Skim first, then return when a project needs structure or clarity | One refactored function or class |
| [2.3.1 Task Manager](ch03-projects/01-todo-cli.md) to [2.3.4 AI API Experience](ch03-projects/04-ai-api-experience.md) | Build small projects that save data, collect data, expose an API, and call an AI API | Project folders with README run commands |
| [2.3.5 Follow-Along Workshop](ch03-projects/05-hands-on-python-workshop.md) | Combine CLI commands, JSON persistence, stats, and report export | `ch02_output/` plus terminal output |

Key terms for this chapter:

| Term | Meaning |
|---|---|
| `CLI` | Command-Line Interface: a program controlled by typed commands |
| `I/O` | Input/Output: data entering a program and results leaving it |
| `JSON` | A text format for nested data such as tasks and API responses |
| `API` | A doorway that lets one program call another program |
| `SDK` | A library that wraps an API into easier functions |

## First Runnable Loop

Run this in an empty practice folder. It creates a tiny JSON task manager without any third-party package.

```python
import json
from pathlib import Path

DATA = Path("tasks.json")

def load_tasks():
    if not DATA.exists():
        return []
    try:
        return json.loads(DATA.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return []

def save_tasks(tasks):
    DATA.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

tasks = load_tasks()
tasks.append({"title": "Learn Python file I/O", "done": False})
save_tasks(tasks)
print(f"saved {len(tasks)} task(s)")
```

Expected output:

```text
saved 1 task(s)
```

Run it twice. The second run should print `saved 2 task(s)`. That proves the program can save state and read it back.

## Common Failures

| Symptom | First thing to check | Usual fix |
|---|---|---|
| Syntax error | The reported line and the line above it | Check indentation, parentheses, quotes, and colons |
| File not found | The current working directory | Print `Path.cwd()` and move the file or change the path |
| JSON parsing failed | Whether the file is empty or malformed | Add `try/except` and fall back to an empty list |
| Function is confusing | Inputs, return value, and hidden global state | Split it into smaller functions with one responsibility |
| API call fails | Parameters, status code, and returned error body | Print the response safely and handle the error path |

## Pass Check

Move to Chapter 3 when you can answer these five questions:

- What data enters the program, and what result leaves it?
- When is a dictionary better than a list?
- What folder is a file path relative to?
- What is the difference between `print` and `return`?
- Can another person run your project from the README?

For a printable checklist, use [2.0 Study Guide and Task Sheet](./study-guide.md). The next chapter will use Python to process CSV files, analyze data, and connect databases.
