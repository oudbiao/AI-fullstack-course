---
title: "2 Python Programming Fundamentals"
sidebar_position: 0
description: "Build the programming skills needed for AI full-stack learning, from Python syntax, data structures, functions, modules, and file operations to Web API and AI API calls."
keywords: [Python Basics, Python Fundamentals, Python Tutorial, Programming Basics, AI API]
---

# 2 Python Programming Fundamentals

![Main visual for Python programming fundamentals](/img/course/ch02-python-foundation-en.png)

This stage focuses on whether you can “describe problems with code, process data, and call services.” Data analysis, machine learning, RAG, and Agent work later will all rely on Python again and again, so the goal here is not to memorize syntax, but to build programming thinking and debugging habits.

## Story-Based Introduction: Treat Python as Your First AI Swiss Army Knife

Imagine you are running a small “AI studio”: every day you need to organize tasks, scrape information, save data into files, provide an interface for others, and eventually connect an LLM to build a small assistant. Python is the toolkit that runs through all of this work. At first it may look like just variables, loops, and functions, but every syntax point you learn will later become a real capability: automatically organizing data, batch-processing files, calling APIs, and wrapping services.

## Learning Quest Map

![Python learning quest map](/img/course/ch02-learning-quest-map-en.png)

## Interactive Practice: Ask Yourself One Question After Each Chapter

When learning basic syntax, ask yourself, “Can this knowledge help me handle input and produce output?” When learning data structures, ask yourself, “Is this set of information better stored in a list, a dictionary, or a file?” When learning functions and modules, ask yourself, “Will I use this logic repeatedly in the future?” When working on projects, ask yourself, “If someone else uses it, will they know how to run it, how to exit it, and where errors might happen?”

## Project Bonus

After completing this stage, you will not just have “learned Python”; you will be able to organize four small projects into a portfolio: a command-line tool that saves tasks, a web scraper that collects webpage data, a Web API that others can call, and an API program that talks with AI. These four works together form the smallest prototype of RAG and Agent work later on.

## Acronym and Keyword Map for This Stage

You will meet many English abbreviations in Python and AI tutorials. Do not memorize them as isolated words. Read each one as a small capability:

| Term | Full name | Beginner meaning | Where you use it |
|---|---|---|---|
| AI | Artificial Intelligence | Software that performs tasks that look intelligent, such as understanding text or recognizing images | The whole course |
| API | Application Programming Interface | A doorway that lets one program call another program | Web API and AI API projects |
| CLI | Command-Line Interface | A text-based interface where users type commands | Task manager project |
| I/O | Input/Output | Data entering a program and results leaving a program | `input()`, `print()`, files, APIs |
| JSON | JavaScript Object Notation | A lightweight text format for nested data | Saving tasks, API responses |
| CSV | Comma-Separated Values | A table saved as plain text | Data files and reports |
| HTTP | HyperText Transfer Protocol | The request-response rule used by browsers and web APIs | Web scraper and FastAPI |
| HTML | HyperText Markup Language | The tag-based structure of a web page | Web scraper parsing |
| SDK | Software Development Kit | A library that wraps an API so you can call it more easily | OpenAI SDK and cloud APIs |
| LLM | Large Language Model | A model trained on large-scale text that can generate and understand language | AI API project |
| RAG | Retrieval-Augmented Generation | Letting an LLM look up external knowledge before answering | Later knowledge-base projects |
| Agent | AI system that plans steps and uses tools | An assistant that can search, call APIs, and act through code | Later Agent projects |

When a later article uses one of these terms, pause for one second and ask: **is this about data, an interface, a model, or a workflow?** That small habit prevents vocabulary from becoming fog.

## Stage Positioning

| Item | Description |
|---|---|
| Suitable for | Learners who know a little programming but are not systematic with Python, or who want to enter AI with Python |
| Estimated time | 90–130 hours |
| Prerequisites | Complete the developer tools basics and be able to use the terminal and editor |
| Stage output | Command-line tool, webpage data collection script, simple Web API, AI API practice project |

## Minimum Path for Beginners

Beginners should first focus on high-frequency skills such as variables, conditions, loops, lists, dictionaries, functions, file reading/writing, and exception handling. There is no need to struggle with every advanced syntax point at the start. As long as you can independently complete a command-line task manager and understand the main flow of crawler, Web API, and AI API examples, you have completed the minimum path.

## Advanced Deep-Dive Path

Experienced learners can focus on module decomposition, type annotations, error handling, code quality, and project refactoring. Try writing the same feature as a script, as a module, and as an API, and feel how Python changes from a small utility into an engineering project.

## What Beginners Should Do First, and What Advanced Learners Should Do Later

When learning this stage for the first time, beginners should start by writing small programs around “input → processing → output.” Variables, lists, dictionaries, functions, files, and exception handling are enough to complete most beginner projects; you do not need to chase all advanced syntax at the beginning.

Experienced learners can focus more on engineering: how to split one feature into modules, how exceptions are propagated upward, how type annotations make code clearer, and how a script gradually evolves into an API. Your goal is to write Python code that later data, model, and LLM projects are willing to reuse.

## Why Python Is the Main AI Language

Python became mainstream in AI not because its syntax is the most powerful, but because it connects data processing, machine learning, deep learning, Web APIs, automation scripts, and the LLM ecosystem at the same time. Later you will use it to read data, train models, call LLMs, build RAG, wrap tools, and write Agents.

![Python AI backbone capability chain](/img/course/ch02-python-ai-backbone-en.png)

## Learning Path for This Stage

The first chapter covers Python language basics, including variables, types, operators, input/output, control flow, data structures, functions, and modules. You should focus on understanding the main thread of “input, processing, output.”

The second chapter covers advanced Python topics, including object-oriented programming, exception handling, file I/O, functional programming, iterators and generators, type annotations, and code quality. You do not need to memorize all advanced syntax at once, but you should know what problems they solve.

The third chapter moves into hands-on projects. You will combine the previous knowledge to complete a command-line task manager, a web scraper, a Web API, and a quick AI API demo.

If you want one page that connects the whole stage into a follow-along flow, do [2.5 Follow-Along Workshop: Build a Local Learning Task Assistant](./ch03-projects/05-hands-on-python-workshop.md). It uses only the Python standard library to practice CLI commands, JSON saving, error handling, statistics, and report export.

## What You Should Be Able to Do After Learning This Stage

- Break a small requirement into functions, modules, and files
- Read and write common file types such as JSON, CSV, and plain text
- Install and use third-party libraries
- Understand basic error messages and locate problems through print statements, breakpoints, or logs
- Call external APIs and handle request parameters, return results, and exceptions
- Complete a small Python project with a clear structure

## Common Misconceptions

Do not try to memorize every syntax point all at once. What matters most is being able to use them repeatedly in projects. Lists, dictionaries, functions, file operations, exception handling, and third-party library calls are much more useful than obscure syntax.

Also, do not get stuck too early on “writing elegant code.” The first priority is to write code that runs, can be explained, and can be modified. After you finish a few projects, then gradually pay more attention to type annotations, module decomposition, and coding style.

## Programming Error Theater: Where to Check First When Code Won’t Run

If your program raises a syntax error, first check the parentheses, indentation, and colons around the reported line; if the result is wrong, first print key variables and confirm whether each step of input, processing, and output matches expectations; if file reading/writing fails, first check the path, encoding, and whether the file exists; if API calling fails, first check the request parameters, status code, and returned error message.

## Minimum Runnable Experiment: JSON Task Manager

The minimum experiment for this stage is to write a command-line program that can save data. It can be very simple: use a list to store tasks, persist them with a JSON file, and support adding, viewing, and completing tasks. This experiment uses variables, lists, dictionaries, functions, file reading/writing, and exception handling at the same time.

```python
import json
from pathlib import Path

DATA = Path("tasks.json")

def load_tasks():
    if not DATA.exists():
        return []
    return json.loads(DATA.read_text(encoding="utf-8"))

def save_tasks(tasks):
    DATA.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")

tasks = load_tasks()
tasks.append({"title": "Learn Python file I/O", "done": False})
save_tasks(tasks)
print(tasks)
```

You do not need to build a complete CLI right away. First prove that the program can accept input, save state, and read it back, then gradually add command-line arguments and error handling.

## Python Failure Case Library: Check Input, Path, and Data Structure First

| Symptom | Common Cause | How to Locate | Fix Direction |
|---|---|---|---|
| Syntax error | Indentation, parentheses, colons, or quotes do not match | Check the reported line and the line above it | Narrow the code range and run it piece by piece |
| File not found | The current working directory is different from the file location | Print `Path.cwd()` | Use the project root or specify the path clearly |
| JSON parsing failed | The file is empty, malformed, or encoded inconsistently | Open the raw file and inspect the content | Add exception handling and a default empty value |
| Functions become messy | Input/output is unclear and there are too many global variables | Write comments: what are the parameters, what is returned | Split into smaller functions and reduce hidden state |

## Stage Assessment Rubric

| Level | Assessment Criteria | Portfolio Evidence |
|---|---|---|
| Minimum pass | Can write functions, read/write files, and handle common exceptions | `todo_cli.py`, sample JSON |
| Recommended pass | Can split a script into modules and clearly document run commands | Project structure, README, sample input/output |
| Portfolio pass | Can build a reusable small tool or API entry point | Error-handling notes, test cases, version iteration notes |

## Stage Projects

The basic version is to complete a command-line task manager that can add, view, complete, and delete tasks, and save data to a JSON file. The standard version continues with a web scraper and a Web API, connecting “data collection, data storage, and external interfaces.” The challenge version integrates an AI API to build a small assistant with multi-turn conversation, exception handling, and token usage prompts, and then organizes the four projects into a portfolio.

If you want a more detailed learning rhythm, you can read [Study Guide: How to Learn Python Programming Without Getting Stuck Easily](./study-guide.md).


## Fun Task Card for This Stage

| Playstyle | Task for This Stage |
|---|---|
| Story mission | Give the assistant a notebook: it can add, view, complete learning tasks, and safely save data to JSON. |
| Boss fight | **The JSON Dungeon Manager** |
| Unlockable badges | JSON Tamer, Exception Catcher |
| Easy mode for beginners | Complete just one minimal input-to-output loop and keep a screenshot or command output first |
| Portfolio evidence | A CLI that can handle both normal and abnormal input |

If this stage feels like a lot, treat this task card as your minimum target. Once you can complete the easy mode for beginners, you can keep learning; when you prepare a portfolio later, come back and upgrade to the standard and challenge versions.

## Stage Deliverables

| Deliverable | Minimum Version | Portfolio Version |
|---|---|---|
| Command-line tool | Can add, view, complete, and delete tasks | Supports JSON saving, error handling, and module decomposition |
| File I/O exercise | Can read/write text, CSV, or JSON | Includes data format notes, exception handling, and sample data |
| API calling script | Can call a public Web API | Includes parameter validation, status code handling, and return structure notes |
| Python project README | Clearly states run commands and sample output | Includes project structure, dependencies, input/output, and known limitations |
| Debugging record | Records at least 1 error and the fix process | Forms notes on common errors, locating methods, and postmortem review |
| Follow-along workshop evidence | Run the 2.5 CLI assistant and keep generated output | Includes script, JSON data, Markdown report, terminal output, and error-handling notes |

## Relationship to the AI Learning Assistant Continuous Project

This stage can correspond to AI Learning Assistant v0.2: build a command-line learning task manager that supports adding, viewing, completing, and saving to JSON. If you are learning along the continuous project path, it is recommended that by the end of this stage you submit at least one version log: what capabilities were added in this stage, how to run it, what the sample input/output is, what problems were encountered, and what you plan to improve next.


## Stage Completion Criteria

| Pass Level | What You Need to Do |
|---|---|
| Minimum pass | Can write functions, read/write files, call modules, and complete a small command-line project. |
| Recommended pass | Complete at least one runnable small project in this stage and document the run method, sample input/output, and problems encountered in the README. |
| Portfolio pass | Connect the output of this stage to the continuous “AI Learning Assistant” project and leave screenshots, logs, evaluation samples, and a next-step plan. |

After finishing this stage, you do not need to memorize every detail. More importantly, you should be able to explain clearly: what problem this stage solves, how it relates to the previous stage, and how it will support later learning. The next stage will use Python for data processing, CSV analysis, and database connections.
