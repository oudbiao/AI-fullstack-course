---
title: "2.0 Study Guide and Task Sheet: Python Programming Fundamentals"
sidebar_position: 1
description: "A compact study guide and task sheet for Chapter 2: learn Python syntax, data structures, functions, files, exceptions, and projects in the right order."
keywords: [Python study guide, Python task sheet, programming fundamentals, JSON, CLI]
---

# 2.0 Study Guide and Task Sheet: Python Programming Fundamentals

Use this page as the control panel for Chapter 2. Your goal is not to memorize every syntax rule. Your goal is to write a small Python program that runs, saves data, handles errors, and can be explained.

## 2.0.1 Recommended Learning Order

![Python learning guide minimal closed loop](/img/course/ch02-study-guide-program-loop-en.png)

Keep one line in mind: **input -> data structure -> function -> file or output**.

| Order | Section | Focus | Evidence to leave behind |
|---|---|---|---|
| 1 | `2.1 Python Language Basics` | Variables, types, operators, input/output, control flow | 5 tiny scripts you typed and changed |
| 2 | `2.1 Data Structures and Functions` | Lists, dictionaries, functions, modules | A script split into small functions |
| 3 | `2.2 Intermediate Python` | Classes, exceptions, files, iterators, type hints | A file-reading or JSON-saving example |
| 4 | `2.3 Stage Projects` | CLI, scraper, Web API, AI API experience | One runnable project folder |
| 5 | `2.5 Follow-Along Workshop` | CLI commands, JSON persistence, stats, report export | `ch02_output/` and a README note |

Do not start by chasing advanced syntax. First make the program run, save something, read it back, and explain why each function exists.

## 2.0.2 Terms You Need Before You Start

| Term | Full name | First meaning in this chapter |
|---|---|---|
| `CLI` | Command-Line Interface | A program controlled by typed commands. |
| `I/O` | Input/Output | Data entering a program and results leaving it. |
| `JSON` | JavaScript Object Notation | A lightweight text format for saving nested data. |
| `API` | Application Programming Interface | A doorway that lets one program call another program. |
| `SDK` | Software Development Kit | A library that wraps an API into easier functions. |

Read abbreviations as capabilities: interface, storage, request, response, or workflow.

## 2.0.3 Tasks You Must Complete in This Stage

| Task | Deliverable | Pass criteria |
|---|---|---|
| Master the basic loop | 5 small scripts | Can use variables, conditionals, loops, and functions to solve simple tasks |
| Organize data | A list/dict practice file | Can choose list, dict, set, or tuple and explain the choice |
| Split logic into functions | A small program with multiple functions | Each function has clear inputs, outputs, and responsibility |
| Read and write files | Text, JSON, or JSONL example | Can save results and read them back without guessing paths |
| Handle errors | A debug note | Can locate the line, inspect variable values, and fix at least 2 common errors |
| Complete the guided workshop | `ch02_output/` | Can run commands, save JSON, mark tasks done, show stats, and export a report |

## 2.0.4 Mini Project Pattern

For every Python project in this stage, use the same simple pattern:

1. Write the run command.
2. Create one tiny input.
3. Print or save one clear output.
4. Add one error case.
5. Record what changed in the README.

This pattern is enough for the task manager, scraper, Web API, and AI API demo.

## 2.0.5 Stage Portfolio Deliverables

| Deliverable | What it proves |
|---|---|
| `todo_cli.py` or project folder | You can build a small command-line tool. |
| `data/tasks.json` | You can persist and reload structured data. |
| `api_demo.py` | You can call an external service and inspect the response. |
| `README.md` | Others can rerun your project. |
| `debug_notes.md` | You can learn from errors instead of hiding them. |
| `ch02_output/` | You completed the follow-along workshop evidence pack. |

## 2.0.6 Stage Completion Questions

Before moving to Chapter 3, check that you can answer these questions:

- When should a repeated block become a function?
- When is a dictionary better than a list?
- What folder is a file path relative to?
- What is the difference between `print` and `return`?
- How does one script become a small project with modules, data, and a README?

You are ready to continue when you can build a command-line task manager from scratch and add one small feature such as search, sorting, categories, or export.
