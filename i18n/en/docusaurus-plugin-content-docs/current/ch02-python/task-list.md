---
title: "Stage Learning Task List"
description: "Break the Python programming fundamentals stage into executable learning tasks, practice deliverables, and passing criteria."
keywords: [Python, learning task list, programming fundamentals, AI full-stack]
---

# Stage Learning Task List: Python Programming Fundamentals

The goal of this stage is to help you clearly express the input, processing, and output of a small program using Python. Later data analysis, machine learning, RAG, and Agent work will all depend on Python, so the most important thing here is to write code that is readable, runnable, and debuggable.

![Python Stage Task Workflow](/img/course/ch02-task-list-workflow.png)

## Tasks that must be completed in this stage

| Task | Deliverable | Passing Criteria |
| --- | --- | --- |
| Master basic syntax | 5 small exercise scripts | Can use variables, conditionals, loops, and functions to solve simple problems |
| Become familiar with data structures | A data organization script | Can choose between list, dict, set, and tuple and explain why |
| Write functions | A small program split into multiple functions | Each function has a clear responsibility, with inputs and return values |
| Work with files | A script that reads and writes text or JSON | Can save results to a file and read them back |
| Finish the stage project | A command-line tool | Can explain how to run it and show example output in the README |

## Recommended learning order

First get variables, types, conditionals, loops, and functions working smoothly, then learn lists, dictionaries, and file operations. For advanced topics like OOP, decorators, and generators, you only need to understand their use cases at first; there is no need to master them on the first pass.

When learning Python, do not just read syntax explanations. Every time you learn a syntax point, immediately write a small script, such as a task list, text statistics, file organization, API response parsing, or a learning log generator.

## Relationship to the AI learning assistant project

This stage corresponds to the v0.2 command-line interaction version of the AI learning assistant. You can let the program read a question, save it to a log file, and return a fixed-template answer. At this stage, there is still no need to call a large model; the focus is on practicing input/output, function decomposition, and file logging.

Suggested feature: when the user enters a learning question, the program writes the question, time, and current stage into `notes/questions.jsonl`, then returns the prompt: “Recorded. Please check the relevant chapter first.”

## Common stumbling blocks

Common beginner issues include indentation errors, variable name overwrites, mixing lists and dictionaries, incorrect file paths, functions that only print and do not return, and error messages that are hard to understand. When you encounter an error, first locate the line where it happens, then confirm whether the variable types and values used on that line match your expectations.

## Easy version / Standard version / Challenge version tasks

| Difficulty | What you need to complete | Who it is for |
|---|---|---|
| Easy version | Complete add, view, finish tasks and save JSON | First-time learners, learners with limited time, or complete beginners |
| Standard version | Support categories, search, and damaged file prompts | Learners who want to include this stage in their portfolio |
| Challenge version | Prepare three test cases: normal, empty input, and corrupted JSON | Learners with a foundation who want stronger project evidence |

## Badges and Boss Battle for this stage

| Type | Content |
|---|---|
| Boss Battle | JSON Dungeon Manager |
| Unlockable Badges | JSON Tamer, Exception Catcher |
| Minimum Passing Slogan | Get it running first, then explain it, then record failures |
| Evidence storage suggestion | Save screenshots, logs, failure samples, or evaluation tables to `reports/`, `evals/`, or `logs/` |

You can move on after finishing the easy version; only the standard version is recommended for your portfolio; do the challenge version only if you have extra energy.

## Stage portfolio deliverables

If you want to turn the results of this stage into portfolio material, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `todo_cli.py` or project directory | Source code for a command-line task manager, supporting add, view, complete, and save |
| `data/tasks.json` | Sample data file showing file reading/writing and data structure design |
| `api_demo.py` | A Web API calling exercise, including error handling and response parsing |
| `README.md` | Run commands, example input/output, project structure, and known limitations |
| `debug_notes.md` | Record at least 2 Python errors, their causes, and the fix process |

These materials will turn the Python stage from “syntax practice” into project evidence that shows you can independently write small tools, debug them, and explain their structure.

## Stage passing questions

After finishing this stage, you should be able to answer these questions: when should you write functions, when should you use dictionaries, how are file paths calculated relative to what, what is the difference between print and return, and why real projects need logic split into multiple small functions.

## Completion status Checklist

- [ ] I can use variables, conditionals, loops, and functions to write a complete small program.
- [ ] I can use list and dict to organize tasks, questions, or learning records.
- [ ] I can read and write text, JSON, or JSONL files.
- [ ] I have completed a command-line learning assistant or an equivalent small tool.
- [ ] I can explain the differences between print, return, function parameters, and file paths.
