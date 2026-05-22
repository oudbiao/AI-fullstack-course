---
title: "2.0 Study Guide and Task Sheet: Python Programming Fundamentals"
description: "A short printable checklist for Chapter 2 after the main guide has been merged into the chapter entry page."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Python study guide, Python task sheet, programming fundamentals, JSON, CLI"
---
![Python learning guide minimal closed loop](/img/course/ch02-study-guide-program-loop-en.webp)

The main study route is now in [Chapter 2 entry](./). Use this page only as a quick checklist while you practice.

## One-Line Mental Model

```text
input -> data structure -> function -> file/API/output
```

If a topic does not help this loop yet, skim it first and return later.

## Practice Checklist

| Check | Evidence |
|---|---|
| I can run 5 tiny scripts with variables, conditions, and loops | `practice/` folder |
| I can choose between list, dict, tuple, and set | short data-structure note |
| I can turn repeated code into a function | refactored script |
| I can save and reload JSON | `tasks.json` |
| I can handle one broken file or bad input | debug note |
| I can finish the workshop | `ch02_output/` |

## Depth Checks

| Skill | Challenge |
|---|---|
| Data choice | Store the same task as a list item and as a dictionary. Explain which one is easier to extend. |
| Error handling | Break `tasks.json` on purpose, then make the program recover without hiding the error from the user. |
| Refactoring | Move repeated code into a function whose input and return value can be tested without touching files. |
| Communication | Write a README command that a new terminal can run without guessing hidden setup steps. |

<details>
<summary>Check reasoning and explanation</summary>

1. The minimum pass is a rerunnable folder with small scripts and workshop output, not screenshots alone.
2. A data-structure note should explain the tradeoff, such as using a dictionary for task `id`, `status`, and `due_date`, but a list for the ordered sequence of tasks.
3. A refactored function should have clear inputs and a return value, and be testable without calling `input()` or reading a file.
4. Broken JSON recovery should tell the user the file was reset or backed up. It should not silently erase data.
5. A README passes when another learner can create the environment, run the script, and see the same expected output.

</details>

## Ready To Continue

Continue to Chapter 3 when your task manager can add a task, save it, reload it, and explain the run command in a README.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
program_loop: input, processing, output, and saved state if any
code_file: Python file or notebook cell that can be rerun
output: printed result, file result, or user-facing behavior
failure_check: syntax, path, type, dependency, or control-flow issue
Expected_output: a rerunnable Python artifact that prepares for data and AI apps
```
