---
title: "7.8.3 Project: Deliverables Kit"
sidebar_position: 28
description: "Turn an LLM project into a portfolio-ready package with a README template, evaluation records, failure notes, screenshots, and next-step plans."
keywords: [LLM project deliverables, README template, evaluation record, failure notes, portfolio]
---

# 7.8.3 Project: Deliverables Kit

The last step in an LLM project is not just “it runs.”

It is:

> **Can another person understand the goal, reproduce the result, inspect the evaluation, and continue from where you stopped?**

![Project Deliverables Kit](/img/course/ch07-project-deliverables-kit-en.png)

:::tip Why this page exists
Many projects fail the portfolio test not because the model is weak, but because the final handoff is weak. A good project needs evidence, not just a demo.
:::

## What should be in the final package?

At minimum, a good LLM project package should include:

1. `README.md`
2. one reproducible run command
3. example inputs and outputs
4. evaluation records
5. one failure case analysis
6. screenshots or charts
7. a next-step plan

This is the smallest set that lets the project stand on its own.

## The folder structure that makes review easy

```text
project/
├── README.md
├── examples/
│   ├── input-01.json
│   └── output-01.json
├── reports/
│   ├── evaluation.md
│   └── failure_cases.md
├── screenshots/
│   ├── run-01.png
│   └── before-after.png
└── src/
    └── ...
```

This structure is simple on purpose:

- `README.md` tells the story
- `examples/` proves the task
- `reports/` proves the evaluation
- `screenshots/` proves the project runs

## A README template you can reuse

Copy this outline and fill it with your own project content.

```md
# Project Name

## 1. Goal
What problem does this project solve?

## 2. Task Scope
What is in scope and what is not?

## 3. Baseline
What is the simplest method you compared against?

## 4. Data
Where did the data come from? How many samples?

## 5. Evaluation
What metrics or manual checks did you use?

## 6. Results
What improved? What still failed?

## 7. Failure Cases
Show one real failure and explain the cause.

## 8. Run Instructions
How do I reproduce the result?

## 9. Next Steps
What would you improve next?
```

## A simple evaluation record template

If your project has a fixed test set, keep the results in a small table like this:

| Case ID | Input | Baseline | New Method | Pass? | Note |
|---|---|---|---|---|---|
| 001 | Refund request | Generic answer | Domain-aware answer | Yes | Covers policy points |
| 002 | Address change | Too vague | Clear rule-based reply | Yes | Better structure |
| 003 | Invoice question | Misses key detail | Correct answer | No | Need more data |

This makes it easy to compare versions later.

## A failure note template that is actually useful

Do not just write “the model is bad.”

Write a note like this:

```md
# Failure Case: Missing JSON field

- Phenomenon: The output sometimes adds extra text before the JSON object.
- Clues: This happens more often on long prompts.
- Suspected cause: The prompt does not strongly constrain the output format.
- Investigation: Compare prompt versions and inspect the raw outputs.
- Fix action: Add a strict schema reminder and a short example.
- Regression check: Run the same fixed test cases again.
```

That one note is often more valuable than ten screenshots.

## What a strong project handoff looks like

The handoff should let another person answer three questions quickly:

- What problem does this project solve?
- How do I reproduce it?
- Why is the solution better than the baseline?

If those three questions are easy to answer, the project is ready for a portfolio or a team review.

![Project Deliverables Kit](/img/course/ch07-project-deliverables-kit-en.png)

## Final checklist

Before you close the project, check these items:

- README explains the goal and scope
- run command works
- baseline and new method are both shown
- evaluation set is fixed
- at least one failure case is included
- screenshots or charts are present
- next-step plan is written

## Summary

A good LLM project is not just a working script.

It is a package that can be understood, reproduced, evaluated, and extended.

When you can do that, you are no longer just learning techniques.
You are building something other people can actually use.
