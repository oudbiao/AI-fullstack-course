---
title: "Learning Rescue for Stuck Points"
sidebar_position: 5
---

# Learning Rescue for Stuck Points

![Troubleshooting map for learning stuck points](/img/course/appendix-troubleshooting-rescue-map.png)

![Flowchart for minimal reproduction and asking for help](/img/course/appendix-debug-mre-help-flow.png)

:::tip Reading guide
When you run into a problem, don’t rush to change direction. Follow the steps “reproduce the problem -> collect the environment -> narrow the input -> record the error -> ask a specific question.” Many stuck points can shift from an emotional problem into an engineering problem that can be located and fixed.
:::

The goal of this page is not to explain theory, but to help you “get moving again as quickly as possible when you get stuck.” Many learning interruptions are not because the material is too hard, but because one small issue has dragged on for too long.

## 1. First figure out what kind of problem you’re stuck on

| Symptom | Most likely type of problem | First reaction |
|---|---|---|
| `ModuleNotFoundError` | Environment or dependencies not installed correctly | Check the current Python environment first |
| Code runs but the result looks weird | Inputs, labels, or evaluation misunderstood | Print intermediate results first |
| Training is very slow or GPU memory blows up | Batch too large, model too heavy, or device mismatch | Shrink the experiment scale first |
| You finish a chapter but still don’t know how to solve the exercises | Concepts are not tied to code | Go back and rerun the minimal example |
| You don’t know how to start a project | The task is too big and not broken down | Draw the smallest closed loop first |
| You’ve learned a lot but can’t remember it | Lacking review and output | Take notes and write short summaries |

## 2. How to check environment issues

Environment problems are the easiest to make people feel stuck, but in fact many of them are repeated problems.

First run this set of commands:

```bash
python --version
which python
pip --version
pip list
pwd
ls
```

If you have a GPU, add one more command:

```bash
nvidia-smi
```

### 2.1 The most common environment problems

#### Problem 1: The package is installed, but import fails

Common reasons:

- You installed it in a different Python environment
- The terminal and the IDE are not using the same interpreter
- `pip` does not point to the current environment

A more reliable way is:

```bash
python -m pip install numpy
python -m pip install pandas
```

This reduces the chance of installing into the wrong environment.

#### Problem 2: Path not found

First confirm:

- Is the current working directory the one you think it is?
- Is the relative path based on the current directory?
- Is the filename capitalization correct?

You can temporarily print:

```python
from pathlib import Path

print(Path.cwd())
print(Path("data").exists())
```

#### Problem 3: Version conflicts

If you keep running into dependency problems, prioritize two things:

1. Put commonly used dependencies into a unified environment file
2. Use separate virtual environments for different major directions whenever possible

Don’t cram all experiments into one dirty environment.

## 3. The code runs, but you don’t know whether it is correct

This kind of problem is more dangerous than a direct error.

### 3.1 Check the inputs first

You should confirm at least:

- Is the number of data samples correct?
- Is the input shape correct?
- Is the label range correct?
- Are there any missing values, garbled text, or duplicate samples?

Example:

```python
print("Number of samples:", len(texts))
print("First two texts:", texts[:2])
print("First two labels:", labels[:2])
print("Label set:", sorted(set(labels)))
```

### 3.2 Then check intermediate results

Don’t only look at the final accuracy. More important is:

- What exactly is the model outputting?
- Are the probabilities reasonable?
- Are the intermediate features all zeros or all the same?

### 3.3 Finally check the evaluation method

Common mistakes:

- Training and test sets are mixed together
- The classes are extremely imbalanced, but you only look at accuracy
- For text tasks, you only inspect a single example and ignore the overall error distribution

## 4. What to do if training does not converge

Don’t rush to blame the model architecture. In many cases, the problem is not that complicated.

You can check in this order:

1. Try a very small dataset to see whether it can overfit
2. Check whether the learning rate is too large or too small
3. Check whether the loss function matches the label format
4. Only then suspect the model architecture and more advanced tricks

If a model cannot even learn a very small batch of data, you should usually check first:

- Whether the inputs are fed in correctly
- Whether the labels are misaligned
- Whether the loss function is chosen correctly
- Whether the optimizer and learning rate are unreasonable

## 5. What to do if GPU memory is not enough

The most direct order is:

1. Reduce `batch size`
2. Reduce the input size
3. Turn off unnecessary logging and caching first
4. Use a smaller model or train only the head
5. Use gradient accumulation or mixed precision when needed

Many beginners immediately think about “getting a bigger GPU,” but shrinking the experiment scale is usually more effective first.

## 6. What to do if you don’t know how to turn a project into something real

If you feel like “I know a little bit of everything, but I don’t know how to build a project,” it is usually because the topic is too big.

A more reliable way to break it down is:

### 6.1 First write a one-sentence goal

For example:

- Build a knowledge base assistant that can answer company policy questions
- Build a small system that can recognize image categories
- Build a tool that can generate summaries for articles

### 6.2 Then break it into the smallest closed loop

A minimal closed loop usually contains only:

1. One clear input
2. One runnable processing pipeline
3. One observable output
4. One basic evaluation method

### 6.3 Then gradually add features

Common additions include:

- Logging
- Error handling
- Caching
- User interface
- Evaluation set
- Deployment

## 7. What to do if you feel anxious about learning and always feel behind

Remember one thing: the AI field changes fast, but core abilities do not change that fast.

The abilities that really matter are:

- Reading code
- Debugging
- Breaking down problems
- Building small closed loops
- Explaining your own solution

If you are already getting better at these five things, you are not falling behind.

## 8. How to ask questions so it’s easier to get useful help

Don’t just send “There’s an error, please help me take a look.” A better question template is:

```text
What I’m doing:
What I expected to see:
What actually happened:
The last 20 lines of the full error:
What I have already tried:
Minimal reproducible code:
```

This makes it easier for others to help you, and it also forces you to clarify the problem first.

## 9. A minimal reproducible template

When you suspect your big project is too messy, first shrink the problem to something like this:

```python
def predict(x):
    return x * 2

data = [1, 2, 3]
preds = [predict(x) for x in data]
print(preds)
```

Then add the real logic back little by little. The point is that you can quickly locate which layer, once added, caused the problem to appear.

## 10. When should you pause, and when should you keep pushing?

Situations where it is appropriate to pause:

- You have been randomly trying things for more than half an hour
- You are mechanically copying commands without knowing what you’re doing
- You can no longer understand the error message

Situations where it is appropriate to keep pushing:

- The problem has been narrowed down to 1–2 verifiable hypotheses
- You know what to try next
- You can explain the purpose of each change

Real effective learning is not about always pushing forward blindly. It is about knowing when to stop and organize your thinking.
