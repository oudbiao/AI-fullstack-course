---
title: "5.5 Prompt Engineering Practice"
sidebar_position: 18
description: "From bad prompts to good prompts, systematically practice rewriting, constraints, example design, and output control, and really use Prompt in tasks."
keywords: [prompt engineering, few-shot, instruction design, prompt practice, output control]
---

# Prompt Engineering Practice

:::tip Section Overview
In the previous sections, we covered:

- Prompt basics
- Advanced techniques
- Structured output

In this section, we will not introduce new terms. Instead, we will do something even more important:

> **Treat Prompt as an engineering object and practice with it.**

In other words, do not just ask, “Can this prompt run?” Ask, “Why is it more stable, what is clearer, and why is it more suitable for the current task?”
:::

## Learning Objectives

- Learn how to judge why a prompt is bad
- Learn how to improve a prompt from four angles: goal, constraints, examples, and output format
- Understand the prompt iteration process for several typical tasks
- Build the habit of debugging Prompts instead of writing them by guesswork

---

## 1. The Most Common Misunderstandings About Prompt Engineering

### 1.1 Misunderstanding: A prompt is just “writing more politely”

In fact, Prompt Engineering really cares about:

- Whether the task definition is clear
- Whether the output requirements are explicit
- Whether the constraints are executable
- Whether the examples provide enough guidance

Politeness is usually not the key point.

### 1.2 A more accurate sentence

> **A Prompt is the task interface documentation you write for the model.**

If the documentation is vague, the model’s output will naturally be unstable.

---

## 2. First, Look at a “Bad Prompt”

### 2.1 Task: Sentiment classification for user reviews

A very poor prompt might look like this:

```text
Help me analyze this comment.
```

What is wrong with it?

- It does not say what to analyze
- It does not specify the output format
- It does not define the label set
- It does not say whether an explanation is needed

### 2.2 A clearer version

```text
Please determine the sentiment of the review below. Only output positive or negative. Do not output anything else.

Review: This course is explained very clearly, and there are many examples.
```

This version is much clearer because it defines:

- Task: sentiment classification
- Output set: positive / negative
- Output constraint: no extra content

---

## 3. The Four Core Dimensions of Prompt Debugging

### 3.1 Is the task goal clear enough?

First ask:

- Is the model supposed to classify, summarize, extract, or rewrite?

### 3.2 Is the output format clear enough?

Then ask:

- Is the output a sentence?
- A label?
- JSON?
- A table?

### 3.3 Are the constraints clear enough?

For example:

- Do not hallucinate
- Do not output extra explanations
- Answer only based on the given text

### 3.4 Are the examples guiding enough?

For some tasks, instructions alone are not enough. It is better to add few-shot examples.

These four questions basically form the main thread of Prompt practice.

---

## 4. A Runnable Prompt Practice Helper

The example below does not call a real large model. Instead, it uses a “task specification object” to help you learn how to break down Prompt requirements.

```python
prompt_spec = {
    "task": "sentiment_classification",
    "allowed_labels": ["positive", "negative"],
    "output_format": "single_label",
    "constraints": ["Do not output explanations", "Only output the label"]
}

print(prompt_spec)
```

This example looks simple, but it teaches you something very important:

> Behind a good Prompt, there is usually a clearer task specification.

---

## 5. Prompt Iteration for a Typical Task

### 5.1 Task: Text summarization

#### Version 1: Too vague

```text
Summarize this paragraph.
```

Problems:

- It does not say how long the summary should be
- It does not say what style to use
- It does not say whether key points should be preserved

#### Version 2: More specific

```text
Please summarize the text below into 3 bullet points in Chinese, with no more than 20 characters per point.
```

This is much better.

#### Version 3: Add boundaries

```text
Please summarize the text below into 3 bullet points in Chinese, with no more than 20 characters per point.
Keep only the facts, and do not add any information that is not in the original text.
```

At this point, the prompt has moved from “able to respond” to “more stable and controllable.”

---

## 6. When Is few-shot Especially Useful?

### 6.1 When the task definition is not clear enough from language alone

For example, if you ask the model to decide whether a sentence is:

- fact
- opinion

If you only provide definitions, the model may interpret them inconsistently.
In this case, few-shot examples are very helpful.

### 6.2 An example

```python
few_shot_examples = [
    {"input": "Beijing is the capital of China.", "output": "fact"},
    {"input": "This course is very interesting.", "output": "opinion"}
]

for ex in few_shot_examples:
    print(ex)
```

The role of few-shot is not “writing more words,” but:

> Showing the model the judgment style you want.

---

## 7. How Can You Write Prompts More Stably for Structured Tasks?

### 7.1 A typical scenario: Information extraction

If you only say:

```text
Help me extract resume information.
```

Then the model may:

- Miss fields
- Use inconsistent field names
- Output extra explanations

### 7.2 A better version

```text
Please extract the information from the resume below and output JSON.

Fields:
- name: string
- school: string
- skills: list[string]

Do not output any extra explanation.
```

This clearly explains the task, structure, and boundaries.

---

## 8. The “Minimal Experiment” Habit in Prompt Practice

### 8.1 Do not change too many things at once

The biggest trap in Prompt debugging is this:

- The task description changes
- The examples change
- The output format changes too

Then you have no idea which change actually mattered.

### 8.2 A better way

Change only one variable at a time, for example:

1. First add output constraints only
2. Then add few-shot examples only
3. Then change only the format requirements

This is very similar to tuning model hyperparameters.

![Prompt debugging loop comic](/img/course/ch07-prompt-debug-loop-en.png)

:::tip How to read this loop
Prompt debugging should feel like engineering, not guessing. Prepare test cases, change only one layer at a time, run the same cases again, compare pass/fail results, then record the Prompt version and failure samples. Regression means old cases should still pass after the new change.
:::

---

## 9. A Small Prompt Evaluation Example

### 9.1 First define test samples

```python
test_cases = [
    {"input": "This course is explained very clearly.", "expected": "positive"},
    {"input": "The content is a bit messy.", "expected": "negative"}
]

for case in test_cases:
    print(case)
```

### 9.2 Why is this step important?

Because Prompt Engineering also needs evaluation.
Without test samples, you can only judge whether a prompt is “good” based on feeling.

A more mature approach is:

- Have an input set
- Have expected outputs
- Check whether the prompt consistently matches expectations

---

## 10. Common Pitfalls for Beginners

### 10.1 Not clearly defining the output when writing prompts

This makes post-processing increasingly painful.

### 10.2 Thinking prompt tuning can only rely on inspiration

In fact, it is very similar to ordinary engineering debugging: run small experiments, look at the results, and improve step by step.

### 10.3 Only looking at one successful case

Getting one example right does not mean the prompt is stable.

---

## Summary

The most important thing in this section is not memorizing how many Prompt techniques you know, but building this habit:

> **Treat Prompt as a task interface to design, and as a system component to debug.**

When you start iterating around the task goal, format, constraints, and examples instead of writing one sentence by intuition, Prompt Engineering truly begins to mature.

---

## Exercises

1. Choose a task you are familiar with, first write a “bad prompt,” then improve it step by step into a better version.
2. Add a few-shot version for the “sentiment classification” task.
3. Rewrite the “text summarization” task into a structured output format, such as JSON.
4. Explain in your own words: Why is Prompt Engineering not “writing one nice sentence,” but “designing a task interface”?
