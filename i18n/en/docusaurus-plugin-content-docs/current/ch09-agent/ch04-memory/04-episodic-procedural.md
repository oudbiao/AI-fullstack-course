---
title: "4.5 Episodic and Procedural Memory [Optional]"
sidebar_position: 22
description: "Understand why episodic memory and procedural memory are especially important for complex Agents from two perspectives: “remembering what happened” and “remembering how to do things.”"
keywords: [episodic memory, procedural memory, workflow memory, agent experience, skill memory]
---

# Episodic and Procedural Memory [Optional]

:::tip Section overview
Long-term memory is more like a “stable archive.”  
But some information is neither a long-term archive nor a short-term window, and is more like:

- A specific experience
- A validated way of doing things

This is the layer that episodic memory and procedural memory are meant to address.

In one sentence:

> **Episodic memory is “what I experienced,” and procedural memory is “what I learned how to do.”**
:::

## Learning objectives

- Understand the difference between episodic memory and procedural memory
- Understand why these two kinds of memory are especially important for complex tasks
- Use a runnable example to understand the smallest loop of “extracting a workflow from experience”
- Learn how to judge whether a piece of information is better stored as an episode or as a workflow

---

## 1. What exactly is episodic memory?

### 1.1 It is more like a single experience

For example:

- Last week, I handled a refund dispute because the learning progress exceeded the threshold
- During one weekly report generation, a database API timeout caused the task to fail

The characteristics of this kind of memory are:

- It has time and context
- It includes a specific event process
- It is not always directly reusable

### 1.2 Why does an Agent need episodic memory?

Because complex systems often need to refer to things that happened in the past:

- What problems the user encountered before
- How a similar task ended last time
- Which situations are likely to fail

This kind of information is not a static archive, but it is very helpful for decision-making.

---

## 2. What is procedural memory?

### 2.1 It is more like skills and workflows

For example:

- When handling a refund issue, first check the order, then check the policy, then determine eligibility
- When creating a competitor report, first collect data, then categorize it, then summarize it

The focus of this kind of memory is not “which specific time in the past,”  
but rather “what method can be reused the next time a similar task appears.”

### 2.2 Why is procedural memory important?

Because it helps an Agent avoid planning from scratch every time.  
Many tasks are not completely new problems, but rather:

- New instance + old workflow

In this case, procedural memory can significantly reduce reasoning burden.

---

## 3. What is the biggest difference between them?

### 3.1 Episodic memory answers “what happened”

Example:

- “Last time I generated a weekly report, too many logs caused the summary quality to drop”

### 3.2 Procedural memory answers “how similar problems are usually handled”

Example:

- “The general steps for generating a weekly report are: pull data -> cluster issues -> generate summary -> review”

### 3.3 An analogy

Episodic memory is like a project postmortem.  
Procedural memory is like an SOP manual.

Both are important, but they serve different purposes.

---

## 4. First, run an “experience -> workflow” example

The example below does two things:

1. Records several specific episodes
2. Extracts a reusable workflow from those episodes

```python
from dataclasses import dataclass


@dataclass
class Episode:
    task_type: str
    context: str
    steps: list
    result: str


episodes = [
    Episode(
        task_type="refund_case",
        context="The user asked whether an unshipped order can be refunded",
        steps=["Check order status", "Check refund policy", "Determine eligibility", "Return conclusion"],
        result="success",
    ),
    Episode(
        task_type="refund_case",
        context="The user asked whether a refund is allowed after learning progress exceeded 20%",
        steps=["Check order status", "Check refund policy", "Determine eligibility", "Return conclusion"],
        result="success",
    ),
    Episode(
        task_type="weekly_report",
        context="Weekly report generation task",
        steps=["Pull data", "Cluster issues", "Write summary"],
        result="partial_failure",
    ),
]


def build_procedural_memory(episodes, min_support=2):
    grouped = {}
    for episode in episodes:
        key = (episode.task_type, tuple(episode.steps))
        grouped[key] = grouped.get(key, 0) + 1

    workflows = {}
    for (task_type, steps), count in grouped.items():
        if count >= min_support:
            workflows[task_type] = list(steps)
    return workflows


procedural_memory = build_procedural_memory(episodes)
print("procedural_memory:", procedural_memory)
```

### 4.1 What does this code actually show?

It shows that:

- Episodic memory can accumulate “specific things we have done”
- After enough repetitions, it can be abstracted into procedural memory

In other words, procedural memory is often not written from scratch,  
but distilled from repeatedly successful episodes.

### 4.2 Why didn’t `weekly_report` enter procedural memory?

Because it appeared only once,  
so there was not enough support.

This matches reality very well:

- A one-time success or failure is not necessarily worth turning into a standard workflow immediately

### 4.3 Why is this more insightful than writing a workflow directly?

Because it shows a very realistic process of knowledge accumulation:

- Do it first
- Then review it
- Finally abstract it into a workflow

This is exactly the path many mature Agent systems follow as they evolve.

---

## 5. How is episodic memory usually used in a system?

### 5.1 Retrieve similar cases

When facing a current problem, the system can first check:

- Whether a similar situation happened before
- How it was handled
- What the final result was

### 5.2 Review failures

If a certain type of task often fails,  
episodic memory is very suitable for answering:

- Which step is likely to go wrong
- Which context tends to trigger failure

### 5.3 Serve as training material for procedural memory

It can also become the raw data for later workflow abstraction.

---

## 6. How is procedural memory usually used in a system?

### 6.1 As the default template for the planner

Once a task type is recognized, the system can directly load:

- A default workflow

### 6.2 As a skill library

In essence, many procedural memories are like:

- Reusable skills
- Standard operating procedures
- Task templates

### 6.3 As a safety boundary

Procedural memory can also act as a “don’t improvise” mechanism.  
For high-risk tasks, for example, the system may only be allowed to follow approved workflows.

---

## 7. Common pitfalls

### 7.1 Mistake 1: Calling all history episodic memory

Not all history is worth keeping as an episode.  
Episodes are better for records with:

- A clear task
- A process
- A result

### 7.2 Mistake 2: Assuming every episode automatically becomes procedural memory

Procedural memory needs:

- Repetition
- Stability
- Transferability

### 7.3 Mistake 3: Never updating procedural memory after it is written

If the workflow changes, procedural memory should also be updated.  
Otherwise, it will go from “experience” to “outdated experience.”

---

## Summary

The most important thing in this section is to establish a clear accumulation logic:

> **Episodic memory preserves specific experiences, while procedural memory abstracts repeatedly validated experiences into reusable workflows.**

Once you understand this logic,  
a memory system is no longer just an “archive,” but something that truly helps an Agent learn.

---

## Exercises

1. Add two more successful `weekly_report` cases to the example so that it can also accumulate procedural memory.
2. Think about which tasks are better handled by checking episodes, and which tasks are better handled by directly applying a workflow.
3. Why is procedural memory more like a “skill library” rather than just a “history log”?
4. If a workflow has become outdated, how would you design an update mechanism?
