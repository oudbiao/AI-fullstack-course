---
title: "7.5 Multi-Agent Practical Patterns"
sidebar_position: 41
description: "From research-oriented, writing-oriented, and development-oriented collaboration to review-oriented collaboration, understand several common ways Multi-Agent systems are combined in real tasks."
keywords: [multi-agent patterns, research team, writer-reviewer, dev team, agent collaboration]
---

# Multi-Agent Practical Patterns

:::tip Section Overview
We have already covered:

- Multi-Agent architecture patterns
- Communication between Agents
- Task assignment and coordination

In this section, we will apply these ideas to scenarios that feel more like “real projects”:

> **In practical tasks, how are multiple Agents usually combined?**
:::

## Learning Objectives

- Understand several high-frequency Multi-Agent practical patterns
- Learn how to choose a more suitable collaboration style based on task goals
- Read a small Multi-Agent workflow example
- Understand why “patterns” matter more than simply “adding more Agents”

---

## 1. Why talk about “practical patterns”?

### 1.1 Because real systems are usually not pure theoretical architectures

Many projects do not say:

- “I want a peer-to-peer Multi-Agent system”

They are more likely to say:

- “I want a research assistant team”
- “I want a writing + review workflow”
- “I want a software development team”

In other words, real projects are more like “task organization forms” than abstract architecture names.

### 1.2 So what is the point of learning practical patterns?

It helps you move from:

- understanding abstract structures

to:

- implementing concrete products

---

## 2. Pattern 1: Research Collaboration

### 2.1 Typical division of work

- Planner: breaks down the problem
- Researcher: retrieves information
- Synthesizer: integrates the results

### 2.2 What tasks is it good for?

- Background research
- Collecting materials
- Producing structured reports

### 2.3 A minimal example

```python
def planner(query):
    return ["collect refund policy", "organize time conditions", "form a conclusion"]

def researcher(task):
    docs = {
        "collect refund policy": "After course purchase, you can get a refund within 7 days if learning progress is below 20%.",
        "organize time conditions": "Key conditions include the time window and learning progress."
    }
    return docs.get(task, "No information found")

def synthesizer(items):
    return "Conclusion: " + " ".join(items)

plan = planner("What is the refund policy?")
materials = [researcher(task) for task in plan[:-1]]
answer = synthesizer(materials)

print(plan)
print(materials)
print(answer)
```

The key idea of this pattern is:

> First expand to collect information, then converge to summarize it.

---

## 3. Pattern 2: Writing + Review

### 3.1 One of the most classic and practical patterns

The usual division of work is:

- Writer: writes the first draft
- Reviewer: checks for issues
- Reviser: revises based on feedback

### 3.2 Why is this pattern so common?

Because many tasks are naturally suited to:

- generation
- checking
- correction

For example:

- report writing
- answer generation
- code documentation

### 3.3 A minimal example

```python
def writer(topic):
    return f"Draft: The key point of {topic} is that refunds are available within 7 days."

def reviewer(draft):
    if "within 7 days" in draft:
        return "Suggest adding the learning progress condition."
    return "Missing the time condition."

def reviser(draft, review):
    return draft + " " + review

draft = writer("refund policy")
review = reviewer(draft)
final = reviser(draft, review)

print(draft)
print(review)
print(final)
```

The biggest advantage of this pattern is:

> It separates “generation ability” from “error-correction ability”.

---

## 4. Pattern 3: Development Team Mode

### 4.1 A common abstraction for an AI development team

For example:

- PM / Planner: defines requirements
- Coder: writes the implementation
- Reviewer: checks the code
- Tester: verifies whether the result meets expectations

### 4.2 Why is this pattern so common in AI coding scenarios?

Because software development already has this kind of role division.  
Multi-Agent simply makes it programmatic and automated.

### 4.3 A minimal example

```python
workflow = [
    {"agent": "planner", "task": "define the feature to implement"},
    {"agent": "coder", "task": "write the implementation code"},
    {"agent": "reviewer", "task": "check for logic issues"},
    {"agent": "tester", "task": "verify whether the output meets expectations"}
]

for step in workflow:
    print(step["agent"], "->", step["task"])
```

The key point of this pattern is not that the roles sound impressive, but that:

> each layer can catch different types of problems.

---

## 5. Pattern 4: Double Verification / High-Risk Review

### 5.1 When is it needed?

If the task is high risk, such as:

- legal advice
- medical assistance
- financial decision-making

then in many cases, you should not let just one Agent produce the conclusion by itself.

### 5.2 Common approach

- One Agent generates the answer
- Another Agent performs fact-checking
- A third Agent checks risk and compliance

This kind of pattern is slower, but more stable.

---

## 6. A Small Multi-Agent Workflow Example

```python
def planner(query):
    return ["retrieve", "write", "review"]

def retriever(query):
    return "Retrieval result: refunds require both time and progress conditions."

def writer(material):
    return f"Answer draft: {material}"

def reviewer(draft):
    if "progress conditions" in draft:
        return {"approved": True, "comment": "The information is fairly complete"}
    return {"approved": False, "comment": "Missing a key condition"}

query = "What is the refund policy?"
steps = planner(query)
material = retriever(query)
draft = writer(material)
review = reviewer(draft)

print("steps  :", steps)
print("draft  :", draft)
print("review :", review)
```

Although this code is small, it already shows the core feel of practical patterns:

- plan first
- execute next
- review afterward

---

## 7. How do you choose the right practical pattern?

### 7.1 If the task mainly involves finding information

Prefer:

- research collaboration

### 7.2 If the task mainly involves content quality

Prefer:

- writing + review

### 7.3 If the task mainly involves engineering delivery

Prefer:

- development team mode

### 7.4 If the task is high risk

Prefer:

- double verification / high-risk review

So the real question is not:

> “Which pattern is the coolest?”

but:

> “Which pattern best fits the current task’s failure risk and goal structure?”

---

## 8. Common Beginner Pitfalls

### 8.1 Tying patterns to a fixed number of roles

It is not “3 Agents must mean one specific pattern.”  
The key is the responsibility relationship, not the number.

### 8.2 Adding patterns just to look complex

For many tasks, a single Agent or two Agents is already enough.

### 8.3 Not having clear evaluation criteria

If you do not know why one pattern is better than another, system iteration will be hard to move forward.

---

## Summary

The most important point in this section is not memorizing labels like “research-oriented” or “development-oriented,” but understanding:

> **The value of Multi-Agent practical patterns is in mapping abstract collaboration structures to real task goals.**

When you can match task shape with collaboration pattern, Multi-Agent will truly move from concept to product.

---

## Exercises

1. Choose a task you are familiar with and judge whether it is more like research collaboration, writing + review, or a development team.
2. Add a `reviser` Agent to the small workflow in this section so that it modifies the draft based on the review.
3. Think about this: why do high-risk tasks need a combination of “generation + verification + risk review”?
4. Explain in your own words: why is the focus of Multi-Agent not the number of roles, but the collaboration structure?
