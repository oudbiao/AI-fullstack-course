---
title: "9.7.5 Multi-Agent Practical Patterns"
description: "From research-oriented, writing-oriented, and development-oriented collaboration to review-oriented collaboration, understand several common ways Multi-Agent systems are combined in real tasks."
sidebar:
  order: 41
head:
  - tag: meta
    attrs:
      name: keywords
      content: "multi-agent patterns, research team, writer-reviewer, dev team, agent collaboration"
---

# 9.7.5 Multi-Agent Practical Patterns

:::tip[Section Overview]
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

## Why talk about “practical patterns”?

### Because real systems are usually not pure theoretical architectures

Many projects do not say:

- “I want a peer-to-peer Multi-Agent system”

They are more likely to say:

- “I want a research assistant team”
- “I want a writing + review workflow”
- “I want a software development team”

In other words, real projects are more like “task organization forms” than abstract architecture names.

### So what is the point of learning practical patterns?

It helps you move from:

- understanding abstract structures

to:

- implementing concrete products

---

## Pattern 1: Research Collaboration

### Typical division of work

- Planner: breaks down the problem
- Researcher: retrieves information
- Synthesizer: integrates the results

### What tasks is it good for?

- Background research
- Collecting materials
- Producing structured reports

### A minimal example

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

Expected output:

```text
['collect refund policy', 'organize time conditions', 'form a conclusion']
['After course purchase, you can get a refund within 7 days if learning progress is below 20%.', 'Key conditions include the time window and learning progress.']
Conclusion: After course purchase, you can get a refund within 7 days if learning progress is below 20%. Key conditions include the time window and learning progress.
```

The key idea of this pattern is:

> First expand to collect information, then converge to summarize it.

---

## Pattern 2: Writing + Review

### One of the most classic and practical patterns

The usual division of work is:

- Writer: writes the first draft
- Reviewer: checks for issues
- Reviser: revises based on feedback

### Why is this pattern so common?

Because many tasks are naturally suited to:

- generation
- checking
- correction

For example:

- report writing
- answer generation
- code documentation

### A minimal example

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

Expected output:

```text
Draft: The key point of refund policy is that refunds are available within 7 days.
Suggest adding the learning progress condition.
Draft: The key point of refund policy is that refunds are available within 7 days. Suggest adding the learning progress condition.
```

The biggest advantage of this pattern is:

> It separates “generation ability” from “error-correction ability”.

---

## Pattern 3: Development Team Mode

### A common abstraction for an AI development team

For example:

- PM / Planner: defines requirements
- Coder: writes the implementation
- Reviewer: checks the code
- Tester: verifies whether the result meets expectations

### Why is this pattern so common in AI coding scenarios?

Because software development already has this kind of role division.
Multi-Agent simply makes it programmatic and automated.

### A minimal example

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

Expected output:

```text
planner -> define the feature to implement
coder -> write the implementation code
reviewer -> check for logic issues
tester -> verify whether the output meets expectations
```

The key point of this pattern is not that the roles sound impressive, but that:

> each layer can catch different types of problems.

---

## Pattern 4: Double Verification / High-Risk Review

### When is it needed?

If the task is high risk, such as:

- legal advice
- medical assistance
- financial decision-making

then in many cases, you should not let just one Agent produce the conclusion by itself.

### Common approach

- One Agent generates the answer
- Another Agent performs fact-checking
- A third Agent checks risk and compliance

This kind of pattern is slower, but more stable.

---

## A Small Multi-Agent Workflow Example

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

Expected output:

```text
steps  : ['retrieve', 'write', 'review']
draft  : Answer draft: Retrieval result: refunds require both time and progress conditions.
review : {'approved': True, 'comment': 'The information is fairly complete'}
```

![Multi-Agent workflow trace result map](/img/course/ch09-multi-agent-practice-trace-result-map-en.webp)

:::tip[Read the handoffs, not just the roles]
The trace matters because each printed row answers a different question: what the planner chose, what material was retrieved, and why the reviewer approved the draft.
:::
Although this code is small, it already shows the core feel of practical patterns:

- plan first
- execute next
- review afterward

---

## How do you choose the right practical pattern?

### If the task mainly involves finding information

Prefer:

- research collaboration

### If the task mainly involves content quality

Prefer:

- writing + review

### If the task mainly involves engineering delivery

Prefer:

- development team mode

### If the task is high risk

Prefer:

- double verification / high-risk review

So the real question is not:

> “Which pattern is the coolest?”

but:

> “Which pattern best fits the current task’s failure risk and goal structure?”

---

## Common Beginner Pitfalls

### Tying patterns to a fixed number of roles

It is not “3 Agents must mean one specific pattern.”
The key is the responsibility relationship, not the number.

### Adding patterns just to look complex

For many tasks, a single Agent or two Agents is already enough.

### Not having clear evaluation criteria

If you do not know why one pattern is better than another, system iteration will be hard to move forward.

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

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

<details>
<summary>Reference implementation and walkthrough</summary>

1. Classify the task by its dominant risk: research collaboration if evidence coverage matters, writing + review if expression and accuracy matter, and development team if implementation and tests matter.
2. A `reviser` Agent should read the draft plus review comments, change only the rejected or weak parts, and return both the revised output and a short change note.
3. High-risk tasks need generation + verification + risk review because fluent output can still be wrong, incomplete, unsafe, or unsupported by evidence.
4. The focus is collaboration structure because roles only help when they create useful boundaries, handoffs, checks, and decisions. A long role list without structure is just extra conversation.

</details>
