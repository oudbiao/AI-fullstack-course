---
title: "6.9 Framework Selection Guide"
sidebar_position: 37
description: "Build a more practical method for choosing an Agent framework by considering task structure, knowledge focus, state complexity, team capability, and long-term maintenance cost."
keywords: [framework selection, LangGraph, LlamaIndex, CrewAI, architecture decision, agent framework]
---

# Framework Selection Guide

:::tip Section Overview
After exploring a full set of frameworks, the truly important question finally arrives:

> **Which one should you choose?**

This is not a “memorize the answer” kind of question. It is a “make a judgment based on task structure” kind of question.
What this lesson will do is make that judgment process clear.
:::

## Learning Objectives

- Learn to judge frameworks based on task structure, not popularity
- Build several of the most practical selection dimensions
- Understand a minimal framework scoring example
- Know when you should even delay using a framework at all

---

## 1. Why is framework selection essentially an architecture decision?

Because once a framework is chosen, many things follow from it:

- Code organization
- Team learning cost
- Debugging approach
- Observability integration method
- Deployment and maintenance complexity

So it is not a trivial dependency. It is more like:

> **You are deciding how the system will grow.**

That is also why “which one is hottest” is usually not the most important question.

---

## 2. First, look at the five most important selection dimensions

### 2.1 Is the task a complex state flow?

If your system has:

- Clear branching
- Loops
- Rollbacks
- Explicit intermediate states

Then graph/workflow-style abstractions are more valuable.

### 2.2 Is the system knowledge / retrieval driven?

If the core challenges are:

- Document ingestion
- Indexing
- Retrieval
- Query organization

Then a knowledge-oriented framework will feel more natural.

### 2.3 Does the task naturally look like role collaboration?

If the task is something like:

- Research
- Writing
- Review

Then team-role division makes a role-based framework a better fit.

### 2.4 What does the team care about more?

For example:

- Higher control
- Lower learning cost
- Faster prototyping
- More stable long-term maintenance

### 2.5 Is the project a demo or a long-term system?

This difference is very important.

- A demo values speed of setup
- A long-term system values clear structure and maintainability

---

## 3. A minimal selection scoring example

This example is not meant to give you a “standard answer.” It is here to teach you:

> First spread out the task dimensions, then make the judgment.

![Agent framework selection decision map](/img/course/ch09-framework-selection-decision-map-en.png)

:::tip Reading the diagram
When choosing a framework, do not first ask “which one is the hottest?” Instead, ask which category the task is closer to: complex state flow, knowledge retrieval, role collaboration, rapid demo building, or a long-term maintainable system. The branches in the diagram are the basis for selection.
:::

```python
frameworks = {
    "langgraph": {"stateful_flow": 9, "knowledge": 6, "role_collab": 6, "ease_of_start": 6},
    "llamaindex": {"stateful_flow": 5, "knowledge": 9, "role_collab": 4, "ease_of_start": 7},
    "crewai": {"stateful_flow": 5, "knowledge": 5, "role_collab": 9, "ease_of_start": 8}
}

weights = {
    "stateful_flow": 0.35,
    "knowledge": 0.25,
    "role_collab": 0.20,
    "ease_of_start": 0.20
}

def score(framework_info, weights):
    return sum(framework_info[k] * weights[k] for k in weights)

for name, info in frameworks.items():
    print(name, "->", round(score(info, weights), 3))
```

### 3.2 What really matters in this code is not the score

What really matters is that you start asking:

- What does my system care about most?
- Why is this dimension weighted more heavily?

That is the framework-selection mindset itself.

---

## 4. Intuitive choices for several typical tasks

### 4.1 If you are building a complex state-flow Agent

Prioritize:

- Graph / workflow-style frameworks

Because you need more of:

- Explicit state
- Conditional edges
- Rollback and retry

### 4.2 If you are building around a knowledge base / RAG main line

Prioritize:

- Retrieval- and knowledge-organization-oriented frameworks

Because your key problems are:

- How documents enter the system
- How retrieval is organized

### 4.3 If you are building a role-based multi-Agent prototype

Prioritize:

- Team / role collaboration frameworks

Because what matters most here is:

- Natural expression of task division
- Clear role relationships

---

## 5. When should you avoid rushing into a complex framework?

### 5.1 A very common but easily overlooked case

If your project is just:

- One model
- One tool
- One linear workflow

Then in many cases:

- Handwritten code
- Lightweight wrapping

Is already enough.

### 5.2 Why might that actually be better?

Because frameworks bring:

- Learning cost
- Abstraction cost
- Debugging cost

If the system is still small, the framework may become extra overhead.

So remember this first:

> **Not every project needs “framework-ness.”**

---

## 6. Why can’t team factors be ignored?

A framework does not only serve an individual developer; it also affects the whole team:

- Is it easy for newcomers to get started?
- Is there enough community documentation?
- Is it easy to investigate issues?
- Will it be easy to maintain later?

A framework that is technically powerful may still have a high real-world cost if no one on the team knows it and the documentation is sparse.

So “team fit” is a very practical dimension in framework selection.

---

## 7. Several of the most common wrong ways to choose

### 7.1 Choosing what is most popular

This is almost the most common misunderstanding.

### 7.2 Choosing what looks coolest in a demo

A good-looking demo does not mean it is suitable for a long-term system.

### 7.3 Choosing a framework before understanding the task structure

This can turn into “forcing a framework onto the problem” instead of “choosing abstractions based on the problem.”

### 7.4 Expecting one framework to solve every problem

In reality, many systems are naturally mixed:

- One style for the retrieval layer
- Another style for the workflow layer

---

## 8. A more practical selection order

Instead of “listing frameworks first,” it is better to follow this path:

1. First write out the main task flow
2. Sketch the state flow or workflow
3. Identify whether knowledge, tools, or roles are the dominant need
4. Then choose the matching abstraction

This way, when you choose a framework, your basis will be much clearer.

---

## Summary

The most important thing in this section is not to find one “uniquely correct framework,” but to learn to:

> **First look at the shape of the task, then look at the shape of the framework.**

When you start thinking in terms of state flow, knowledge organization, role collaboration, and team constraints, framework selection is no longer just about following trends.

---

## Exercises

1. For your current project, assign weights to the four dimensions: “state flow / knowledge organization / role collaboration / ease of getting started.”
2. Think about this: why can forcing a complex framework onto a simple project actually make long-term progress slower?
3. Explain in your own words why framework selection is an architecture decision, not a library choice.
4. If your team values controllability and observability especially highly, what style of framework would you prioritize?
