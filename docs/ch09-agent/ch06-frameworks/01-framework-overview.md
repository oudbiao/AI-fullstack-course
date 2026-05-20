---
title: "9.6.2 Agent Framework Overview"
sidebar_position: 30
description: "From why frameworks are needed to the differences among Agent frameworks in abstraction level, control, and applicable scenarios, build a map for choosing frameworks."
keywords: [agent frameworks, LangGraph, LlamaIndex, CrewAI, AutoGen, framework overview]
---

# 9.6.2 Agent Framework Overview

:::tip Where this section fits
When people reach the Agent stage, many quickly fall into another trap:

> “There are so many frameworks — which one should I learn?”

This lesson is not about picking sides or memorizing framework names. It is about building a decision map first:

> What exactly do different frameworks abstract away, what do they give up, and what are they best suited for?
:::

## Learning objectives

- Understand why Agent projects often introduce frameworks
- Distinguish differences in abstraction levels across frameworks
- Know what frameworks usually save you from, and what they make you lose
- Build an initial perspective for choosing frameworks

![Agent frameworks position map](/img/course/ch09-frameworks-position-map-en.webp)

Read this map as a trade-off view: lower-level frameworks give more control, higher-level frameworks reduce setup cost, and retrieval-oriented frameworks are stronger when the system is centered on documents and knowledge.

---

## Why do we need Agent frameworks?

### What happens if you write everything yourself without a framework?

A slightly more complex Agent system usually requires you to handle:

- State management
- Tool calling
- Message passing
- Failure retries
- Trace
- Multi-Agent collaboration

Of course, you can write it by hand, but you will quickly run into:

- Lots of repetitive boilerplate code
- Inconsistent structure across projects
- Increasing difficulty in debugging and extending

### What do frameworks actually solve?

You can remember it with one sentence:

> **A framework does not build the product for you; it helps you set up the high-frequency structure first.**

For example:

- Graph-based state flow
- Tool registration mechanisms
- Abstractions for Agent collaboration
- Execution and observability interfaces

---

## The biggest differences among frameworks are usually not about “can it do it,” but about “how does it do it”

### A very important perspective: abstraction level

Many frameworks can:

- Connect tools
- Run workflows
- Support multiple Agents

But they differ in abstraction level:

- Some are closer to “building blocks at the lower level”
- Some are closer to “high-level role orchestration”
- Some are more focused on retrieval and data organization

### An analogy

You can think of different frameworks as different kinds of kitchens:

- Some give you pots, pans, and utensils, and you cook yourself
- Some give you semi-finished meal kits that you combine following instructions
- Some are especially good at certain types of dishes

So the difference between frameworks is often not “who is stronger,” but:

> **Who fits your current task and team habits better.**

---

## Let’s first look at a coarse-grained map

The following table is not an exact ranking, but a way to build intuition quickly:

| Framework direction | What it is good at | Common feeling |
|---|---|---|
| Graph/Workflow-oriented | Complex state flow, explicit control | Flexible but more engineering-heavy |
| Retrieval/Knowledge-oriented | Documents, indexing, RAG | Stronger data orientation |
| Role/Team-oriented | Multi-Agent role collaboration | Quick to start, but higher-level abstraction |
| General experimentation-oriented | Rapid demo building | Flexible, but you need to fill in the engineering layers yourself |

The most important purpose of this map is:

> Don’t ask “which is best” first. Ask “which category does my problem resemble more?”

---

## What work do frameworks usually save you from?

### State flow and node management

For example:

- Where the current task state is stored
- Where the next step goes
- How to roll back on errors

### Tool integration and message structure

For example:

- Tool registration
- Wrapping call results
- Error handling

### Execution and observation

For example:

- Trace
- Step records
- Visualization of intermediate states

So the most common value of a framework is not “the model becomes smarter,” but:

> **The system is organized more clearly.**

---

## Frameworks also bring trade-offs

### The higher the abstraction, the easier it is to lose low-level control

A framework saves you effort, but it also brings:

- The cost of learning the framework itself
- Framework constraints
- The need to understand its internal abstractions when debugging

### A very common problem

Many beginners do not fail because they cannot build an Agent. They fail because they:

- Have not clarified the task yet
- But have already learned many framework interfaces

In the end, what they learn is framework usage, not the essence of Agents.

So the right order is usually:

> First understand the system, then use a framework to speed things up.

---

## A minimal “framework-like” example

The example below is not from a real framework, but a very small example that has the flavor of framework abstraction.

```python
class MiniWorkflow:
    def __init__(self):
        self.steps = []

    def add_step(self, name, fn):
        self.steps.append((name, fn))

    def run(self, state):
        for name, fn in self.steps:
            state = fn(state)
            print(name, "->", state)
        return state

def retrieve(state):
    state["docs"] = ["refund policy"]
    return state

def answer(state):
    state["answer"] = f"Generate an answer based on {state['docs']}"
    return state

wf = MiniWorkflow()
wf.add_step("retrieve", retrieve)
wf.add_step("answer", answer)

wf.run({"query": "What is the refund policy?"})
```

Expected output:

```text
retrieve -> {'query': 'What is the refund policy?', 'docs': ['refund policy']}
answer -> {'query': 'What is the refund policy?', 'docs': ['refund policy'], 'answer': "Generate an answer based on ['refund policy']"}
```

### Why does this code feel “framework-like”?

Because it is already abstracting:

- step
- state
- workflow organization

Real frameworks are simply more complete and more sophisticated versions of this direction.

---

## When is it more suitable to avoid a framework?

If your system is:

- A small experiment
- A single Agent
- Few tools
- Very simple state flow

Then writing it yourself is not necessarily worse.

Many times:

- Hand-written code is easier to understand in terms of the essence
- A framework may instead add abstraction overhead

So do not treat “using a framework” as the only sign of maturity.

---

## A very practical framework-selection approach

First ask these questions:

1. Is my system highly complex?
2. Is the state flow obviously complex?
3. Is multi-Agent collaboration the core?
4. Is retrieval/document capability the main focus?
5. Does the team prefer low-level control or faster startup?

If you can answer these questions, framework comparison becomes much clearer.

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
problem_shape: workflow graph, retrieval app, role team, or experiment
framework_choice: what abstraction it adds and what control it hides
trace: state, node, tool call, message, or run id
failure_check: framework magic hides state, retries, or permissions
decision: choose framework only after the single-agent loop is clear
```

## Common pitfalls for beginners

### Learn the framework first, then the system

This is the easiest way to end up “able to call APIs, but unable to make architectural decisions.”

### Choose a framework just because it is popular

Popularity does not mean it fits your current project.

### Treat the framework as the capability itself

A framework is only an organizing approach, not a guarantee of system quality.

---

## Summary

The most important point in this section is not to memorize a list of framework names, but to understand:

> **The essence of an Agent framework is to abstract away high-frequency state flow, tool flow, and collaboration structure, helping you organize systems faster.**

Once you understand this, when you look at specific frameworks later, it will feel more like comparing “ways of organizing” rather than chasing trends.

---

## Exercises

1. Using your own project scenario, decide whether it is more suitable for a “graph/workflow-oriented” framework or a “role-collaboration-oriented” framework.
2. Think about this: why might hand-written code be better for projects with lower complexity?
3. Explain in your own words: what work does a framework really save you from?
4. If your team especially values controllability, what style of framework would you be more inclined to choose?

<details>
<summary>Reference answers and explanation</summary>

1. Choose graph/workflow orientation when the hard part is state, branching, retry, and observability. Choose role-collaboration orientation when the hard part is dividing work among specialized roles and reviewing their outputs.
2. Hand-written code can be better for low-complexity projects because it has fewer abstractions, fewer hidden defaults, and a shorter debugging path.
3. A framework saves repeated plumbing: message passing, state containers, tool binding, tracing, retries, memory hooks, and common workflow patterns. It does not save you from understanding the system design.
4. If controllability is the priority, prefer explicit workflow or graph-style frameworks, or even a small hand-written orchestration layer, because state transitions and failure paths are easier to inspect.

</details>
