---
title: "9.6.3 LangChain / LangGraph"
sidebar_position: 31
description: "From chain-style component composition to explicit state graphs, truly understand what LangChain and LangGraph are each abstracting for you."
keywords: [LangChain, LangGraph, chain, graph, stateful workflow, agent framework]
---

# 9.6.3 LangChain / LangGraph

:::tip Section Focus
When many people first encounter Agent frameworks, they tend to mix up LangChain and LangGraph.
But from an engineering perspective, the problems they solve are not exactly the same:

- LangChain is more like “stringing common components together”
- LangGraph is more like “drawing complex state flow as a graph”

Understanding the difference between these two matters much more than memorizing APIs.
:::

## Learning Objectives

- Understand which scenarios LangChain is better suited for
- Understand why LangGraph is better for complex state flows
- See the real difference between “chain abstraction” and “graph abstraction”
- Learn how to judge when you should upgrade from a chain-style workflow to a graph-style workflow

---

## Why did LangChain come first, and LangGraph come later?

### Early requirements are usually linear

Many LLM applications start out like this:

1. Receive input
2. Rewrite the query
3. Retrieve documents
4. Call the model to answer

This kind of workflow is very much like a straight line:

> The output of one step is fed into the next step.

In this situation, a “chain” is a very natural abstraction.

### But systems quickly become more complex

Once you start needing things like:

- What if retrieval returns nothing?
- Should we retry if the answer is not trustworthy?
- Should we call a tool first, then go back to QA?
- Should some cases require human confirmation?

The workflow is no longer a straight line. It becomes:

- branched
- stateful
- cyclical

At that point, what you need is not just “stringing steps together,” but:

> **Explicitly representing state and edges.**

That is the core reason LangGraph matters.

---

## First understand LangChain: what exactly is it abstracting?

### What it is best at: component pipelines

LangChain’s typical strength is connecting things like:

- prompt templates
- model calls
- output parsing
- retrieval modules
- tool modules

You can think of it as:

> A framework that is biased toward component orchestration.

It is like turning “prompts, models, retrievers, parsers” into building blocks that are easier to assemble.

### A minimal chain-style example

The example below does not use real LangChain, but it already has the LangChain style.

```python
class SimpleChain:
    def __init__(self, steps):
        self.steps = steps

    def run(self, value):
        for step in self.steps:
            value = step(value)
        return value

def normalize_query(text):
    return text.strip().lower()

def retrieve_docs(query):
    if "refund" in query:
        return {"query": query, "docs": ["You can request a refund within 7 days after purchasing the course."]}
    return {"query": query, "docs": []}

def format_answer(payload):
    if payload["docs"]:
        return f"According to the materials: {payload['docs'][0]}"
    return "No relevant materials were found."

chain = SimpleChain([
    normalize_query,
    retrieve_docs,
    format_answer
])

print(chain.run("  What is the refund policy? "))
```

Expected output:

```text
According to the materials: You can request a refund within 7 days after purchasing the course.
```

### What should you remember most from this example?

It expresses a very clear idea:

> **Each step does one thing, and the whole system completes the task by chaining steps together.**

This is exactly why LangChain is so easy for beginners to pick up.

---

## When is LangChain especially useful?

### Good fits for these situations

- The workflow is basically linear
- There are not many branches
- The main goal is to combine several modules
- You want to quickly build a prototype

Typical examples:

- FAQ retrieval QA
- Text extraction
- Retrieval-augmented generation
- Single-tool-enhanced QA

### Its advantages

- Quick to get started
- Rich component ecosystem
- Great for assembling “small modules” first

### Where does it start to struggle?

When you begin to need:

- long state chains
- multiple branching decisions
- node jumps back to earlier steps
- explicit intermediate state management

At that point, chain thinking starts to feel increasingly forced.

---

## Then understand LangGraph: why is it more like a “state machine”?

### LangGraph is not just about nodes, but about state

The most important perspective in LangGraph is not:

- which component to call next

but:

- what the current state is
- which edge this state should take
- how the state changes after a node runs

You can first think of it as:

> **A stateful workflow graph.**

### A minimal graph-style example

```python
def plan_node(state):
    if "refund" in state["query"]:
        state["next"] = "retrieve"
    else:
        state["next"] = "fallback"
    return state

def retrieve_node(state):
    state["docs"] = ["You can request a refund within 7 days after purchasing the course."]
    state["next"] = "answer"
    return state

def answer_node(state):
    state["answer"] = f"According to the materials: {state['docs'][0]}"
    state["next"] = None
    return state

def fallback_node(state):
    state["answer"] = "No matching workflow was found."
    state["next"] = None
    return state

nodes = {
    "plan": plan_node,
    "retrieve": retrieve_node,
    "answer": answer_node,
    "fallback": fallback_node
}

state = {"query": "What is the refund policy", "next": "plan"}

while state["next"] is not None:
    current = state["next"]
    state = nodes[current](state)
    print(current, "->", state)
```

Expected output:

```text
plan -> {'query': 'What is the refund policy', 'next': 'retrieve'}
retrieve -> {'query': 'What is the refund policy', 'next': 'answer', 'docs': ['You can request a refund within 7 days after purchasing the course.']}
answer -> {'query': 'What is the refund policy', 'next': None, 'docs': ['You can request a refund within 7 days after purchasing the course.'], 'answer': 'According to the materials: You can request a refund within 7 days after purchasing the course.'}
```

### What is the biggest difference between this and the chain example?

In a chain system:

- the next step is usually fixed

In a graph system:

- the next step is determined by the current state

That is the most fundamental advantage of graph workflows.

![LangGraph state machine and conditional edge graph](/img/course/ch09-langgraph-state-machine-map-en.webp)

:::tip Reading the Diagram
This diagram can help you shift from “chain steps” to a “state machine”: nodes handle the state, conditional edges decide the next stop, and checkpoints let the system recover and be debugged in complex workflows.
:::

---

## When should you switch from LangChain thinking to LangGraph thinking?

### A very practical rule of thumb

If you draw the workflow and find that it is no longer a straight line, but instead has:

- obvious branches
- failure fallbacks
- loops
- “decide the next step based on intermediate results”

then it is usually time to consider a graph-style abstraction.

### A very obvious red flag

If your code starts to look like this:

```text
if ...
    if ...
        if ...
            while ...
```

and all of those conditions are centered around intermediate state, that usually means:

> You are no longer just building a “chain application”; you are building a state graph system.

---

## Why do many teams mention LangChain and LangGraph together?

Because real systems are often not an “either/or” choice.

### A very common combination

- LangChain-style code handles:
  - prompt
  - retriever
  - parser

- LangGraph-style code controls:
  - state transitions
  - branching
  - retries
  - human confirmation nodes

So in many cases, they are more like:

> Component layer + workflow layer.

rather than two completely opposing camps.

---

## A practical recommendation for real projects

### If what you are building now is:

- single-turn FAQ
- simple RAG
- a few fixed steps

then chain thinking is usually enough.

### If what you are building now is:

- multi-step Agent
- tool loops
- human confirmation nodes
- strong dependence on intermediate state

then graph thinking will be more robust.

### Don’t jump to a graph just for “advanced” vibes

This is also an important judgment call.
Graph abstractions are more powerful, but they also bring:

- higher learning cost
- more structural design work

When complexity is not high, a chain-based approach can actually be clearer.

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

## Common beginner mistakes

### Learning a bunch of framework APIs before understanding task structure

This means you end up learning only the “framework syntax,” not system design.

### Forcing a graph-shaped problem into a chain

The system keeps growing more and more `if/else`, but you still do not want to switch to a graph abstraction.

### Starting with a graph framework right away

Even though the requirement is simple, you make the system heavy from the beginning.

---

## Summary

The most important thing in this section is not memorizing framework names, but building this judgment:

> **LangChain is more like stringing common components together, while LangGraph is more like explicitly drawing complex state flows.**

Once you start judging them by “task structure” rather than “framework popularity,” your architectural choices will become much more stable.

---

## Exercises

1. Draw one of your own Agent systems and decide whether it looks more like a chain or a graph.
2. Add logic to the chain example: “If no documents are found, rewrite the query and search again,” and see whether it starts to get messy.
3. Explain in your own words: why is graph abstraction more suitable than chain abstraction for systems with stateful loops?
4. Think about this: in what situations is a chain-based approach actually more appropriate than a graph-based one?

<details>
<summary>Reference answers and explanation</summary>

1. If every step runs once in a fixed order, it is chain-like. If the system can branch, loop, retry, pause, or revisit a previous state, it is graph-like.
2. Adding “no documents -> rewrite query -> search again” often makes a chain messy because the code now needs conditional routing, loop limits, and state history. That is the point where a graph abstraction starts to help.
3. Graph abstraction fits stateful loops because nodes and edges make transitions explicit: what state enters, what decision is made, where the flow goes next, and when it stops.
4. A chain is better when the task is linear, short, stable, and easy to inspect, such as formatting input, calling one retriever, then generating a short answer.

</details>
