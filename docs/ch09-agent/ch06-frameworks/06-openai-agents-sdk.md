---
title: "9.6.7 OpenAI Agents SDK【Elective】"
sidebar_position: 35
description: "Starting from high-level abstractions such as Agent, Tool, and Runner, understand why the OpenAI Agents SDK feels more like a unified Agent programming model."
keywords: [OpenAI Agents SDK, agent runtime, tools, runner, sdk, agent abstraction]
---

# 9.6.7 OpenAI Agents SDK【Elective】

:::tip Section Overview
Many frameworks are helping you organize:

- graphs
- chains
- roles

But a high-level SDK like the OpenAI Agents SDK is more like saying:

> **We unify Agent, Tool, and the runtime into a more standardized development interface.**

Its focus is not necessarily on being “the most flexible,” but on providing a “more unified Agent programming experience.”
:::

## Learning Objectives

- Understand the core objects that this kind of Agents SDK tries to abstract
- Understand why “Runner / Runtime” is often the key value of this kind of SDK
- Read a minimal example of a high-level abstraction
- Build judgment about when this kind of SDK is suitable, and when it may not be

---

## Why does a layer like “Agents SDK” appear?

### Because writing an Agent by hand quickly leads to a lot of repeated boilerplate

A reasonably complete Agent system usually involves:

- tool registration
- parameter validation
- execution loop
- result wrapping
- trace
- state progression

If every project implements this by hand, you will quickly run into:

- inconsistent structure
- poor maintainability
- an unaligned team style

### What does the SDK actually want to do?

It is not there to implement your product logic for you, but to standardize:

- how the Agent object is represented
- how a Tool is attached
- how a single execution process runs

You can first remember this sentence:

> **The value of an SDK is not “being more powerful,” but “being more unified.”**

---

## Several key abstraction objects

### Agent

An intelligent unit with a goal and a set of tools.

### Tool

An external capability that the Agent can call, such as:

- search
- computation
- file access

### Runner / Runtime

This part is especially important.
It is usually responsible for:

- actually running the agent
- managing the execution process
- collecting results

In many cases, the biggest engineering value of this kind of SDK is exactly this:

> **It standardizes “how to run an Agent.”**

---

## A minimal high-level abstraction example

Below, we use pure Python to simulate this SDK style.

```python
class Tool:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

class Agent:
    def __init__(self, name, tools):
        self.name = name
        self.tools = {tool.name: tool for tool in tools}

class Runner:
    def run(self, agent, tool_name, **kwargs):
        if tool_name not in agent.tools:
            return {"error": "unknown_tool"}
        result = agent.tools[tool_name].fn(**kwargs)
        return {"agent": agent.name, "tool": tool_name, "result": result}

def get_weather(city):
    return f"{city} is sunny and 22 degrees right now"

weather_tool = Tool("get_weather", get_weather)
assistant = Agent("weather_assistant", [weather_tool])
runner = Runner()

print(runner.run(assistant, "get_weather", city="Beijing"))
```

Expected output:

```text
{'agent': 'weather_assistant', 'tool': 'get_weather', 'result': 'Beijing is sunny and 22 degrees right now'}
```

![OpenAI Agents SDK: Agent, Tool, Runner Split the Work](/img/course/ch09-openai-agents-sdk-runner-flow-en.webp)

### Why does this code feel very “SDK-like”?

Because it clearly separates three things:

- the Agent itself
- the Tool itself
- the execution layer Runner

This is exactly the structure many high-level SDKs want to standardize.

---

## What does this abstraction actually save you?

### A unified way to connect tools

You do not need to redefine for every project:

- how tools are attached
- how tools are invoked

### A unified execution entry point

As systems become more complex, “who runs the Agent” becomes a very important question.
Runner / Runtime makes this more standardized.

### Easier to form a consistent team style

Because:

- how the Agent is defined
- how the Tool is attached
- how results are returned

these parts will not be written randomly each time.

---

## Why are Runner / Runtime especially important?

### Because an Agent is not a normal function

An Agent is not just:

- input -> output

It may also include:

- tool selection
- execution process
- intermediate state
- error returns

So “how to run it” is itself an independent layer.

### An intuitive analogy

You can think of Runner as:

> the execution scheduler for the Agent.

The Agent is the participant, and the Runner is the one responsible for actually running it and managing the process.

---

## When does this kind of high-level SDK feel especially convenient?

### When what you want is a unified development experience

For example:

- multiple Agent projects want to use the same structure
- the team wants to write less repeated runtime logic
- you want a more unified expression for tools and agents

### Especially suitable for

- small to medium Agent applications
- the stage between prototype and product
- team projects that need a consistent runtime experience

In these scenarios, high-level abstractions often save a lot of effort.

---

## Its limitations must also be understood clearly

### High-level abstractions mean more constraints

What you get is:

- consistency
- clarity
- less boilerplate

What you may lose is:

- very fine-grained low-level control

### If your system is very special

For example:

- you already have a very complex state graph
- you have highly customized execution strategies

In that case, a high-level SDK may not be the most comfortable way to express it.

So the key question is not “Is it powerful?”, but:

> **Does its abstraction fit your system?**

---

## How is it different from other frameworks?

### Compared with LangGraph

LangGraph is more focused on:

- graphs
- state flows
- conditional edges

Agents SDK is more focused on:

- Agent
- Tool
- Runner

### Compared with CrewAI

CrewAI is more focused on:

- team roles and collaboration expression

Agents SDK is more focused on:

- a unified Agent execution model

So it is not directly competing with all frameworks on the same layer, but is more like:

> a high-level development interface style.

---

## Common mistakes beginners make

### Only looking at the SDK name and not its abstraction boundaries

The result is:

- as you use it, it starts to feel “not smooth”

### Thinking “high-level abstraction = more advanced”

That is not true.
High-level only means less boilerplate; it does not always mean more suitable.

### Memorizing SDK APIs before understanding the Agent itself

This makes it easy to write calls, but hard to make architectural judgments.

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

## Summary

The most important thing in this section is not memorizing class names, but understanding this:

> **The value of frameworks like the OpenAI Agents SDK lies in unifying Agent, Tool, and the execution process into a more stable programming model.**

When what you need is a consistent Agent development experience, it can be very helpful;
when you need extremely fine-grained low-level control, it may not be your first choice.

---

## Exercises

1. Explain in your own words: why is Runner / Runtime often the key value of this kind of SDK?
2. Think about it: what is the difference between this kind of high-level SDK and CrewAI’s “team collaboration abstraction”?
3. If your system already has a complex state machine, would you still prioritize this kind of high-level SDK? Why?
4. Explain in your own words: what kind of high-frequency boilerplate work does the SDK actually save you from?

<details>
<summary>Reference answers and explanation</summary>

1. Runner / Runtime is valuable because production Agents need more than a prompt: they need tool execution, state movement, handoffs, error handling, tracing, and a consistent way to run the loop.
2. A high-level SDK usually focuses on making the Agent runtime easier to build and observe. CrewAI focuses more on modeling a team of roles and tasks. The right choice depends on what you want to make explicit.
3. If you already have a complex state machine, do not automatically replace it. First ask whether the SDK can integrate with your existing transitions, tracing, and failure policy without hiding important control.
4. The SDK can save boilerplate around tool registration, call execution, handoff wiring, structured outputs, tracing, and common runtime concerns. It does not remove the need for task design or evaluation.

</details>
