---
title: "9.3.4 Tool Calling Strategies"
description: "Systematically understand Agent tool-layer calling strategies: when to call, which tool to call, how many times to call, and what to do after failures."
sidebar:
  order: 13
head:
  - tag: meta
    attrs:
      name: keywords
      content: "tool strategy, Agent, routing, retry, fallback, verification, tool policy"
---

# 9.3.4 Tool Calling Strategies

![Tool calling strategy routing map](/img/course/ch09-tool-strategy-routing-map-en.webp)

:::tip[Section Overview]
In the previous section, you learned how to safely connect tools behind the model.
This section takes one more step forward:

> **It is not enough to connect tools; the key is how to use them.**

What really creates differences in system quality is often not “whether tools exist,” but “when to call them, which one to call first, and what to do after failures.”
:::
## Learning Objectives

- Understand why tool calling strategy is one of the keys to Agent success or failure
- Distinguish between common patterns such as "do not call / single call / multi-step call / fallback strategy"
- Learn how to design basic routing, retry, and verification logic
- Understand a more complete example of a tool strategy

---

## Why does "having tools" not mean "knowing how to use tools"?

### A common misunderstanding

Many people’s first step in building an Agent is:

- connect a search tool
- connect a calculator
- connect a database

Then they assume the system will become stronger.

But in reality, you often see:

- tools are called randomly when they should not be
- tools are not called when they should be
- the wrong tool is called
- the system keeps calling tools 5 times in a row and never stops

So the real question is not:

> Does the system have tools?

but:

> Does the system have a "tool usage strategy"?

### A real-life analogy

Having a knife, pot, oven, and microwave in your kitchen does not mean you know how to cook.
What matters is:

- when to cut
- when to boil
- when to bake
- how to recover when something goes wrong

Tool calling strategy in an Agent works the same way.

---

## First, distinguish the common strategies

### Do not call a tool

Suitable for:

- common-sense explanations
- simple rewriting
- style conversion

For example:

> "Rewrite this paragraph to sound more formal."

This kind of task usually does not need an external tool.

### Single call

Suitable for:

- checking the weather
- calculating a mathematical expression
- looking up one knowledge base record

This is the simplest and most stable calling pattern.

### Multi-step calls

Suitable for:

- first check the order, then check refund rules, then give a conclusion
- first search for materials, then summarize, then generate output

At this point, the strategy is no longer just "which tool to call," but also "should I keep calling the next step?"

### Fallback and backup

If:

- the main tool fails
- the result is not trustworthy
- parameter validation does not pass

then the system must decide whether to:

- retry
- switch tools
- ask the user for more information
- admit it cannot complete the task

This is also an important part of tool strategy.

---

## What should be judged before calling a tool?

### Does this task really need a tool?

Not every problem is worth going through a tool chain.
Every tool call adds:

- latency
- cost
- failure paths

So the first step is often:

> **First decide whether a tool is needed.**

### If a tool is needed, which one should be chosen?

For example, if the user asks:

> "Can I still get a refund for this order?"

You may need to:

1. check order status
2. check refund policy

So tool selection is not always a "single-choice question"; sometimes it is a "combination question with an order."

### Are the parameters sufficient?

Sometimes you know which tool to call, but the parameters are still incomplete.

For example:

> "Help me check the weather"

The city name is missing.
In that case, the most reasonable strategy is not to guess randomly, but to:

> ask the user for clarification first.

---

## What should be judged after calling a tool?

### Is the result trustworthy?

A tool returned something, but that does not mean it can be used directly.

For example:

- the API timed out and returned an empty value
- search results are not very relevant
- the database returned no record

### Is another step needed?

Some tasks cannot be solved in one call.

For example:

- first query the knowledge base
- then do a calculation
- then summarize it in user-friendly language

So tool strategy is often essentially:

> call -> observe -> decide the next step

---

## A minimal but educational strategy example

The example below distinguishes three cases:

- do not call a tool
- call a single tool
- ask the user first when parameters are insufficient

```python
def route_query(query):
    normalized = query.lower()

    if "summarize" in normalized or "rewrite" in normalized:
        return {"action": "no_tool", "reason": "pure text task"}

    if "weather" in normalized:
        if "beijing" in normalized:
            return {"action": "tool", "tool": "weather", "arguments": {"city": "Beijing"}}
        return {"action": "ask_user", "question": "Which city's weather would you like to check?"}

    if "calculate" in normalized:
        expression = query[normalized.index("calculate") + len("calculate") :].strip()
        return {"action": "tool", "tool": "calculator", "arguments": {"expression": expression}}

    return {"action": "fallback", "reason": "no suitable strategy available"}

queries = [
    "Summarize this paragraph",
    "How's the weather in Beijing?",
    "Help me check the weather",
    "Calculate 12 * 7"
]

for q in queries:
    print(q, "->", route_query(q))
```

Expected output:

```text
Summarize this paragraph -> {'action': 'no_tool', 'reason': 'pure text task'}
How's the weather in Beijing? -> {'action': 'tool', 'tool': 'weather', 'arguments': {'city': 'Beijing'}}
Help me check the weather -> {'action': 'ask_user', 'question': "Which city's weather would you like to check?"}
Calculate 12 * 7 -> {'action': 'tool', 'tool': 'calculator', 'arguments': {'expression': '12 * 7'}}
```

Although this example is simple, it already shows the level of "strategy":

- not every input should be sent to a tool
- do not guess when parameters are missing
- have a fallback when you do not know what to do

---

## A more complete strategy loop

### Define a few tools

```python
import ast
import operator

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_calculate(expression):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError("unsupported_expression")

    return visit(ast.parse(expression, mode="eval"))


def get_weather(city):
    return {"city": city, "temperature": 22, "condition": "sunny"}

def calculate(expression):
    return {"result": safe_calculate(expression)}
```

### Scheduling + validation + execution

```python
def execute_strategy(query):
    decision = route_query(query)

    if decision["action"] == "no_tool":
        return {"type": "answer", "content": "This kind of task is better handled directly by the model generating text."}

    if decision["action"] == "ask_user":
        return {"type": "question", "content": decision["question"]}

    if decision["action"] == "tool":
        if decision["tool"] == "weather":
            result = get_weather(**decision["arguments"])
            return {"type": "tool_result", "content": result}
        if decision["tool"] == "calculator":
            result = calculate(**decision["arguments"])
            return {"type": "tool_result", "content": result}

    return {"type": "fallback", "content": "This request cannot be handled reliably right now."}

for q in ["How's the weather in Beijing?", "Help me check the weather", "Calculate 9 + 8"]:
    print(q, "->", execute_strategy(q))
```

Expected output:

```text
How's the weather in Beijing? -> {'type': 'tool_result', 'content': {'city': 'Beijing', 'temperature': 22, 'condition': 'sunny'}}
Help me check the weather -> {'type': 'question', 'content': "Which city's weather would you like to check?"}
Calculate 9 + 8 -> {'type': 'tool_result', 'content': {'result': 17}}
```

![Tool strategy routing and execution result map](/img/course/ch09-tool-strategy-routing-execution-result-map-en.webp)

What this code really teaches is:

> Tool calling strategy is not a single `if` statement, but a chain of "decision + routing + execution + fallback."

---

## Several common strategy patterns in real systems

### Router pattern

First determine which tool or subsystem the question belongs to.

Suitable for:

- many tools
- clear task boundaries

### Verify pattern

After calling a tool, do not trust the result immediately; check it again.

Suitable for:

- unstable external data
- tools with a high failure rate

### Retry / Fallback pattern

Retry first, then degrade, then fall back.

Suitable for:

- fluctuating external APIs
- unstable online services

### Plan-then-tool pattern

First make a plan, then decide the order of tool use.

Suitable for:

- multi-step tasks
- tasks with multiple tool dependencies

---

## When should you "call tools less"?

This is also an important strategic ability.

### Typical scenarios where fewer tool calls are better

- pure summarization
- pure rewriting
- style conversion
- enough existing context

### Why is calling fewer tools sometimes better?

Because each additional tool call adds:

- latency
- failure risk
- state management cost

So a mature system is not one that says "call whenever possible," but one that says:

> **Save calls when saving is the better choice.**

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
tool_contract: name, description, input schema, output schema
permission: what the tool is allowed to read or change
call_trace: arguments, result, error, retry or fallback
failure_check: wrong tool, bad arguments, unsafe action, or missing observation
safety_action: validate, confirm, sandbox, rate-limit, or rollback
```

## Common pitfalls for beginners

### Treating tool calling strategy as just "routing rules"

Routing is only one part of it.
A real strategy also includes:

- whether to call
- whether to ask follow-up questions
- whether to continue to the next step
- whether to fall back

### No next step after a call fails

No retry, no fallback, no clarification request — such systems are fragile in production.

### Always assuming the model should decide everything

In real engineering, many strategies should be explicitly constrained by the program framework, rather than being left entirely to the model.

---

## Summary

The most important idea in this section is not knowing "which tools can be called," but understanding:

> **Tool calling strategy determines whether an Agent calls tools at the right time, in the right order, and in the right way.**

This often affects system quality more than "adding a few more tools."

---

## Exercises

1. Add a `search_docs(keyword)` tool to the example in this section and extend the routing logic.
2. Add a branch for "if a tool execution error occurs, fall back to human confirmation."
3. Think about this: if the user asks, "Help me check the weather and calculate a clothing index," how should the strategy layer split this task?
4. Explain in your own words: why is tool calling strategy one of the dividing lines for Agent quality?

<details>
<summary>Reference implementation and walkthrough</summary>

1. `search_docs(keyword)` should add a routing branch for knowledge lookup and return evidence snippets, not just a free-form paragraph.
2. Tool execution errors should be classified first. Retry only safe transient errors; otherwise fall back to human confirmation or a controlled failure.
3. The weather-and-clothing task should split into weather lookup, interpretation of conditions, and a small calculation or rule-based recommendation.
4. Tool strategy is a quality boundary because it decides tool order, state passing, error recovery, and when to stop.

</details>
