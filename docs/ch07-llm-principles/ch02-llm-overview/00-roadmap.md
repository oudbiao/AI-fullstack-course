---
title: "7.2.1 LLM Overview Roadmap: Capability, Cost, Product Fit"
sidebar_position: 0
description: "A compact LLM overview roadmap: development history, core concepts, industry landscape, and first API call workbench."
keywords: [LLM overview, large language model, model capability, LLM application, API call]
---

# 7.2.1 LLM Overview Roadmap: Capability, Cost, Product Fit

LLM overview is not a model-name list. It helps you decide what a large model can do, what it costs, and when prompting, RAG, Agent, or fine-tuning is a better route.

## Look at the Capability Stack First

![LLM overview chapter relationship diagram](/img/course/ch07-llm-overview-chapter-flow-en.png)

![Large model capability stack and application ecosystem diagram](/img/course/ch07-llm-capability-stack-en.png)

| Route | Use when... |
|---|---|
| prompt | the model already knows enough and task is simple |
| RAG | private or changing knowledge must be cited |
| Agent | the model must use tools or take steps |
| fine-tuning | behavior/style/format needs repeated adaptation |

## Run One Route Decision

```python
request = {
    "needs_private_docs": True,
    "needs_tool_action": False,
    "needs_repeated_style": False,
}

if request["needs_tool_action"]:
    route = "Agent"
elif request["needs_private_docs"]:
    route = "RAG"
elif request["needs_repeated_style"]:
    route = "fine-tuning"
else:
    route = "prompt"

print("recommended_route:", route)
```

Expected output:

```text
recommended_route: RAG
```

This is not a full architecture decision. It is the habit: choose the smallest route that solves the actual product need.

## Learn in This Order

| Order | Read | What to keep |
|---|---|---|
| 1 | [7.2.2 Development History](./01-development-history.md) | why scaling and instruction tuning mattered |
| 2 | [7.2.3 Core Concepts](./02-core-concepts.md) | context, tokens, temperature, latency, cost |
| 3 | [7.2.4 Industry Landscape](./03-industry-landscape.md) | model/provider selection notes |
| 4 | [7.2.5 LLM Call Workbench](./04-llm-call-workbench.md) | one request/response record |

## Pass Check

You pass this roadmap when you can explain one model choice in terms of capability, context, cost, latency, data privacy, and route fit.
