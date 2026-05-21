---
title: "7.2.1 LLM Overview Roadmap: Capability, Cost, Product Fit"
sidebar_position: 0
description: "A compact LLM overview roadmap: development history, core concepts, industry landscape, and first API call workbench."
keywords: [LLM overview, large language model, model capability, LLM application, API call]
---

# 7.2.1 LLM Overview Roadmap: Capability, Cost, Product Fit

LLM overview is not a model-name list. It helps you decide what a large model can do, what it costs, and when prompting, RAG, Agent, or fine-tuning is a better route.

## Look at the Capability Stack First

![LLM overview chapter relationship diagram](/img/course/ch07-llm-overview-chapter-flow-en.webp)

![Large model capability stack and application ecosystem diagram](/img/course/ch07-llm-capability-stack-en.webp)

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

![LLM route decision run result map](/img/course/ch07-llm-route-decision-result-map-en.webp)

This is not a full architecture decision. It is the habit: choose the smallest route that solves the actual product need.

## Learn in This Order

| Order | Read | What to keep |
|---|---|---|
| 1 | [7.2.2 Development History](./01-development-history.md) | why scaling and instruction tuning mattered |
| 2 | [7.2.3 Core Concepts](./02-core-concepts.md) | context, tokens, temperature, latency, cost |
| 3 | [7.2.4 Industry Landscape](./03-industry-landscape.md) | model/provider selection notes |
| 4 | [7.2.5 LLM Call Workbench](./04-llm-call-workbench.md) | one request/response record |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
capability_stack: tokens, context, pretraining, instruction, alignment
cost_check: context length and output length affect cost/latency
product_fit: choose model behavior by task need, not hype
evaluation_loop: fixed cases, score, failure note
next_action: connect overview to prompt testing in 7.5
```

## Pass Check

You pass this roadmap when you can explain one model choice in terms of capability, context, cost, latency, data privacy, and route fit.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer explains how tokens, context, attention, prompts, and generation behavior connect in one request-response path.
2. The evidence should include at least one reproducible prompt or structured-output test, plus notes on why the output passed or failed.
3. A good self-check separates prompt design, RAG, fine-tuning, and alignment: use the lightest method that fixes the observed problem.

</details>
