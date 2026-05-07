---
title: "E.F AI Product Design Thinking"
sidebar_position: 6
description: "Starting from user problems, task success criteria, cost, and risk constraints, understand why AI product design is first and foremost a trade-off problem."
keywords: [AI product design, product thinking, evaluation, cost, UX, product strategy]
---

# E.F AI Product Design Thinking

![AI Product Decision Matrix](/img/course/elective-ai-product-decision-matrix-en.png)

![AI Product Experiment and Metrics Loop](/img/course/elective-ai-product-experiment-metrics-loop-en.png)

:::info Hands-on checkpoint
If you want to see how this module can become a portfolio artifact, run the [Elective Hands-on Workshop](./hands-on-elective-workshop) first and inspect the Module F product canvas.
:::

:::tip Reading guide
Product decisions should not stop at “this AI feature is cool.” When reading the diagram, look at the user problem, hypothesis, MVP, success metrics, risk boundaries, experiment feedback, and iteration together to judge whether the feature is truly worth continuing.
:::

:::tip Section focus
One of the easiest mistakes in AI product work is:

- thinking first about what the model can do

instead of:

- what problem the user actually needs solved

This often leads to projects that are technically impressive but product-wise hollow.

So this lesson aims to answer:

> **How do we turn an AI feature from a “cool demo idea” into “verifiable product value”?**
:::

## Learning objectives

- Understand the difference between AI product design and pure technical implementation
- Learn to evaluate a solution from four dimensions: user problem, success criteria, cost, and risk
- Build a minimal product prioritization mindset through a runnable example
- Develop a product mindset of “judge first, build later”

---

## What is the core question in AI product design?

### Not “Can we build it?”, but “Is it worth building?”

Many AI features are technically possible,
but from a product perspective, what matters more is:

- Does the user really feel the pain?
- Is the cost acceptable?
- Is the risk manageable?
- Is the experience sustainable?

### An analogy

Technical implementation is like being able to cook many dishes.
Product design is deciding:

- Which dish to make today
- Who it is for
- Whether the cost can be recovered

### So AI product design is not about “weakening technology”

It is about making technology serve:

- user problems
- business constraints
- risk boundaries

---

## Four dimensions commonly used when designing AI products

### User value

Does it really solve a frequent and clear user problem?

### Cost

Includes:

- model usage cost
- engineering maintenance cost
- human review cost

### Risk

Includes:

- incorrect-answer risk
- compliance risk
- brand risk

### Experience quality

Includes:

- waiting time
- explainability
- output stability

---

## First, let’s run a simple product prioritization example

The following example uses a very simple approach
to help you rank several AI product directions in the first round.

```python
ideas = [
    {"name": "AI Tutor", "value": 9, "cost": 6, "risk": 4, "ux": 8},
    {"name": "AI Customer Service", "value": 8, "cost": 5, "risk": 5, "ux": 7},
    {"name": "AI Code Review", "value": 7, "cost": 4, "risk": 6, "ux": 6},
]


def score(item):
    return (
        item["value"] * 0.45
        + (10 - item["cost"]) * 0.2
        + (10 - item["risk"]) * 0.2
        + item["ux"] * 0.15
    )


ranked = sorted(
    [{**item, "score": round(score(item), 2)} for item in ideas],
    key=lambda x: x["score"],
    reverse=True,
)

for item in ranked:
    print(item)
```

### What should you take away from this example?

Product judgment is usually not one-dimensional.
An idea being “high value” does not automatically mean it is the most worth doing.

Because you also need to consider:

- cost
- risk
- experience

### Why is this more useful than “I think this direction is cool”?

Because it forces you to make your decision criteria explicit:

- What exactly are you using to make the decision?

instead of relying on intuition.

---

## Three common traps in AI product design

### Mistake 1: Starting from model capability instead of user problems

For example:

- “I have a large model, so I want to find a scenario to plug it into”

This usually leads to products that are neither painful nor truly useful.

### Mistake 2: Looking only at features and ignoring risk and cost

Some features have dazzling demos,
but if:

- each call is too expensive
- the risk is too high
- the human fallback is too heavy

then it is hard to turn them into real products.

### Mistake 3: Attributing all experience issues to the frontend

The experience of an AI product depends heavily on:

- output stability
- waiting time
- explainability

This is not just a UI problem, but a product design problem as a whole.

---

## A practical product judgment sequence

### First ask where the user is getting stuck

Don’t ask first what the model can write,
ask first which step is most painful for the user.

### Then ask whether AI is really better than rules or a traditional workflow

Not every problem should use AI.
For some problems:

- rules are enough
- database retrieval is enough
- a form-based workflow is enough

### Only then ask “Which model should we use?”

Model selection is usually not the first-layer problem,
but an implementation question after the solution direction is decided.

---

## Two very important outputs in product design

### Success criteria

For example:

- higher task completion rate
- lower average waiting time
- reduced manual workload

### Failure boundaries

For example:

- which scenarios must be handed off to humans
- which outputs must not be released automatically
- which features should not be launched yet

This makes the product more stable and more realistic.

---

## Summary

The most important thing in this lesson is to build a product perspective:

> **AI product design is first about problem definition and trade-off judgment, and only then about model selection and feature implementation.**

Once this layer of judgment is solid, you are less likely to fall into the trap of building something that looks flashy but offers vague value.

---

## Exercises

1. Use the four dimensions in the example to score one of your own AI ideas.
2. Think about why a seemingly cool AI feature may not be worth prioritizing.
3. If a feature has very high user value but also high risk, how would you handle it?
4. How would you explain to your team the idea of “defining success criteria first, then building the feature”?
