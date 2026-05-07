---
title: "E.F AI Product Design Thinking"
sidebar_position: 6
description: "A concise hands-on guide to AI product judgment: define the user problem, score value, cost, risk, and UX before building."
keywords: [AI product design, product thinking, evaluation, cost, UX, product strategy]
---

# E.F AI Product Design Thinking

AI product design starts with the user problem, not the model capability. A feature is worth building only when value, cost, risk, and user experience can be explained.

## See the Decision Loop First

![AI Product Decision Matrix](/img/course/elective-ai-product-decision-matrix-en.png)

![AI Product Experiment and Metrics Loop](/img/course/elective-ai-product-experiment-metrics-loop-en.png)

The first product habit is to make trade-offs explicit before implementation starts.

## Run a Tiny Prioritization Score

```python
ideas = [
    {"name": "AI Tutor", "value": 9, "cost": 6, "risk": 4, "ux": 8},
    {"name": "AI Customer Service", "value": 8, "cost": 5, "risk": 5, "ux": 7},
    {"name": "AI Code Review", "value": 7, "cost": 4, "risk": 6, "ux": 6},
]

def score(item):
    return round(item["value"] * 0.45 + (10 - item["cost"]) * 0.2 + (10 - item["risk"]) * 0.2 + item["ux"] * 0.15, 2)

ranked = sorted(({**item, "score": score(item)} for item in ideas), key=lambda item: item["score"], reverse=True)

for item in ranked:
    print(item["name"], item["score"])
```

Expected output:

```text
AI Tutor 7.25
AI Customer Service 6.65
AI Code Review 6.05
```

The numbers are not final truth. They force you to say what you are optimizing for.

## Product Checklist

| Question | Good Answer |
|---|---|
| Who is stuck? | A specific user group and task |
| What improves? | Completion rate, time saved, quality, or cost |
| What can go wrong? | Risk boundary and human fallback |
| What proves progress? | A metric or user test result |

## Pass Check

You pass this elective when you can score one AI feature idea, explain the trade-off, define a success metric, and name one condition where the feature should not launch.
