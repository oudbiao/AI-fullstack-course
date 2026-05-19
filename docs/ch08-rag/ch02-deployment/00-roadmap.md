---
title: "8.2.1 Deployment Roadmap: Local Model, Service, Unified API"
sidebar_position: 0
description: "A concise hands-on roadmap for model deployment: choose where a model runs, expose it as a service, and keep the application calling one stable API contract."
keywords: [model deployment guide, local models, inference services, unified API]
---

# 8.2.1 Deployment Roadmap: Local Model, Service, Unified API

Deployment turns a model from a notebook experiment into a reusable capability. The application should call a stable interface, even when the model, provider, hardware, or cost policy changes.

## See the Serving Decision First

![Model deployment chapter learning flowchart](/img/course/ch08-deployment-chapter-flow-en.webp)

![Model serving selection decision map](/img/course/ch08-model-serving-decision-map-en.webp)

![Unified API provider gateway map](/img/course/ch08-unified-api-provider-gateway-map-en.webp)

Deployment choices balance quality, latency, cost, privacy, and operational complexity. The strongest model is not always the model you should call.

## Run a Model Route Check

Use this as a mental model before setting up real serving tools. It turns deployment into an explicit routing decision.

```python
request = {
    "privacy": "high",
    "latency_ms": 800,
    "quality_need": "medium",
    "budget": "low",
}

if request["privacy"] == "high":
    route = "local model or private endpoint"
elif request["quality_need"] == "high":
    route = "frontier cloud model"
else:
    route = "small hosted model"

print("route:", route)
print("contract:", "/v1/chat/completions")
print("watch:", "latency, cost, errors")
```

Expected output:

```text
route: local model or private endpoint
contract: /v1/chat/completions
watch: latency, cost, errors
```

The route can change, but the application contract should stay stable.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Local models | Load or call one local/private model and record limits |
| 2 | Inference servers | Expose model calls through a service endpoint |
| 3 | Unified API | Keep one application interface for multiple providers |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
runtime_choice: local model, inference server, or unified API
request_contract: endpoint, payload, output format, and error shape
latency_or_cost: one measured or estimated number
failure_check: timeout, memory pressure, model mismatch, or version drift
rollback_plan: fallback model, retry policy, or traffic switch
```

## Pass Check

You pass this chapter when you can explain where the model runs, how the app calls it, what can fail, and what metrics you watch: latency, cost, errors, rate limits, and fallback behavior.

The exit mini project is a small model gateway note or script that routes one request to a chosen model endpoint and records the decision reason.
