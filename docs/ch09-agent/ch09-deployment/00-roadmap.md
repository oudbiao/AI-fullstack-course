---
title: "9.9.1 Deployment Roadmap: Runtime, Persistence, Recovery"
sidebar_position: 0
description: "A concise hands-on roadmap for Agent deployment and operations: expose an API, persist state, record traces, control cost, and recover from failures."
keywords: [Agent deployment guide, Agent operations, cost optimization, runtime, observability]
---

# 9.9.1 Deployment Roadmap: Runtime, Persistence, Recovery

Deploying an Agent means more than putting code on a server. You need model calls, tool services, queues, state storage, traces, permissions, cost limits, and rollback paths.

## See the Runtime Loop First

![Agent production runtime architecture diagram](/img/course/ch09-production-runtime-map-en.webp)

![Agent deployment and operations chapter learning flow diagram](/img/course/ch09-deployment-chapter-flow-en.webp)

![Agent deployment observability and recovery loop](/img/course/ch09-deployment-observability-loop-en.webp)

The production question is not “did it work once?” It is “can it keep working, fail safely, and recover?”

## Run a Deployment Readiness Check

This check highlights missing production basics.

```python
service = {
    "api_entry": True,
    "state_store": True,
    "trace_log": True,
    "cost_limit": True,
    "rollback": False,
}

missing = [name for name, ok in service.items() if not ok]

print("ready:", not missing)
print("missing:", missing)
```

Expected output:

```text
ready: False
missing: ['rollback']
```

If the system cannot roll back or recover, do not call it production-ready.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Deployment architecture | Draw frontend, backend, model service, tool service, storage |
| 2 | Runtime management | Handle sync, async, long-running tasks, queues, interruption |
| 3 | Persistence and recovery | Save task state, memory, traces, intermediate results |
| 4 | Cost optimization | Track model calls, tool calls, caching, batching, routing |
| 5 | Production practices | Add monitoring, alerts, canary release, rollback, permissions |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
runtime: queues, workers, state store, tool services, and model endpoint
persistence: checkpoints, event log, memory store, and recovery path
ops_signal: latency, cost, error rate, trace coverage, and saturation
failure_check: stuck run, duplicate action, partial failure, or runaway cost
recovery_action: resume, rollback, cancel, human handoff, or degrade gracefully
```

## Pass Check

You pass this chapter when a local Agent demo becomes a small service with API entry, state persistence, trace logs, error responses, cost records, and deployment instructions.
