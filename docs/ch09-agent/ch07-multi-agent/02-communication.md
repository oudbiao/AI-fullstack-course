---
title: "9.7.3 Communication Between Agents"
sidebar_position: 39
description: "Build a systematic understanding of how multiple Agents communicate, from message format, synchronous vs. asynchronous communication, and shared state to failure retries."
keywords: [multi-agent communication, message passing, event bus, shared state, async, protocol]
---

# 9.7.3 Communication Between Agents

:::tip Section Overview
If the previous section answered “How should these Agents split up the work?”, then this section answers:

> **After the work is divided, how do they actually pass information back and forth?**

In many multi-Agent systems, the final failure is not because each individual Agent is not smart enough, but because the communication design is too weak.
:::

## Learning Objectives

- Understand why communication is a key factor in whether a multi-Agent system succeeds or fails
- Distinguish between three common communication patterns: message passing, shared state, and event bus
- Read a minimal event bus example
- Understand the engineering differences between synchronous and asynchronous communication

---

## Why Does Communication Become the Core Problem in Multi-Agent Systems?

### The Biggest Risk in Multi-Agent Systems Is Not “Not Doing the Work,” but “Not Staying Aligned”

Even if each Agent is strong on its own, the system can still fail because of poor communication design:

- Repeated work
- Lost messages
- Inconsistent understanding of information
- Continuing to discuss a task after it has already been completed

### A Very Intuitive Analogy

A multi-Agent system is a lot like a small team working together:

- Division of labor is only the first step
- What often determines efficiency is the communication mechanism: meetings, handoffs, synchronization, and feedback

That is why communication is not an “extra module” — it is a core structure.

---

## Three of the Most Common Communication Patterns

### Direct Message Passing

One Agent explicitly sends a message to another Agent.

Pros:

- Simple
- Clear
- Easy to trace

Cons:

- The coupling between Agents is relatively strong

### Shared State / Blackboard

All Agents write to and read from one shared workspace.

Pros:

- No need for explicit point-to-point messaging every time
- Very suitable for multiple parties collaboratively observing the same task state

Cons:

- Easier to get messy
- Harder to control permissions and conflicts

### Event Bus

Agents do not necessarily know each other directly; instead, they publish messages to a bus, and subscribers receive them.

Pros:

- More decoupled
- Better for complex systems

Cons:

- More difficult to debug

---

## Start with the Simplest Point-to-Point Message Passing

### A Minimal Example

```python
message = {
    "from": "planner",
    "to": "worker",
    "type": "task_assignment",
    "content": "Please summarize the key conditions of the refund policy"
}

print(message)
```

Expected output:

```text
{'from': 'planner', 'to': 'worker', 'type': 'task_assignment', 'content': 'Please summarize the key conditions of the refund policy'}
```

### Why Is This Already Important?

Because it makes the key elements of communication explicit:

- Who sent it
- Who it was sent to
- Message type
- Message content

This is much more robust than “just passing some natural language.”

---

## Why Should Message Formats Be Standardized?

### A Bad Message Format

```python
bad_message = "Help me do this task"
print(bad_message)
```

Expected output:

```text
Help me do this task
```

The problem is:

- You do not know who sent it
- You do not know the task type
- You do not know the context
- You do not know what to do next

### A More Reliable Message Structure

```python
good_message = {
    "from": "planner",
    "to": "researcher",
    "type": "search_request",
    "task_id": "task_001",
    "payload": {
        "query": "refund policy"
    }
}

print(good_message)
```

Expected output:

```text
{'from': 'planner', 'to': 'researcher', 'type': 'search_request', 'task_id': 'task_001', 'payload': {'query': 'refund policy'}}
```

This is much closer to a message that can enter a system pipeline.

![Agent communication contract diagram](/img/course/ch09-multi-agent-communication-contract-map-en.webp)

:::tip Reading the Diagram
Do not send only one sentence of natural language in multi-Agent communication. In the diagram, every message should include sender, receiver, type, task_id, payload, and status, so the system can trace, retry, and assign responsibility.
:::

---

## A Minimal Event Bus Example

### Runnable Code

```python
from collections import defaultdict

class EventBus:
    def __init__(self):
        self.handlers = defaultdict(list)

    def subscribe(self, event_type, handler):
        self.handlers[event_type].append(handler)

    def publish(self, event_type, payload):
        for handler in self.handlers[event_type]:
            handler(payload)

def planner_handler(payload):
    print("[planner] received result:", payload)

def worker_handler(payload):
    print("[worker] received task:", payload)
    result = {
        "task_id": payload["task_id"],
        "summary": f"Finished retrieving information about {payload['query']}"
    }
    bus.publish("task_done", result)

bus = EventBus()
bus.subscribe("task_assignment", worker_handler)
bus.subscribe("task_done", planner_handler)

bus.publish("task_assignment", {
    "task_id": "task_001",
    "query": "refund policy"
})
```

Expected output:

```text
[worker] received task: {'task_id': 'task_001', 'query': 'refund policy'}
[planner] received result: {'task_id': 'task_001', 'summary': 'Finished retrieving information about refund policy'}
```

### What Does This Code Actually Teach?

It teaches you:

- Communication does not have to be point-to-point coupled
- You can decouple components through event types
- Completion messages and result messages can use the same underlying infrastructure

This is already very close to the communication backbone of a real system.

---

## Shared State: When Is It More Suitable?

### A Very Typical Scenario

If multiple Agents are working around the same task, such as:

- `planner` writing the plan
- `retriever` collecting materials
- `writer` generating a draft
- `reviewer` writing review comments

Then much of the information can be placed in a shared workspace.

### A Minimal Example

```python
shared_state = {
    "goal": "Complete the refund policy summary",
    "plan": [],
    "evidence": [],
    "draft": None,
    "review": None
}

# planner
shared_state["plan"] = ["check policy", "organize key points", "output summary"]

# retriever
shared_state["evidence"].append("Refunds are available within 7 days after purchase if study progress is below 20%")

# writer
shared_state["draft"] = "Refund conditions include time limits and study progress limits."

print(shared_state)
```

Expected output:

```text
{'goal': 'Complete the refund policy summary', 'plan': ['check policy', 'organize key points', 'output summary'], 'evidence': ['Refunds are available within 7 days after purchase if study progress is below 20%'], 'draft': 'Refund conditions include time limits and study progress limits.', 'review': None}
```

### Pros and Cons of This Approach

Pros:

- Everyone can see the same blackboard
- The state is more centralized

Cons:

- You need to control who can write what
- Conflicts are easy to create

---

## How Should We Understand Synchronous and Asynchronous Communication?

### Synchronous Communication

After an Agent sends a request, it must wait for the other side to reply before it can continue.

Pros:

- Simple
- Easy to understand

Cons:

- Can easily block progress

### Asynchronous Communication

After sending a message, the Agent continues doing other work first, and handles the result later when the other side finishes.

Pros:

- More flexible
- Better for complex systems and high concurrency

Cons:

- More complex state management

### A Very Practical Engineering Rule of Thumb

If your task chain is short and the process is clear, start with synchronous communication.
If the task is long and waiting time is unstable, then consider asynchronous communication.

---

## The Most Common Failure Points in Agent-to-Agent Communication

### Inconsistent Message Formats

Today it is called `task_id`, tomorrow `id`, and the day after `job_id` — the system will quickly become messy.

### A Message Was Sent, but Nobody Handles It

This is a very common issue in event systems:

- It was published
- But there are no subscribers

### Multiple Agents Interpret the Same Message Differently

For example:

- One Agent thinks it is a “retrieval request”
- Another Agent thinks it is a “summary request”

This will cause the system to drift off course.

### No Timeouts or Retries

If one Agent gets stuck, the whole system may keep waiting forever.

---

## How Can Real Systems Make Communication More Reliable?

### Unify the Message Protocol

At minimum, standardize:

- `from`
- `to`
- `type`
- `task_id`
- `payload`

### Unify State Tracking

Each task should ideally have a unique ID to make it easier to:

- Trace the full chain
- Replay
- Debug

### Unify Timeout and Failure Policies

For example:

- Automatic fallback after timeout
- Escalate to a human on failure
- Stop after multiple retries

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

## Summary

The most important thing in this section is not memorizing the terms “message passing,” “event bus,” and “shared state,” but understanding this:

> **The key to multi-Agent communication is not just sending messages out, but making the message structure stable, responsibilities clear, and failures controllable.**

Only when the communication layer is solid can a multi-Agent system avoid wasting model capability due to organizational chaos.

---

## Exercises

1. Add a `reviewer_handler` to the event bus example and make it subscribe to `task_done`.
2. Design your own unified message protocol. It should include at least `type`, `task_id`, and `payload`.
3. Think about it: when would you prefer shared state over point-to-point messaging?
4. Explain in your own words: why is communication design often just as important as task division in a multi-Agent system?

<details>
<summary>Reference answers and explanation</summary>

1. `reviewer_handler` should subscribe to `task_done`, read the payload, check whether the result satisfies the criteria, and publish a review event or attach review status to shared state.
2. A useful protocol might include `type`, `task_id`, `from`, `to`, `payload`, `evidence`, `status`, and `timestamp`. The exact fields can vary, but message meaning should be stable.
3. Prefer shared state when many agents need the same evolving artifact or when point-to-point messages would duplicate large context. Prefer direct messages for simple handoffs and narrow requests.
4. Communication design matters because even good roles fail if they receive ambiguous inputs, lose evidence, duplicate work, or cannot tell whether a task is done.

</details>
