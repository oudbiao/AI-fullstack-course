---
title: "9.4.7 Hands-on Practice: Complete Memory System"
description: "Build a runnable multi-layer memory Agent: short-term window, long-term preferences, episodic records, and procedural memory work together, showing a closed loop of querying, writing, compression, and reply generation."
sidebar:
  order: 24
head:
  - tag: meta
    attrs:
      name: keywords
      content: "memory practice, short term, long term, episodic memory, procedural memory, agent"
---

# 9.4.7 Hands-on Practice: Complete Memory System

:::tip[Section Positioning]
In the previous sections, we split memory into concepts and strategies.
In this section, we will directly build a runnable “small system” and connect these layers together:

- Short-term memory: recent conversation and current state
- Long-term memory: user preferences and stable information
- Episodic memory: records of each task experience
- Procedural memory: fixed workflow steps

The goal is not to build something huge and complete, but to first make the “complete memory closed loop” work.
:::
## Learning Objectives

- Learn how to put multi-layer memory into the same Agent state machine
- Learn how to design rules for “when to write to which memory layer”
- Learn how to make memory actually participate in answering, instead of only acting as storage
- Build a reusable project skeleton through a runnable example

---

## What does the system we want to build look like?

### Target Scenario

We will continue using the customer support assistant scenario.
The user may ask continuously about:

- Refund conditions
- Refund progress
- Response style preferences

The system needs to do two things:

1. Keep the current conversation coherent
2. Remember the user’s preferences the next time they come back

### Division of Labor Across Four Memory Layers

In this example, we divide responsibilities like this:

- `short_term`
  Recent N turns of messages + current task state
- `long_term`
  User’s long-term preferences
- `episodic`
  Summary entries after each task is handled
- `procedural`
  Predefined workflow templates, such as refund handling steps

### Evaluation Goals

The most important checkpoints for this hands-on example are:

- Can it correctly write preferences?
- Can it reference those preferences in later answers?
- Can it leave retrievable episodic records?
- Can it reference procedural memory before answering?

---

## First, run a complete executable version

The code below simulates two rounds of conversation:

1. In the first round, the user says “please answer briefly” and asks about refund conditions
2. In the second round, the user asks about progress, and the system should automatically keep using the concise style

It will print:

- Reply result
- Snapshots of the four memory layers

```python
from collections import deque
from dataclasses import dataclass, asdict


def get_refund_policy():
    return "Refund policy: You may apply for a refund within 7 days after purchase and when learning progress is below 20%. The payment will be returned to the original method, usually arriving within 3-7 business days."


def get_order_status(order_id):
    mock = {
        "ORD-1001": {"status": "not shipped", "progress": 0.12, "amount": 299},
        "ORD-1002": {"status": "shipped", "progress": 0.35, "amount": 499},
    }
    return mock.get(order_id, {"status": "unknown", "progress": None, "amount": None})


@dataclass
class Episode:
    user_id: str
    topic: str
    summary: str


class MemoryAgent:
    def __init__(self, short_window=4):
        self.short_term_messages = deque(maxlen=short_window)
        self.short_term_state = {}
        self.long_term_profile = {}
        self.episodic_memory = []
        self.procedural_memory = {
            "refund_workflow": [
                "Read order status",
                "Read refund policy",
                "Determine whether conditions are met",
                "Return the conclusion and arrival time explanation",
            ]
        }

    def _remember_short(self, role, content):
        self.short_term_messages.append({"role": role, "content": content})

    def _update_profile(self, user_id, message):
        if "briefly" in message:
            self.long_term_profile.setdefault(user_id, {})["style"] = "concise"
        if "detailed" in message:
            self.long_term_profile.setdefault(user_id, {})["style"] = "detailed"

    def _style_for_user(self, user_id):
        return self.long_term_profile.get(user_id, {}).get("style", "default")

    def _format_answer(self, text, style):
        if style == "concise":
            return text[:70] + ("..." if len(text) > 70 else "")
        if style == "detailed":
            return text + " If you'd like, I can also add the specific steps and common reasons for failure."
        return text

    def _write_episode(self, user_id, topic, summary):
        self.episodic_memory.append(Episode(user_id=user_id, topic=topic, summary=summary))

    def handle(self, user_id, user_message, order_id):
        self._remember_short("user", user_message)
        self._update_profile(user_id, user_message)

        self.short_term_state["active_workflow"] = "refund_workflow"
        self.short_term_state["order_id"] = order_id

        workflow = self.procedural_memory["refund_workflow"]
        order_info = get_order_status(order_id)
        policy = get_refund_policy()

        if order_info["status"] == "unknown":
            answer = "I can't find that order right now. Please check the order number and try again."
        elif order_info["progress"] is not None and order_info["progress"] < 0.2:
            answer = (
                f"Order {order_id} currently has a learning progress of {order_info['progress']*100:.0f}%, "
                f"which meets the refund progress requirement. {policy}"
            )
        else:
            answer = (
                f"Order {order_id} currently has a learning progress of {order_info['progress']*100:.0f}%, "
                "which has exceeded the refund progress threshold and does not currently meet the direct refund condition."
            )

        style = self._style_for_user(user_id)
        final_answer = self._format_answer(answer, style)
        self._remember_short("assistant", final_answer)

        self._write_episode(
            user_id=user_id,
            topic="refund",
            summary=f"workflow={workflow}; order={order_id}; style={style}; result={final_answer}",
        )

        return final_answer

    def snapshot(self, user_id):
        return {
            "short_term_messages": list(self.short_term_messages),
            "short_term_state": dict(self.short_term_state),
            "long_term_profile": self.long_term_profile.get(user_id, {}),
            "episodic_memory_tail": [asdict(x) for x in self.episodic_memory[-2:]],
            "procedural_memory": self.procedural_memory,
        }


agent = MemoryAgent(short_window=4)

print("round1:")
print(agent.handle("u_001", "Please answer briefly, I want to see the refund conditions", "ORD-1001"))
print("\nround2:")
print(agent.handle("u_001", "Then when will it arrive?", "ORD-1001"))

print("\nmemory snapshot:")
print(agent.snapshot("u_001"))
```

Expected output:

```text
round1:
Order ORD-1001 currently has a learning progress of 12%, which meets t...

round2:
Order ORD-1001 currently has a learning progress of 12%, which meets t...

memory snapshot:
{'short_term_messages': [{'role': 'user', 'content': 'Please answer briefly, I want to see the refund conditions'}, {'role': 'assistant', 'content': 'Order ORD-1001 currently has a learning progress of 12%, which meets t...'}, {'role': 'user', 'content': 'Then when will it arrive?'}, {'role': 'assistant', 'content': 'Order ORD-1001 currently has a learning progress of 12%, which meets t...'}], 'short_term_state': {'active_workflow': 'refund_workflow', 'order_id': 'ORD-1001'}, 'long_term_profile': {'style': 'concise'}, 'episodic_memory_tail': [{'user_id': 'u_001', 'topic': 'refund', 'summary': "workflow=['Read order status', 'Read refund policy', 'Determine whether conditions are met', 'Return the conclusion and arrival time explanation']; order=ORD-1001; style=concise; result=Order ORD-1001 currently has a learning progress of 12%, which meets t..."}, {'user_id': 'u_001', 'topic': 'refund', 'summary': "workflow=['Read order status', 'Read refund policy', 'Determine whether conditions are met', 'Return the conclusion and arrival time explanation']; order=ORD-1001; style=concise; result=Order ORD-1001 currently has a learning progress of 12%, which meets t..."}], 'procedural_memory': {'refund_workflow': ['Read order status', 'Read refund policy', 'Determine whether conditions are met', 'Return the conclusion and arrival time explanation']}}
```

![MemoryAgent four-layer snapshot result map](/img/course/ch09-memory-four-layer-snapshot-result-map-en.webp)

### What four-layer collaboration does this code demonstrate?

1. `short_term_messages`
   Keeps recent conversation
2. `long_term_profile`
   Remembers the user’s style preference
3. `episodic_memory`
   Stores one “experience record” after each task is completed
4. `procedural_memory`
   Defines the refund task workflow template

All four layers are used here, so this is no longer “just a concept, not actually running”.

### Why can the second round still stay concise?

Because in the first round, the user said “please answer briefly,”
and the system wrote that into long-term preference:

- `long_term_profile["u_001"]["style"] = "concise"`

So even if the user does not repeat it in the second round, the reply will continue to follow that style.

### What is the value of episodic memory here?

After each task is handled, the system writes one episode summary.
This allows us to answer later questions such as:

- What refund decisions has the user experienced before?
- What was the basis at that time?

This is very useful for review and explanation.

---

## How else can this system be extended?

### Add “confidence” and “update time” to long-term memory

This helps prevent very old or low-confidence information from continuing to affect answers.

### Add retrieval to episodic memory

For example, retrieve past experiences by topic and keywords,
so that complex questions can be answered with historical reference.

### Version procedural memory

When the workflow changes, you can track:

- Which version of the workflow was used in each conversation

This is important for auditing and replay.

---

## The easiest pitfalls to fall into in real practice

### Writing everything into long-term memory

The result will be:

- More and more retrieval noise

### No “write threshold”

For example, if the user casually says something once and it gets written into long-term memory,
the system can easily learn the wrong preference.

### Storing memory without letting it affect decisions

In that case, the system may look like it “has memory,”
but in reality the answers do not change at all.

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
memory_type: short-term, long-term, episodic, or procedural
write_rule: when memory is created or updated
retrieve_rule: query, relevance, recency, and permission check
failure_check: stale memory, privacy leak, contradiction, or over-retrieval
cleanup_action: summarize, merge, expire, delete, or ask for confirmation
```

## Summary

The most important thing in this section is to turn the “complete memory system” into an executable closed loop:

> **Short-term memory keeps the current task, long-term memory retains stable preferences, episodic memory accumulates historical experiences, and procedural memory stores reusable workflows.**

When these four layers work together, the Agent can evolve from a “one-shot Q&A tool” into a “continuously usable task system.”

---

## Exercises

1. Add a `user_blacklist_topic` long-term preference to the example and see whether the system can avoid irrelevant topics in its answers.
2. Make `episodic_memory` support retrieving the latest record by `topic`.
3. Change `procedural_memory` into a multi-workflow version, such as `refund_workflow` and `invoice_workflow`.
4. Think about it: Which information is best kept only in short-term memory, and not written into long-term memory?

<details>
<summary>Reference implementation and walkthrough</summary>

1. `user_blacklist_topic` should be stored as an explicit long-term preference with a clear scope and should suppress unrelated suggestions, not block necessary safety or task information.
2. Retrieving the latest episode by `topic` usually means filtering by topic and sorting by timestamp or monotonically increasing id.
3. A multi-workflow procedural memory can be a dictionary keyed by workflow name, such as `refund_workflow` and `invoice_workflow`, each with steps and risk gates.
4. Keep one-off constraints, temporary goals, current tool results, draft choices, and sensitive session-only information in short-term memory instead of long-term memory.

</details>
