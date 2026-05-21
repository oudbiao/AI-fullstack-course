---
title: "9.4.3 Short-Term Memory"
sidebar_position: 20
description: "From context windows and conversation windows to runtime state and summary compression, understand what an Agent’s short-term memory really is and how to design it."
keywords: [short-term memory, context window, conversation memory, state, summary memory, Agent]
---

# 9.4.3 Short-Term Memory

![Short-term memory context window and runtime state](/img/course/ch09-short-term-memory-window-map-en.webp)

:::tip Section Overview
When many people hear “Agent memory,” they first think of “long-term storage.”
But in real systems, what most directly determines the experience is often short-term memory:

> **Can the system steadily remember “what is happening in this task right now”?**

This section is about that layer of “working memory.”
:::

## Learning Objectives

- Understand the difference between short-term memory and long-term memory
- Understand why you cannot keep stuffing the entire history into the model forever
- Master three common short-term memory approaches: conversation windows, runtime state, and summary memory
- Read a simple short-term memory manager
- Know the most common ways short-term memory fails

---

## What Exactly Is Short-Term Memory?

### A One-Sentence Definition

You can first think of short-term memory as:

> **The context and intermediate state that a system temporarily keeps in order to complete the current task.**

It usually includes:

- The most recent few turns of conversation
- The current task goal
- The steps already executed
- Temporary intermediate results

### How Is It Different from Long-Term Memory?

| Type | What it focuses on |
|---|---|
| Short-term memory | Information needed for the current task |
| Long-term memory | Information that remains valuable across tasks and sessions |

For example:

- “The user said they want to check the refund policy” -> short-term memory
- “This user likes concise answers” -> more like long-term memory

---

## Why Can’t We Just Keep Feeding the Model All the History?

### Because the Context Window Is Not Infinite

The model can only see a limited amount of context.
If you keep stuffing all the history into it, you will run into:

- Higher and higher token cost
- Slower and slower responses
- Important information getting buried

### More Information Is Not Always Better

Many beginners think:

> “If we give the model a bit more history, that should never hurt, right?”

Not necessarily.

If the context contains too much unrelated content, the model is more likely to:

- Focus on the wrong thing
- Repeat old information
- Forget what it is actually supposed to do right now

So the real job of short-term memory is not “the more the better,” but:

> **Keep the most useful information within a limited budget.**

---

## The Three Most Common Forms of Short-Term Memory

### Conversation Window (Sliding Window)

The simplest approach is:

- Keep only the most recent N turns of messages

Advantages:

- Simple
- Low implementation cost

Disadvantages:

- Important information from too long ago gets pushed out

### Runtime State (Task State)

Instead of only remembering chat text, explicitly keep track of:

- The current task goal
- What has already been checked
- What the next step should be

This kind of state is especially important for Agents.

### Summary Memory

When the history gets too long, don’t discard it entirely—compress it into a summary first.

For example:

- Keep the most recent 4 turns in full
- Compress older content into a short summary

This is a very common trade-off.

---

## The Simplest Short-Term Memory: A Sliding Window

### Runnable Example

```python
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello, what can I help you with?"},
    {"role": "user", "content": "I want to understand the refund policy"},
    {"role": "assistant", "content": "Are you asking about the time limit or the specific conditions?"},
    {"role": "user", "content": "Mainly the time limit"},
]

window_size = 3
short_term_memory = messages[-window_size:]

for msg in short_term_memory:
    print(msg)
```

Expected output:

```text
{'role': 'user', 'content': 'I want to understand the refund policy'}
{'role': 'assistant', 'content': 'Are you asking about the time limit or the specific conditions?'}
{'role': 'user', 'content': 'Mainly the time limit'}
```

### This Code Is Simple, but Still Very Important

It teaches you something essential:

> Short-term memory is first and foremost a question of “which messages should be kept.”

Not every piece of history is worth carrying forward.

---

## But a Message Window Alone Is Not Enough

### Why Not?

Look at this conversation:

1. The user says, “I want to check the refund policy”
2. Then they ask several other details in a row
3. On the 10th turn, they ask, “Can I get a refund in my situation?”

If you only keep the most recent 3 turns, the system may have already forgotten:

- That the whole task was actually about “refunds”

### So an Agent Also Needs Structured State

For example:

```python
task_state = {
    "goal": "Help the user determine refund eligibility",
    "last_tool": "search_policy",
    "latest_policy_result": "Refunds are available within 7 days of purchase and if learning progress is below 20%"
}

print(task_state)
```

Expected output:

```text
{'goal': 'Help the user determine refund eligibility', 'last_tool': 'search_policy', 'latest_policy_result': 'Refunds are available within 7 days of purchase and if learning progress is below 20%'}
```

This kind of state is different from raw chat logs. It is more like:

> The workspace for what the system is currently doing.

---

## A More Teaching-Friendly Short-Term Memory Manager

The example below manages both:

- The most recent few messages
- The current task state

```python
class ShortTermMemory:
    def __init__(self, max_messages=4):
        self.max_messages = max_messages
        self.messages = []
        self.state = {}

    def add_message(self, role, content):
        self.messages.append({"role": role, "content": content})
        self.messages = self.messages[-self.max_messages:]

    def update_state(self, **kwargs):
        self.state.update(kwargs)

    def snapshot(self):
        return {
            "messages": self.messages,
            "state": self.state
        }

memory = ShortTermMemory(max_messages=3)
memory.add_message("user", "I want to check the refund policy")
memory.add_message("assistant", "Are you more concerned about the time limit or the conditions?")
memory.add_message("user", "First, let’s look at the time limit")
memory.update_state(goal="Determine refund eligibility", topic="refund policy")

print(memory.snapshot())
```

Expected output:

```text
{'messages': [{'role': 'user', 'content': 'I want to check the refund policy'}, {'role': 'assistant', 'content': 'Are you more concerned about the time limit or the conditions?'}, {'role': 'user', 'content': 'First, let’s look at the time limit'}], 'state': {'goal': 'Determine refund eligibility', 'topic': 'refund policy'}}
```

![Short-term memory snapshot result map](/img/course/ch09-short-term-memory-snapshot-result-map-en.webp)

### What Makes This Example Better Than “Just Storing Message History”?

Because it splits short-term memory into two layers:

- Text context
- Structured state

This is very important in Agent systems.

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

## Summary Memory: What Should We Do When Messages Keep Growing?

### A Common Strategy

In real systems, this is a very common approach:

- Keep the most recent few turns as-is
- Compress older history into a summary

### A Simplified Example

```python
old_messages = [
    "The user first asked about the refund policy",
    "Then they asked about certificate requirements",
    "Finally they returned to the refund conditions"
]

summary = "The user’s main goal in this session is to determine whether they meet the refund conditions, and they also asked about certificates along the way."

recent_messages = [
    {"role": "user", "content": "Can I still get a refund if my learning progress is 30%?"}
]

memory_package = {
    "summary": summary,
    "recent_messages": recent_messages
}

print(memory_package)
```

Expected output:

```text
{'summary': 'The user’s main goal in this session is to determine whether they meet the refund conditions, and they also asked about certificates along the way.', 'recent_messages': [{'role': 'user', 'content': 'Can I still get a refund if my learning progress is 30%?'}]}
```

This is the most basic “summary + recent window” idea.

---

## What Does Short-Term Memory Actually Solve in an Agent?

It mainly solves three things:

### Keeping the Current Task Coherent

The system should not restart from scratch at every step as if it were seeing the user for the first time.

### Preserving State Across Multi-Step Execution

For example:

- Which tool has already been called
- What has already been found
- What step is still missing

### Controlling Context Cost

Short-term memory is not only about “remembering.” It is also about:

- Avoiding unnecessary content
- Reducing token cost
- Improving response stability

---

## The Most Common Ways Short-Term Memory Fails

### Remembering Too Little

Symptoms:

- The system suddenly forgets what it was just talking about

### Remembering Too Much

Symptoms:

- The context becomes long and messy
- Answers drift off track
- Cost goes up

### Storing Only Messages, Not State

Symptoms:

- Multi-step tasks easily break down
- The connection between tool calls before and after becomes weak

### Storing Only State, Not the Original Dialogue

Symptoms:

- The user’s original wording gets lost easily
- Tone, constraints, and details disappear

So short-term memory is usually not “choose just one,” but rather a combined design.

---

## Common Pitfalls for Beginners

### Mixing Up Short-Term Memory and Long-Term Memory

Short-term memory is for the current task, not for a complete user profile.

### Thinking a Bigger Message Window Is Always Better

A window that is too large also brings noise and cost.

### Ignoring Structured State

This can make an Agent start drifting as soon as the task becomes multi-step.

---

## Summary

The most important thing in this section is not to memorize the words “window” or “summary,” but to grasp this main idea:

> **The goal of short-term memory is not to preserve history forever, but to maintain coherence for the current task within a limited context.**

Well-designed short-term memory usually includes both recent messages and task state, and sometimes an additional layer of summary compression.

---

## Exercises

1. Extend the `ShortTermMemory` example in this section to support a `summary` field.
2. Change the maximum message window from 3 to 5 and observe how the `snapshot()` output changes.
3. Think about this: if an Agent often forgets “which tool it has already called,” would you first expand the message window or add structured state?
4. Explain in your own words: why do we say short-term memory solves “current task coherence” rather than “long-term user profiling”?

<details>
<summary>Reference implementation and walkthrough</summary>

1. A `summary` field can compress older turns into a short current-task note while the raw message window keeps the latest details.
2. Changing the window from 3 to 5 should keep more recent messages in `snapshot()`, which may improve coherence but also adds noise and tokens.
3. If the Agent forgets which tools it already called, add structured state first. Enlarging the message window is a weaker and more expensive fix.
4. Short-term memory keeps the current task coherent: goal, constraints, recent corrections, tool results, and next action. It is not meant to become a permanent user profile.

</details>
