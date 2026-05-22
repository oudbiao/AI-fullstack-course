---
title: "9.4.1 Memory Roadmap: Write, Retrieve, Forget"
description: "A concise hands-on roadmap for Agent memory: decide what to remember, retrieve the right context, update stale facts, and avoid memory pollution."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "memory systems overview, Agent memory, short-term memory, long-term memory, episodic memory"
---

# 9.4.1 Memory Roadmap: Write, Retrieve, Forget

Memory is not there to make an Agent feel human. It should help the task: reduce repeated questions, preserve useful context, reuse experience, and avoid stale or private information leaks.

## See the Memory Loop First

![Layered diagram of the Agent memory system](/img/course/agent-memory-system-en.webp)

![Learning order diagram for the Agent memory systems chapter](/img/course/ch09-memory-chapter-flow-en.webp)

![Closed loop diagram of Agent memory writing and retrieval](/img/course/ch09-memory-write-retrieve-loop-en.webp)

The core decision is not “save everything.” It is what to save, when to retrieve it, when to update it, and when to forget it.

## Run a Memory Write Filter

Only stable preferences and reusable facts should become long-term memory.

```python
events = [
    {"type": "preference", "text": "prefers short examples"},
    {"type": "temporary", "text": "debugging one local error"},
    {"type": "fact", "text": "project uses Python"},
]

memory = []
for event in events:
    if event["type"] in {"preference", "fact"}:
        memory.append(event["text"])

print("saved:", memory)
print("count:", len(memory))
```

Expected output:

```text
saved: ['prefers short examples', 'project uses Python']
count: 2
```

If a memory is not useful, current, permitted, and retrievable, it can hurt the Agent more than it helps.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Memory overview | Distinguish context window, short-term memory, long-term memory |
| 2 | Short-term memory | Track current task state across turns |
| 3 | Long-term memory | Save durable preferences, facts, and project background |
| 4 | Episodic and procedural memory | Separate what happened from how to do it next time |
| 5 | Memory engineering | Design write, retrieve, update, expire, and delete rules |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
memory_type: short-term, long-term, episodic, or procedural
write_rule: when memory is created or updated
retrieve_rule: query, relevance, recency, and permission check
failure_check: stale memory, privacy leak, contradiction, or over-retrieval
cleanup_action: summarize, merge, expire, delete, or ask for confirmation
```

## Pass Check

You pass this chapter when you can explain why “remember more” is not the same as “perform better.”

The exit mini project is a learning-planning assistant memory rule set: what to save, what to confirm, what to keep temporary, and what to delete.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer describes the agent loop: goal, plan, tool call, observation, memory or state update, and stop condition.
2. The evidence should include a trace that another developer can inspect, not only the final answer.
3. A good self-check names one safety or reliability control such as tool schemas, permission boundaries, retries, evaluation cases, or a human-review point.

</details>
