---
title: "9.7.1 Multi-Agent Roadmap: Roles, Messages, Owner"
description: "A concise hands-on roadmap for Multi-Agent systems: split roles only when useful, define message contracts, control coordination cost, and keep one final owner."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Multi-Agent guide, collaborative systems, Agent communication, Agent coordination, multi-agent"
---
Multi-Agent is a division-of-labor mechanism, not several chatbots talking. Use it only when role separation, parallel work, cross-checking, or specialist collaboration is worth the coordination cost.

## See the Collaboration Cost First

![Multi-Agent collaboration message flow diagram](/img/course/multi-agent-message-flow-en.webp)

![Multi-Agent chapter learning order diagram](/img/course/ch09-multi-agent-chapter-flow-en.webp)

![Multi-Agent collaboration and coordination map](/img/course/ch09-multi-agent-coordination-map-en.webp)

The key question is: does the benefit of splitting work exceed the cost of messages, repeated context, conflicts, and final merging?

## Run a Role Boundary Check

Every role needs one responsibility and one output. Keep one owner for the final decision.

```python
agents = {
    "researcher": "collect evidence",
    "editor": "rewrite content",
    "reviewer": "check beginner clarity",
}

final_owner = "reviewer"

print("agent_count:", len(agents))
for name, job in agents.items():
    print(f"{name}: {job}")
print("final_owner:", final_owner)
```

Expected output:

```text
agent_count: 3
researcher: collect evidence
editor: rewrite content
reviewer: check beginner clarity
final_owner: reviewer
```

If two roles produce the same output, merge them. If nobody owns the final decision, the system will drift.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | When to use Multi-Agent | Write when a single Agent is better |
| 2 | Common patterns | Compare supervisor-executor, pipeline, debate, expert committee |
| 3 | Communication | Define message format, shared state, and handoff rule |
| 4 | Coordination | Track owner, queue, conflict rule, and aggregation |
| 5 | Practice and risks | Measure cost, loops, duplicated work, and role overreach |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

## Pass Check

You pass this chapter when a 2 to 3 Agent demo has traceable inputs, outputs, handoffs, final ownership, and a clear reason why it beats a single Agent.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer describes the agent loop: goal, plan, tool call, observation, memory or state update, and stop condition.
2. The evidence should include a trace that another developer can inspect, not only the final answer.
3. A good self-check names one safety or reliability control such as tool schemas, permission boundaries, retries, evaluation cases, or a human-review point.

</details>
