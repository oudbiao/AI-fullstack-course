---
title: "9.0 Learning Checklist: AI Agents and Agent Systems"
sidebar_position: 1
description: "A compact checklist for Chapter 9: Agent loops, tool schema, traces, safety boundaries, evaluation, and portfolio evidence."
keywords: [Agent checklist, AI Agent learning, ReAct, MCP, tool calling, Agent evaluation]
---

# 9.0 Learning Checklist: AI Agents and Agent Systems

Use this page as a printable checklist. If you need the full explanation, return to the [Chapter 9 entry page](./index.md).

![Agent trace evidence pack](/img/course/ch09-agent-trace-pack-en.webp)

## Two-Hour First Pass

| Time box | Do this | Stop when you can say |
|---|---|---|
| 20 min | Read the execution loop on the entry page | "An Agent is a goal-state-tool-observation loop." |
| 25 min | Run the trace script | "I can replay every action and observation." |
| 25 min | Skim 9.1 and 9.2 | "I can separate Agent, workflow, RAG, ReAct, and Plan-and-Execute." |
| 25 min | Skim 9.3 tool safety | "Tool schema and permissions matter more than clever prompting." |
| 25 min | Read the boundary map | "I know when not to use an Agent." |

## Required Evidence

| Evidence | Minimum version |
|---|---|
| `tools_schema.md` | 1-2 tools with name, purpose, parameters, return value, errors, and risk level |
| `agent_traces.jsonl` | at least three runs with goal, step, action, input, observation, and result |
| `safety_boundary.md` | maximum steps, tool whitelist, blocked actions, human approval rules |
| `failure_cases.md` | at least three failures: wrong tool, bad parameter, loop, blocked permission, unsupported answer |
| `eval_tasks.csv` | 3-5 fixed tasks with expected outcome and success criteria |
| `README.md` | run command, trace example, safety example, evaluation result, limitation |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
single_agent_trace: one complete goal-plan-action-observation loop
tool_contract: schema, permission, error behavior, and observation
memory_note: what is written, retrieved, forgotten, or updated
eval_note: success score, safety check, and failure reason
project_readme: run command, trace, limitations, and next action
```

## Quality Gates

| Gate | Pass condition |
|---|---|
| Tool schema | Each tool has purpose, parameters, return value, errors, and risk level. |
| Trace replay | A reviewer can replay why every tool call happened. |
| Safety boundary | Non-whitelisted or risky actions are blocked or routed to human approval. |
| Stop control | Max steps and stop conditions prevent loops and cost spikes. |

Expected result: your Chapter 9 project folder contains tool schemas, replayable traces, safety boundaries, fixed eval tasks, failure notes, and a README that explains why the design stays single-Agent until the loop is reliable.

## Exit Questions

- Can you explain why an Agent is different from a normal LLM application?
- Can you show a trace and explain why each tool call happened?
- Can you block a risky or non-whitelisted tool?
- Can you define a stop condition and maximum step count?
- Can you explain why multi-agent should come after single-Agent reliability?

<details>
<summary>Check reasoning and explanation</summary>

1. An Agent keeps a goal-plan-tool-observation loop, so the system can act, inspect the result, and decide the next step instead of only generating one reply.
2. A useful trace shows the goal, planned step, tool call, input, observation, and why the next step followed from that observation.
3. Block risky or non-whitelisted tools with a tool allowlist, schema checks, risk labels, maximum step limits, and human approval when needed.
4. Good stop conditions include success, no progress, max steps reached, or risk escalation.
5. Single-Agent stability comes first because multi-agent systems are harder to trace, debug, and control safely.

</details>

If the answer is yes, continue to the next direction: deployment, multimodal Agents, or the final course project.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
single_agent_trace: one complete goal-plan-action-observation loop
tool_contract: schema, permission, error behavior, and observation
memory_note: what is written, retrieved, forgotten, or updated
eval_note: success score, safety check, and failure reason
project_readme: run command, trace, limitations, and next action
```
