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

## Exit Questions

- Can you explain why an Agent is different from a normal LLM application?
- Can you show a trace and explain why each tool call happened?
- Can you block a risky or non-whitelisted tool?
- Can you define a stop condition and maximum step count?
- Can you explain why multi-agent should come after single-Agent reliability?

If the answer is yes, continue to the next direction: deployment, multimodal Agents, or the final course project.
