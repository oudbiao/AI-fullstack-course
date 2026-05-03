---
title: "Phase Learning Task List"
description: "Break the AI Agent and intelligent agent system phase into executable learning tasks, practice deliverables, and completion criteria."
keywords: [AI Agent, tool calling, MCP, multi-agent, learning task list]
---

# Phase Learning Task List: AI Agent and Intelligent Agent Systems

The goal of this phase is to help you understand how an Agent connects large models, tools, memory, planning, and execution flows. Do not think of an Agent as a “model that chats better.” It is closer to a workflow system that can break down tasks, call tools, record state, and handle failures.

## Required Tasks for This Phase

| Task | Deliverable | Passing Criteria |
| --- | --- | --- |
| Understand the basic Agent structure | An architecture diagram | Can explain the relationship between the model, tools, memory, planner, and executor |
| Get tool calling working end to end | A minimal tool calling example | Can define tool parameters, handle return values, and handle errors |
| Complete the planning and execution flow | A multi-step task demo | Can record the input, output, state, and failure reason for each step |
| Add evaluation and safety | An Agent testing record | Can evaluate task success rate, tool accuracy, and permission risks |
| Complete the phase project | A traceable Agent project | Has trace, logs, replay examples, and human confirmation boundaries |

## Recommended Learning Order

First understand what an Agent is, then study reasoning and planning, tool calling, memory, MCP, frameworks, multi-Agent, evaluation, and safety. Frameworks can help with implementation, but do not let them hide the core questions: how state flows, when tools are called, how failures are recovered, and how permissions are restricted.

When learning Agent systems, pay special attention to observability. If an Agent cannot explain what it did, why it did it, and where it failed, it is hard to use in real applications.

## Relationship to the AI Learning Assistant Project

This phase corresponds to the v0.9 learning planning Agent in the AI Learning Assistant project. It should not only answer questions, but also break down tasks according to learning goals, recommend chapters, generate a study plan, call retrieval tools to look up materials, and record every execution trace.

A recommended minimum feature set includes: input a learning goal, output a phase plan; call the course retrieval tool to find related chapters; generate this week’s task list; record traces. The standard version can add long-term learning records, review reminders, mistake consolidation, and human confirmation.

## Common Roadblocks

Common issues include unclear tool descriptions that lead to wrong calls, an Agent getting stuck in an infinite loop, task decomposition that is too fine-grained or too coarse, memory pollution, passing user input directly to tools and creating security risks, and no logs so problems cannot be debugged. In Agent projects, permission boundaries and stop conditions are just as important as the functionality itself.

## Easy Version / Standard Version / Challenge Version Tasks

| Difficulty | What You Need to Do | Who It Is For |
|---|---|---|
| Easy version | Let the Agent complete 3 fixed tasks and print the steps | First-time learners, learners with little time, or beginners |
| Standard version | Save `agent_traces` and `tool_calls` | Learners who want to include this phase in their portfolio |
| Challenge version | Add privilege escalation tests, human confirmation, and trace replay | Learners with a foundation who want stronger project evidence |

## Badges and Boss Fight for This Phase

| Type | Content |
|---|---|
| Boss fight | The Infinite Loop Boss |
| Unlockable badges | Trace Recorder, Agent Safety Officer |
| Minimum completion slogan | Get it running first, then explain it, then record failures |
| Evidence storage suggestion | Save screenshots, logs, failure samples, or evaluation tables to `reports/`, `evals/`, or `logs/` |

Once you complete the easy version, you can move on. The standard version is the one recommended for your portfolio. The challenge version should only be done if you have extra capacity.

## Phase Portfolio Deliverables

If you want to turn the results of this phase into portfolio material, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `tools_schema.md` | The name, purpose, parameters, return values, and usage boundaries of each tool |
| `agent_traces.jsonl` | The goal, step, action, arguments, observation, and next_decision for each run |
| `safety_boundary.md` | Tool risk levels, human confirmation rules, maximum steps, and permission limits |
| `human_approval_examples.md` | Example confirmation text for high-risk actions, including the action, object, risk, and rollback method |
| `failure_cases.md` | Failure samples such as wrong tool selection, parameter errors, looping calls, unsupported references, and safety blocks |
| `README.md` | Project goals, tool list, how to run, trace examples, evaluation results, and limitations |

These materials will upgrade your Agent project from “can call tools” to a system that is “traceable, controllable, and reviewable.”

## Phase Completion Questions

After finishing this phase, you should be able to answer these questions: What is the difference between an Agent and a regular LLM application? Why are tool descriptions important? What tasks are suitable for Agents? How can you avoid looping calls? Why do you need trace, replay, and human confirmation?

## Completion Checklist

- [ ] I can explain the difference between an Agent, a regular LLM application, and a RAG application.
- [ ] I can define a tool’s name, purpose, parameters, and return values.
- [ ] I can record an Agent’s plan, tool calls, outputs, errors, and stop conditions.
- [ ] I have completed a traceable learning-planning Agent or an equivalent multi-step task demo.
- [ ] I can explain which actions require human confirmation and which tool permissions must be restricted.
