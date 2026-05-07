---
title: "9.0 Study Guide and Task Sheet: How to Learn AI Agents and Agent Systems Without Getting Confused"
sidebar_position: 1
description: "A learning guide for AI full-stack beginners on Agents: task planning, tool calling, memory, MCP, multi-Agent, evaluation, deployment, project roadmap, and acceptance criteria."
keywords: [Agent Learning Guide, How to learn AI Agent, How to learn ReAct, How to learn MCP, Multi-Agent]
---

# 9.0 Study Guide and Task Sheet: How to Learn AI Agents and Agent Systems Without Getting Confused

If you reach `Chapter 9: AI Agents and Agent Systems` and feel that tools, memory, frameworks, multi-Agent, and MCP are all mixed together, first go back to the smallest Agent loop: goal, plan, action, observation, refinement.

## Core principle for this stage

For your first Agent, do not rush into multi-Agent, and do not pile on frameworks right away. First make a single Agent solid: it can understand the goal, decide the next step, call tools, observe the result, and then decide whether to continue.

![Minimal Agent learning loop diagram](/img/course/ch09-study-guide-minimal-agent-loop-en.png)

## Tasks You Must Complete in This Stage

Use these tasks to keep Agent learning observable. An Agent project is not complete when it calls a tool once; it is complete when you can replay the trace, explain the stop condition, and show where human confirmation is required.

| Task | Deliverable | Passing Criteria |
|---|---|---|
| Understand the basic Agent structure | An architecture diagram or notes | Can explain the relationship between the model, tools, memory, planner, and executor |
| Get tool calling working end to end | A minimal tool-calling example | Can define tool parameters, handle return values, and handle errors |
| Complete the planning and execution flow | A multi-step task demo | Can record the input, output, state, and failure reason for each step |
| Add evaluation and safety | An Agent testing record | Can evaluate task success rate, tool accuracy, permission risks, and stop conditions |
| Run the guided single-Agent workshop | `agent_workshop.py` output and trace logs | Can explain `logs/agent_traces.jsonl`, mini evaluation output, and permission checks |
| Complete one stage project | A traceable Agent project | Has trace, logs, replay examples, and human confirmation boundaries |

## Recommended learning order

In the first round, learn the basics of Agents and clearly distinguish between Agents, ordinary chatbots, fixed workflows, and RAG applications.

In the second round, learn reasoning and planning, including ReAct, Chain-of-Thought, Plan-and-Execute, and reasoning evaluation.

In the third round, learn tool calling. Tool descriptions, parameter schemas, error handling, permission boundaries, and safety policies are the keys to whether an Agent can execute reliably.

In the fourth round, learn memory systems. Short-term memory, long-term memory, episodic memory, and procedural memory should serve the task, not exist just to “look smart.”

In the fifth round, learn MCP, frameworks, multi-Agent, evaluation, security, and deployment. Use frameworks only after you understand the system layers.

## Decision Map: When Not to Use an Agent First

Beginners often reach for an Agent too early. A safer rule is to first ask whether the problem really needs a goal-driven execution loop.

| Problem type | First choice | Why |
|---|---|---|
| Fixed process, fully known steps | Workflow | The program can already decide every step in advance |
| Need fresh or citable knowledge | RAG | First connect external knowledge instead of letting the model guess |
| One-off structured action such as classification or extraction | Prompt / Function Calling | A full Agent loop is unnecessary |
| Repeated goal-driven actions with tool use and state updates | Agent | This is where Agent systems are strongest |

If you really do need an Agent, pass three safety gates first:

1. Are permissions minimized?
2. Is there a maximum step limit?
3. Can every step be logged and manually taken over?

## Upgrade Order: Single Agent Before Multi-Agent

1. Make a single Agent solid first
2. Connect tools and memory next
3. Add MCP or a framework after the loop is stable
4. Only then consider multi-Agent collaboration

## Suggested learning pace

| Content type | Suggested time | Learning goal |
|---|---|---|
| Agent basics | 4–8 hours | Distinguish Agents from ordinary applications |
| Reasoning and tools | 8–16 hours | Complete the single-Agent execution loop |
| Memory and MCP | 8–16 hours | Understand external context and the tool ecosystem |
| Frameworks and multi-Agent | 8–20 hours | Judge when complex orchestration is needed |
| Evaluation, deployment, and projects | 16–32 hours | Build an observable, recoverable Agent demo |

## Project roadmap by stage

For the first project, it is recommended to build a research assistant. Given a topic, the Agent breaks down the problem, retrieves materials, organizes summaries, and outputs a structured report.

Before choosing among the larger projects, run the [9.10.5 traceable single-Agent workshop](./ch10-projects/04-stage-hands-on-workshop.md). It is the baseline exercise for this stage: one Agent, a few tools, permission checks, trace logs, and fixed evaluation cases.

For the second project, it is recommended to build a data analysis Agent. It can read data, make an analysis plan, call Python tools, and generate charts and conclusions.

For the third project, you can build a multi-Agent development team demo, where different roles collaborate on requirements analysis, code generation, testing, and documentation.

## Common sticking points

The most common sticking point is doing multi-Agent too early. If single-Agent tool calling, error recovery, and context management are not done well, multi-Agent will only amplify the chaos.

The second sticking point is only demonstrating the success path. A real Agent must consider tool failures, parameter errors, insufficient permissions, infinite loops, uncontrolled cost, and untrustworthy outputs.

The third sticking point is learning more about frameworks than about the system itself. You should first understand the Agent architecture, then choose LangGraph, CrewAI, AutoGen, or another framework.

## Stage Portfolio Deliverables

![Agent trace replay map](/img/course/ch09-workshop-trace-jsonl-replay-map-en.png)

If you want this stage to become portfolio material, keep at least these files or equivalent evidence.

| Deliverable | Description |
|---|---|
| `tools_schema.md` | The name, purpose, parameters, return values, and usage boundaries of each tool |
| `agent_traces.jsonl` | The goal, step, action, arguments, observation, and next decision for each run |
| `safety_boundary.md` | Tool risk levels, human confirmation rules, maximum steps, and permission limits |
| `human_approval_examples.md` | Confirmation text for high-risk actions, including the action, object, risk, and rollback method |
| `failure_cases.md` | Wrong tool selection, parameter errors, looping calls, unsupported references, and safety blocks |
| `README.md` | Project goals, tool list, how to run, trace examples, evaluation results, and limitations |

These materials upgrade an Agent project from “can call tools” to a system that is traceable, controllable, and reviewable.

## Stage Completion Questions

After finishing this stage, you should be able to explain an Agent’s goal, state, tools, memory, planning, and evaluation methods.

Before moving on, check that you can answer these questions:

- What is the difference between an Agent and a regular LLM application?
- Why are tool descriptions important?
- What tasks are suitable for Agents?
- How can you avoid looping calls?
- Why do you need trace, replay, and human confirmation?

If you can build a single-Agent project and record each tool call, observed result, failure handling step, and final output, then you can move on to multi-Agent, deployment, or multimodal directions.
