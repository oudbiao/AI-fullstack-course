---
title: "9 AI Agent and Intelligent Agent Systems"
sidebar_position: 0
description: "Learn AI Agent task planning, reasoning, tool calling, memory systems, MCP, multi-Agent collaboration, evaluation, safety, and deployment operations."
keywords: [AI Agent, agent, Function Calling, ReAct, MCP, Multi-Agent, tool calling]
---

# 9 AI Agent and Intelligent Agent Systems

![Main visual of the AI Agent system](/img/course/ch09-agent-systems-en.png)

This stage is about answering: “How do we make AI do more than answer questions, and actually carry out tasks toward a goal?” An Agent combines a large model, tools, memory, planning, evaluation, and systems engineering into an AI system that can keep acting over time.

## Story-based introduction: upgrading from chat assistant to task teammate

A regular chatbot is like a person sitting at a table answering questions, while an Agent is more like a teammate who can pick up tools, look up materials, break down tasks, execute steps, and then come back to check the results. It does not just “say” things; it also has to “do” things. It does not just give one answer; it keeps moving toward the goal.

## Interactive exercise: first decide whether you really need an Agent

Whenever you come up with an AI application idea, ask three questions first: Is it a multi-step task? Does it need to adjust the route based on intermediate results? Does it need external tools or long-term memory? If the answer to all three is no, a normal workflow or RAG may be more stable. If most answers are yes, then consider an Agent architecture.

## Project bonus content

The bonus project for this stage is a “research assistant Agent”: it can break a topic into questions, call search or knowledge-base tools, organize evidence, generate a report, and record which steps succeeded and which failed. This project connects everything you learned before—Prompt, RAG, tool calling, logs, and evaluation—into one complete system.

## Stage positioning

| Information | Description |
|---|---|
| Suitable for | Learners who have finished LLM applications and RAG, and want to build automation assistants, research assistants, data analysis Agents, or multi-Agent systems |
| Estimated time | 150–200 hours |
| Prerequisites | Completed the main track on large model principles and LLM application development |
| Stage output | Research assistant, data analysis Agent, multi-Agent development team, or office automation Agent |

## Minimal path for beginners to clear the stage

Beginners should first understand the Agent’s goals, state, plan, tools, observations, and memory. Do not rush into complex frameworks. As long as you can build a minimal Agent that breaks down tasks, calls one or two tools, records the execution process, and outputs results, you have completed the minimum path.

## Advanced deep-dive path

Experienced learners can go deeper into ReAct, Plan-and-Execute, memory engineering, MCP, multi-Agent collaboration, evaluation safety, and production deployment. You can also compare fixed workflows, RAG, and Agents on the same task to see their reliability, cost, and failure modes.

## Two ways to read this stage

When newcomers read this stage, they should not rush to framework names or complex papers. On the first pass, focus only on the execution loop: “goal → plan → tool → observation → adjustment → record.” First build a minimal Agent that can finish small tasks, show each step in a trace, and stop when something goes wrong.

Experienced learners can focus on boundaries and reliability: when not to use an Agent, how to limit tool permissions, how to evaluate multi-step tasks, how to prevent memory contamination, and how to recover from failures. You can implement the same task as a fixed workflow, RAG, and Agent, then compare stability, cost, interpretability, and maintenance difficulty.

## How is an Agent different from a normal LLM application?

A normal LLM application usually follows a fixed process: the user inputs something, the system organizes context, and the model outputs an answer. An Agent emphasizes goals, state, and action: it must decide what to do next, choose tools, read results, update context, and replan when needed.

![Main execution line diagram of Agent vs. normal application](/img/course/ch09-agent-vs-workflow-backbone-en.png)

The first half of the journey is about turning “what to do” into “what to do next.” If the goal, plan, or tool choice is unclear, the Agent can easily drift off course later.

## AgentOps deep dive: making Agents traceable, controllable, and recoverable

In 2025–2026, the focus of Agent learning has shifted from “making the model call tools” to “making the intelligent agent system traceable, controllable, and evaluable.” A reliable Agent should not only show successful results; it should also explain what the goal was, why a certain tool was chosen, what parameters were passed, what was observed, how much it cost, how it recovered from failure, and when human confirmation was needed.

| Deep-dive topic | Problem it solves | Learning focus |
|---|---|---|
| MCP | Tool, file, database, and business system access is not unified | Understand the standard connection method between models and external tools |
| Tool Schema | The model does not know tool parameters, constraints, or error meanings | Design clear parameters, return values, error messages, and validation rules |
| Agentic Workflow | Fully open-ended Agents are unstable, but fixed workflows are not flexible enough | Combine deterministic steps with model decisions |
| Human-in-the-loop | High-risk actions should not run automatically by default | Steps like delete, send, order, database writes, and publish require human confirmation |
| Agent Observability | Multi-step failures cannot be reviewed afterward | Record plans, actions, observations, costs, errors, and final results |
| Agent Evaluation | You cannot judge based on one successful demo | Use a fixed task set to measure completion rate, step count, tool errors, and unauthorized behavior |
| Multi-agent Orchestration | Multiple roles can easily wait on each other or duplicate work | Define roles, communication methods, stopping conditions, and the final owner |

## The minimal engineering loop for a controllable Agent

The core of an Agent is not “letting the model do whatever it wants,” but putting freedom inside boundaries. A minimal controllable Agent can first be limited to: only calling whitelisted tools, only reading specified directories or materials, only outputting drafts rather than publishing directly, requiring user confirmation for high-risk actions, and writing every step into an execution trace.

![Controllable AgentOps execution loop](/img/course/ch09-agentops-control-loop-en.png)

When building a project, you can start with a “research assistant Agent”: it can break down questions, retrieve materials, generate summaries, and record traces, but it does not automatically send emails, delete files, or modify databases. This both demonstrates Agent capabilities and reflects engineering boundaries.

## Learning path for this stage

Chapter 1 covers Agent fundamentals, including the difference between Agents and chatbots, their development history, capability levels, and system architecture.

Chapter 2 covers reasoning and planning, including Chain-of-Thought, ReAct, Plan-and-Execute, and reasoning evaluation.

Chapter 3 covers tool use and Function Calling. You will learn tool descriptions, parameter design, calling strategies, safety boundaries, and code-execution Agents.

Chapter 4 covers memory systems, including short-term memory, long-term memory, episodic memory, procedural memory, and memory engineering.

Chapter 5 covers MCP and how model and external tool ecosystems connect through a protocol.

Chapters 6 and 7 cover Agent frameworks and multi-Agent systems, including LangGraph, LlamaIndex, CrewAI, and AutoGen.

Chapters 8 to 10 cover evaluation, safety, deployment, and integrated projects.

## What you should be able to do after finishing

- Explain the structure of an Agent’s goals, state, tools, memory, and planning
- Design an execution flow in a ReAct or Plan-and-Execute style
- Design clear parameters and safety boundaries for tool calling
- Judge whether a task really needs an Agent rather than a normal workflow or RAG
- Build a minimal usable research assistant or data analysis Agent
- Consider evaluation, cost, permissions, and failure recovery for an Agent

## Common misconceptions

Do not think of an Agent as simply “adding tools to the model.” Tools are only one layer. The real difficulty lies in task boundaries, context management, error recovery, permission control, and result evaluation.

Also, do not use an Agent for every task. Fixed workflows and clearly defined, high-risk tasks are sometimes better suited to traditional workflows. Agents are more appropriate for open-ended problems, multi-step exploration, and scenarios that require dynamic tool use.

## Agent failure theater: being able to act does not mean being reliable

If an Agent gets stuck in a loop, first check whether the stopping conditions are clear. If tool calls are wrong, first check the schema, parameter validation, and permission boundaries. If the result seems successful but cannot be reviewed later, then execution traces and logs are missing. If the task is high risk, add human confirmation instead of letting the Agent complete everything automatically.

## How to read this stage the first time: required, project reference, and optional deep dive

| Reading tag | Suggested chapters | Learning goal |
|---|---|---|
| Required | Agent fundamentals, ReAct, Plan-and-Execute, tool descriptions, tool safety | First understand why Agents act and how to control the boundaries |
| Project reference | Function Calling deep dive, common tools, memory engineering, MCP, evaluation safety | Check these sections when building a research assistant, data analysis Agent, or tool-based Agent |
| Optional deep dive | Agent frameworks, multi-Agent, low-code platforms, deployment architecture, and cost optimization | Go deeper only when building complex systems, team collaboration, or production deployments |

On the first pass, do not try to learn every framework. First build a minimal Agent that prints traces, then decide whether you need LangGraph, CrewAI, AutoGen, MCP, or a multi-Agent architecture.

## Small runnable Agent experiment: print the trace first

The smallest Agent experiment does not necessarily need a complex framework. You can first write a script with only two tools: one tool for searching course materials, and one tool for generating a to-do list. Record `thought`, `action`, `input`, `observation`, and `cost_estimate` for every tool call. This helps learners see that an Agent is not magic, but a chain of checkable steps.

```python
trace = []

def call_tool(name, tool_input):
    if name == "search_docs":
        return "Found materials related to RAGOps, AgentOps, and evaluation sets"
    if name == "make_todo":
        return "Generated 3 review tasks"
    return "Tool does not exist"

step = {
    "thought": "The user wants to prepare for RAG review, so first search the course materials",
    "action": "search_docs",
    "input": {"query": "RAGOps evaluation logs"},
}
step["observation"] = call_tool(step["action"], step["input"])
trace.append(step)

for item in trace:
    print(item)
```

The acceptance criterion for this experiment is: even if the result is not perfect, you can still review why each step happened. Later, you can add tool schema, permission checks, failure retries, MCP, and multi-Agent support.

## Agent failure case library: control boundaries first, then pursue intelligence

| Phenomenon | Common cause | Debugging method | Fix direction |
|---|---|---|---|
| Agent keeps looping | Goal and stopping conditions are unclear | Check whether the thought and observation of each step repeat | Limit the maximum number of steps, add completion conditions and stopping conditions |
| Tool parameters are often wrong | Schema is too vague, and parameter validation is missing | Record the raw parameters generated by the model and the tool error messages | Define field types, examples, default values, and error messages clearly |
| Looks finished but cannot be reviewed | No execution trace | Require every step to record plan, action, observation, and result | Add trace logs and task summaries |
| Called a tool that should not have been called | Permission boundaries are unclear | Check the tool whitelist and risk levels | Require human confirmation for high-risk tools, and keep sensitive tools disabled by default |
| Multiple Agents wait on each other | Roles and the final owner are unclear | Review the message flow and task handoff points | Define roles, deliverables, timeouts, and an arbitrator |



| Review question | What you should be able to answer |
|---|---|
| Goal boundary | What should the Agent complete, what is it not responsible for, and when should it stop? |
| Plan execution | How does it break down tasks, choose the next step, and adjust based on results? |
| Tool calling | Are the tool schema, parameter validation, failure retry, and permission boundaries clear? |
| Memory state | Are current state, long-term preferences, and historical experience managed in layers? |
| Failure recovery | How does the system degrade when tools fail, retrieval returns nothing, or results are untrustworthy? |
| Evaluation and safety | Is there a fixed task set, cost estimation, safety boundaries, and a way for humans to take over? |

The real outcome of this stage is building a traceable, reviewable, and evaluable Agent project, not just showing one successful conversation.


## Stage deliverables

| Deliverable | Minimum version | Portfolio version |
|---|---|---|
| Tool schema | Define at least 1–2 tools | Clearly describe name, purpose, parameters, return values, errors, and boundaries |
| Agent trace | Print one execution step | Save `agent_traces.jsonl` and be able to replay goals, actions, observations, and decisions |
| Safety boundary | Set a maximum number of steps | Add a tool whitelist, risk levels, human confirmation, and audit logs |
| Failure samples | Record one tool failure | Cover wrong tool choice, parameter errors, loops, unauthorized access, and unsupported references |
| Evaluation task set | 3–5 fixed tasks | Track completion rate, average steps, tool error rate, and cost |
| README | Explain run commands and example output | Show architecture, trace examples, permission boundaries, evaluation, and limitations |


## Stage project

The basic version is to implement a research assistant that can break a topic into questions, call material tools, and generate structured summaries. The standard version should add execution logs, failure retries, result self-checks, and simple memory. The challenge version can be a data analysis Agent or a multi-Agent development team, with permission boundaries, cost control, evaluation samples, and recovery mechanisms.

If you want to run one concrete baseline before choosing a bigger project, start with [Hands-on: Build a Traceable Single-Agent Assistant](./ch10-projects/04-stage-hands-on-workshop.md). It gives you a runnable single-Agent loop, tool schema validation, permission blocking, JSONL trace logs, and evaluation cases.

If you want a more detailed learning rhythm, you can read [Study Guide: How to Learn Agent Systems Without Getting Confused](./study-guide.md).

## Relationship to the end-to-end AI learning assistant project

This stage can map to AI Learning Assistant v0.9: upgrading from Q&A to a learning-planning Agent that can break down tasks, search materials, call tools, and record execution traces. If you are following the end-to-end project path, it is recommended that by the end of this stage you submit at least one version note: what capabilities were added, how to run it, what the example inputs and outputs are, what problems were encountered, and what the next improvements will be.

## Stage completion standards

| Completion level | What you need to do |
|---|---|
| Minimum completion | Be able to design tool, memory, planning, MCP, framework, evaluation, and safety boundaries |
| Recommended completion | Finish at least one runnable mini-project for this stage and document the run steps, example inputs/outputs, and problems encountered in the README |
| Portfolio completion | Connect the stage output to the end-to-end “AI Learning Assistant” project, leaving screenshots, logs, evaluation samples, and next-step plans |

After finishing this stage, you do not need to memorize every detail. What matters more is being able to explain clearly: what problem this stage solves, how it relates to the previous stage, and how it supports later learning. After that, you can continue with multimodal Agents, deployment, and your graduation project.
