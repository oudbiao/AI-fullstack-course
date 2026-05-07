---
title: "9 AI Agent and Intelligent Agent Systems"
sidebar_position: 0
description: "Build a traceable Agent loop with goals, plans, tools, observations, memory, safety boundaries, evaluation, and deployment awareness."
keywords: [AI Agent, agent, Function Calling, ReAct, MCP, Multi-Agent, tool calling]
---

# 9 AI Agent and Intelligent Agent Systems

![Main visual of the AI Agent system](/img/course/ch09-agent-systems-en.png)

Chapter 8 made the model answer from documents. Chapter 9 makes the system **act toward a goal**: plan a next step, call a tool, read the observation, adjust, stop safely, and leave a trace that people can review.

Do not start with multi-agent frameworks. Start with one small Agent that can show every step.

## 9.0.1 See the Agent Execution Loop

![Agent execution loop](/img/course/ch09-agent-execution-loop-en.svg)

An Agent is not "a chatbot with tools." It is a controlled execution loop.

| Part | Plain meaning | What you must control |
|---|---|---|
| Goal | What the Agent is trying to finish | scope, success criteria, stop condition |
| State | What is known right now | current inputs, previous observations, remaining steps |
| Plan | What to try next | step limit, fallback path, human takeover |
| Tool | External action such as search, file read, API call, code run | schema, validation, whitelist, risk level |
| Observation | What the tool returned | error handling, retry rule, trust boundary |
| Memory | What should persist across steps or runs | short-term state versus long-term preference |
| Trace | The replayable record of the run | goal, action, arguments, observation, cost, final result |

## 9.0.2 Learning Order And Task List

Build a single traceable Agent before multi-agent systems.

| Step | Read | Do | Evidence to keep |
|---|---|---|---|
| 9.1 | Agent basics and architecture | Explain goal, state, plan, tool, observation, memory | one architecture sketch |
| 9.2 | Reasoning and planning | Compare ReAct and Plan-and-Execute on one task | a step-by-step trace |
| 9.3 | Tool calling | Define one or two tools with parameters and errors | `tools_schema.md` |
| 9.4 | Memory | Separate current state from long-term memory | memory boundary notes |
| 9.5 | MCP | Understand MCP as a standard way to connect tools and data sources | one integration note |
| 9.6-9.7 | Frameworks and multi-agent | Study only after the single-Agent loop is stable | framework choice note |
| 9.8-9.10 | Evaluation, safety, deployment, project | Run [9.10.5 Hands-on: Build a Traceable Single-Agent Assistant](./ch10-projects/04-stage-hands-on-workshop.md) | trace logs, safety block, eval cases |

## 9.0.3 First Runnable Loop: Print the Trace

This offline script has no LLM dependency. It teaches the engineering habit: every action must be replayable. Later, replace the fixed `plan` with a model-generated plan, but keep the trace format.

Create `ch09_agent_trace.py` and run it with Python 3.10 or later.

```python
import json


def search_docs(tool_input: dict) -> str:
    return "Found notes about RAGOps, AgentOps, evaluation sets, and trace logs."


def make_todo(tool_input: dict) -> str:
    topic = tool_input["topic"]
    return f"1) Review {topic} notes; 2) add one eval case; 3) write failure notes."


TOOLS = {
    "search_docs": {"fn": search_docs, "risk": "read_only"},
    "make_todo": {"fn": make_todo, "risk": "draft_only"},
}

goal = "Prepare a short RAG review plan."
plan = [
    {
        "thought": "Find relevant course materials before making a plan.",
        "action": "search_docs",
        "input": {"query": "RAGOps AgentOps evaluation trace"},
    },
    {
        "thought": "Turn the materials into a small review checklist.",
        "action": "make_todo",
        "input": {"topic": "RAG evaluation"},
    },
]

trace = []
for step_number, step in enumerate(plan, start=1):
    tool = TOOLS.get(step["action"])
    if tool is None:
        observation = "Blocked: tool is not whitelisted."
        risk = "blocked"
    else:
        observation = tool["fn"](step["input"])
        risk = tool["risk"]

    trace.append(
        {
            "step": step_number,
            "goal": goal,
            "thought": step["thought"],
            "action": step["action"],
            "input": step["input"],
            "risk": risk,
            "observation": observation,
        }
    )

for item in trace:
    print(json.dumps(item, ensure_ascii=False))
```

Expected output starts like this:

```text
{"step": 1, "goal": "Prepare a short RAG review plan.", "thought": "Find relevant course materials before making a plan.", "action": "search_docs", ...
{"step": 2, "goal": "Prepare a short RAG review plan.", "thought": "Turn the materials into a small review checklist.", "action": "make_todo", ...
```

Operation tip: change `make_todo` to a non-whitelisted tool name such as `send_email`. The script should block it. This is the smallest version of a safety boundary.

## 9.0.4 Choose Agent, Workflow, RAG, Or Function Calling

![Agent boundary map](/img/course/ch09-agent-boundary-map-en.svg)

Agents are powerful, but they are not the default solution.

| Problem | Start with | Use an Agent when |
|---|---|---|
| Steps are fixed and known | Workflow | the route must change after each observation |
| Answer needs private or fresh knowledge | RAG | retrieval is only one step inside a larger goal |
| One structured action is enough | Function Calling | multiple tool calls and state updates are required |
| Task is high risk | Workflow with human approval | the Agent can draft, but humans must confirm risky actions |
| Exploration needs planning, tools, memory, and recovery | Agent | you can log every step and stop safely |

## 9.0.5 Common Failures

- Building multi-agent before a single Agent is stable.
- Calling tools without schema, validation, or useful error messages.
- Missing stop conditions, which causes loops and cost spikes.
- Letting high-risk tools run without human confirmation.
- Showing only a successful demo while hiding failed traces.
- Using memory as a dumping ground instead of separating current state, long-term preference, and task history.

## 9.0.6 Pass Check

Before leaving this chapter, you should be able to:

- explain goal, state, plan, tool, observation, memory, trace, and guardrail;
- run the trace script and block a non-whitelisted tool;
- save `agent_traces.jsonl`, `tools_schema.md`, `safety_boundary.md`, and `failure_cases.md`;
- judge whether a task needs workflow, RAG, function calling, or an Agent;
- run the full Chapter 9 workshop and add one evaluation task plus one safety-block example.

For a printable checklist, use [9.0 Learning Checklist](./study-guide.md). For the guided project, start with [9.10.5 Hands-on: Build a Traceable Single-Agent Assistant](./ch10-projects/04-stage-hands-on-workshop.md).
