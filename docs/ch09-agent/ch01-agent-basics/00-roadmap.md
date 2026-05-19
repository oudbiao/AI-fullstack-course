---
title: "9.1.1 Agent Basics Roadmap: Goal, State, Action"
sidebar_position: 0
description: "A concise hands-on roadmap for Agent basics: distinguish agents from chatbots and workflows, then build the smallest goal-state-action loop."
keywords: [Agent guide, intelligent agent guide, Agent system architecture, tool calling, Agent loop]
---

# 9.1.1 Agent Basics Roadmap: Goal, State, Action

An Agent is not a model name. It is a system pattern that uses a model, tools, state, memory, and feedback to keep working toward a goal.

## See the Single-Agent Loop First

![Agent basics position bridging diagram](/img/course/ch09-basics-position-bridge-en.webp)

![Agent basics chapter learning order diagram](/img/course/ch09-basics-chapter-flow-en.webp)

![Single-Agent execution loop diagram](/img/course/ch09-basics-execution-loop-en.webp)

A normal chatbot answers once. A workflow follows fixed steps. An Agent can plan, act, observe, update state, and continue when the goal is not done.

## Run a Tiny Agent State Loop

This script does not call a model yet. It shows the minimum state you need before an Agent can be debugged.

```python
goal = "summarize RAG citation rules"
state = {"steps": [], "done": False}

for action in ["plan", "search_docs", "summarize"]:
    state["steps"].append(action)

state["done"] = True

print("goal:", goal)
print("steps:", " -> ".join(state["steps"]))
print("done:", state["done"])
```

Expected output:

```text
goal: summarize RAG citation rules
steps: plan -> search_docs -> summarize
done: True
```

If a demo cannot show goal, state, action, observation, and stop condition, call it an LLM app first, not an Agent.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | What is an Agent | Compare chatbot, workflow, RAG app, and Agent |
| 2 | Development history | Understand why LLMs revived Agent systems |
| 3 | Capability levels | Place answer, retrieve, tool use, plan, memory, collaboration on one ladder |
| 4 | System architecture | Draw goal, state, planner, tools, memory, observation, executor |
| 5 | RL to Agent breakthroughs | Connect action, reward, feedback, and planning |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
agent_boundary: how this differs from chatbot or fixed workflow
goal_state_action: goal, current state, next action, observation
architecture_parts: planner, tools, memory, guardrails, evaluator
failure_check: over-autonomy, vague goal, missing state, or no trace
next_action: build the smallest traceable single-agent loop
```

## Pass Check

You pass this chapter when you can draw one single-Agent loop and explain why single-Agent stability comes before multi-Agent collaboration.

The exit mini project is a research assistant Agent trace: one goal, one plan, at least one tool decision, one observation, one stop condition, and one final answer.
