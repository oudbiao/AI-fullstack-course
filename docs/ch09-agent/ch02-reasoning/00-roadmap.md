---
title: "9.2.1 Reasoning Roadmap: Plan, Act, Check"
sidebar_position: 0
description: "A concise hands-on roadmap for Agent reasoning and planning: build intermediate steps, choose actions, monitor progress, and evaluate failures."
keywords: [Agent reasoning guide, ReAct, Plan-and-Execute, planning]
---

# 9.2.1 Reasoning Roadmap: Plan, Act, Check

Agent reasoning is not a longer answer. It is the ability to create usable intermediate steps, decide what to do next, and check whether the plan is still working.

## See the Planning Loop First

![Agent reasoning and planning chapter learning sequence diagram](/img/course/ch09-reasoning-chapter-flow-en.webp)

![Plan execute monitor replan map](/img/course/ch09-plan-execute-monitor-replan-map-en.webp)

![Reasoning state checkpoint map](/img/course/ch09-reasoning-state-checkpoint-map-en.webp)

The core habit is: plan a step, act, observe the result, checkpoint state, and replan when the result changes the situation.

## Run a Plan Checklist

Use explicit steps before adding tools. A plan you cannot print is hard to inspect.

```python
task = "prepare a cited RAG demo answer"
plan = ["inspect question", "retrieve sources", "draft answer", "check citations"]

print("task:", task)
for index, step in enumerate(plan, start=1):
    print(f"{index}. {step}")
print("checkpoint:", plan[-1])
```

Expected output:

```text
task: prepare a cited RAG demo answer
1. inspect question
2. retrieve sources
3. draft answer
4. check citations
checkpoint: check citations
```

Good planning is visible. It should make failures easier to locate, not hide them behind a final paragraph.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | LLM reasoning | Distinguish knowing an answer from deriving a path |
| 2 | Chain reasoning | Create intermediate states and self-check points |
| 3 | ReAct | Interleave thought, action, observation, and next step |
| 4 | Plan-and-Execute | Separate planning from execution when tasks grow |
| 5 | Advanced planning | Handle dependency, priority, rollback, and replan |
| 6 | Reasoning evaluation | Score final result, path quality, and failure type |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
task_goal: what the agent is trying to solve
plan_or_trace: reasoning steps, plan, ReAct trace, or execution graph
observation: what changed after each action
failure_check: hallucinated step, stale observation, loop, or unverified conclusion
eval_action: compare against expected result and revise the plan
```

## Pass Check

You pass this chapter when you can explain why a plan failed: bad decomposition, wrong tool choice, stale observation, missing checkpoint, or weak final verification.

The exit mini project is a visible reasoning trace for one task: plan steps, observations, replans, and the final answer.
