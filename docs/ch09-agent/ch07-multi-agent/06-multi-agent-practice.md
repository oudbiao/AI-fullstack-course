---
title: "9.7.7 Practice: Multi-Agent Collaboration System"
sidebar_position: 43
description: "Build a minimal multi-Agent collaboration system from task input, role division, state transitions, to result aggregation."
keywords: [multi-agent project, planner, retriever, writer, reviewer, workflow, collaboration]
---

# 9.7.7 Practice: Multi-Agent Collaboration System

![Multi-Agent collaboration practice run map](/img/course/ch09-multi-agent-collaboration-run-map-en.webp)

:::tip Section Overview
This section is the closing project for this chapter.
You have already learned:

- Architecture patterns
- Communication
- Task allocation
- Collaboration patterns
- Challenges and solutions

What we are going to do now is put all of these together and build a minimal but complete multi-Agent system.
:::

## Learning Objectives

- Build a minimal multi-Agent collaboration loop
- Learn how to let planner, retriever, writer, and reviewer each do their own job
- Understand how task state flows among multiple roles
- Understand what this project truly adds compared with a single-Agent system

---

## First, Define the Project Goal

We will build a minimal research-style multi-Agent system:

User input:

> "Please help me summarize the key conditions of the refund policy."

Internal system roles:

- Planner: break down the task
- Retriever: find information
- Writer: write the summary
- Reviewer: check the result

This task is a good fit because it can naturally be split into parts, and each role has a very clear responsibility.

---

## Prepare a Knowledge Base

```python
knowledge_base = {
    "refund policy": "You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.",
    "certificate policy": "You can receive a completion certificate after finishing all required projects and passing the test.",
    "learning sequence": "It is recommended to learn Python, data analysis, and machine learning first, then move on to deep learning and large models."
}

print(knowledge_base)
```

Expected output:

```text
{'refund policy': 'You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.', 'certificate policy': 'You can receive a completion certificate after finishing all required projects and passing the test.', 'learning sequence': 'It is recommended to learn Python, data analysis, and machine learning first, then move on to deep learning and large models.'}
```

This is the minimal knowledge source that the system will operate on.

---

## Define Four Agents

### Planner

```python
def planner_agent(user_query):
    if "refund" in user_query:
        return ["retrieve refund policy", "organize key conditions", "write summary", "review output"]
    return ["retrieve related materials", "write summary", "review output"]
```

### Retriever

```python
def retriever_agent(task):
    if "refund policy" in task:
        return knowledge_base["refund policy"]
    return "No materials found"
```

### Writer

```python
def writer_agent(evidence):
    return f"Summary: {evidence}"
```

### Reviewer

```python
def reviewer_agent(draft):
    if "7 days" in draft and "20%" in draft:
        return {"approved": True, "comment": "Key information is complete"}
    return {"approved": False, "comment": "Missing key conditions"}
```

---

## Connect Them Together

### A Minimal Multi-Agent Collaboration Flow

Run this in the same Python file or interpreter session after the knowledge base and four Agent functions above.

```python
def multi_agent_system(user_query):
    state = {
        "query": user_query,
        "plan": [],
        "evidence": None,
        "draft": None,
        "review": None
    }

    # 1. Planning
    state["plan"] = planner_agent(user_query)

    # 2. Retrieval
    state["evidence"] = retriever_agent(state["plan"][0])

    # 3. Writing
    state["draft"] = writer_agent(state["evidence"])

    # 4. Review
    state["review"] = reviewer_agent(state["draft"])

    return state

result = multi_agent_system("Please help me summarize the key conditions of the refund policy.")
for k, v in result.items():
    print(k, "->", v)
```

Expected output:

```text
query -> Please help me summarize the key conditions of the refund policy.
plan -> ['retrieve refund policy', 'organize key conditions', 'write summary', 'review output']
evidence -> You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.
draft -> Summary: You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.
review -> {'approved': True, 'comment': 'Key information is complete'}
```

### What Does This Code Already Show?

It already shows that:

- Multi-Agent is not just multiple functions
- The key is state transition
- Each role is only responsible for its own part

This is a true minimal multi-Agent system.

---

## Make the System More Like a Real Workflow

### What If the Reviewer Does Not Approve?

In a real system, if the review does not pass, the process usually should not end immediately.
A more reasonable approach is:

- Send the comment back to the writer
- Revise the output again

### A Small Example with Revision

Continue in the same file or session so `multi_agent_system` and the Agent functions are already defined.

```python
def reviser_agent(draft, review):
    if review["approved"]:
        return draft
    return draft + " Additional note: the refund also requires learning progress to be below 20%."

state = multi_agent_system("Please help me summarize the key conditions of the refund policy.")
final_output = reviser_agent(state["draft"], state["review"])

print("draft :", state["draft"])
print("review:", state["review"])
print("final :", final_output)
```

Expected output:

```text
draft : Summary: You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.
review: {'approved': True, 'comment': 'Key information is complete'}
final : Summary: You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.
```

This step is very important because it shows:

> The value of a multi-Agent system is not only division of labor, but also the ability for roles to form an iterative closed loop.

---

## Add Clearer Task Logs

### Why Must a Project Have Traces?

If the system gives the wrong answer, at least you need to know:

- How the planner broke down the task
- What the retriever found
- What the writer wrote
- Why the reviewer did not catch the problem

### A Minimal Trace Version

Continue in the same file or session so the four Agent functions are already defined.

```python
def traced_multi_agent_system(user_query):
    trace = []

    plan = planner_agent(user_query)
    trace.append({"agent": "planner", "output": plan})

    evidence = retriever_agent(plan[0])
    trace.append({"agent": "retriever", "output": evidence})

    draft = writer_agent(evidence)
    trace.append({"agent": "writer", "output": draft})

    review = reviewer_agent(draft)
    trace.append({"agent": "reviewer", "output": review})

    return trace

for step in traced_multi_agent_system("Please help me summarize the key conditions of the refund policy."):
    print(step)
```

Expected output:

```text
{'agent': 'planner', 'output': ['retrieve refund policy', 'organize key conditions', 'write summary', 'review output']}
{'agent': 'retriever', 'output': 'You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.'}
{'agent': 'writer', 'output': 'Summary: You can apply for a refund within 7 days after purchase, provided your learning progress is below 20%.'}
{'agent': 'reviewer', 'output': {'approved': True, 'comment': 'Key information is complete'}}
```

This trace is the important foundation for debugging and evaluating the system later.

---

## Why Is This System More Worth Learning Than a Single Agent?

### Because It Breaks the Problem Apart

A single Agent often does everything in one go:

- Understand the task
- Retrieve information
- Summarize
- Self-check

A multi-Agent system breaks these actions apart, which makes it easier for you to:

- Observe each layer
- Replace one layer
- Find out where an error happened

### But It Is Also More Expensive and More Complex

So the real engineering judgment is not:

> Multi-Agent is always more advanced

Instead, it is:

> Is this task worth paying extra complexity for better decomposability and controllability?

---

## How Can This Project Be Extended?

You can keep adding:

1. A more realistic retriever
2. Multi-task routing
3. Asynchronous communication
4. Conflict resolution
5. Retry on failure

If you keep expanding it, it will gradually become closer to a real multi-Agent product system.

---

## Common Mistakes Beginners Make

### Writing All Roles in Almost the Same Way

Then the result is just "multiple Agents with different names for the same thing."

### No Shared State or Trace

Once something goes wrong, it becomes very hard to debug.

### The Project Looks Busy, but Each Role Does Not Actually Have a Real Division of Labor

This is one of the most common problems in many multi-Agent demos.

---

## Summary

The most important thing in this section is not to write four functions, but to understand:

> **The core of a multi-Agent project is to let each role take different responsibilities around state transitions, and ultimately converge into an explainable and iterative workflow.**

That is where multi-Agent truly becomes more valuable than a single Agent.

---

## Exercises

1. Add a `fact_checker_agent` to this system to specifically verify numeric conditions.
2. Make `planner_agent` produce different plans for "certificate policy" as well.
3. Think about this: if the reviewer keeps rejecting the output, how should the system limit the number of revision rounds?
4. Explain in your own words: why is the real importance of a multi-Agent project "state transition" rather than "number of roles"?
