---
title: "9.6.1 Frameworks Roadmap: Choose Only When Needed"
sidebar_position: 0
description: "A concise hands-on roadmap for Agent frameworks: compare LangGraph, LlamaIndex, CrewAI, AutoGen, and choose based on state, data, roles, and risk."
keywords: [Agent Framework Guide, LangGraph, LlamaIndex, CrewAI, AutoGen]
---

# 9.6.1 Frameworks Roadmap: Choose Only When Needed

Frameworks do not make an Agent smarter. They organize state, tools, workflows, memory, logs, and collaboration once the task has enough complexity to justify the abstraction.

## See the Selection Map First

![Agent framework position map](/img/course/ch09-frameworks-position-map-en.png)

![Agent framework selection map](/img/course/ch09-framework-selection-map-en.png)

![Agent framework selection decision map](/img/course/ch09-framework-selection-decision-map-en.png)

If a task has three fixed steps, plain Python functions may be better. Add a framework when state, branching, recovery, data connection, or role collaboration becomes hard to manage.

## Run a Framework Route Check

Use this check before choosing a framework because it is popular.

```python
task = {
    "needs_state": True,
    "needs_rag": False,
    "needs_roles": False,
    "needs_resume": True,
}

if task["needs_state"] or task["needs_resume"]:
    route = "LangGraph-style state graph"
elif task["needs_rag"]:
    route = "LlamaIndex-style data app"
elif task["needs_roles"]:
    route = "CrewAI or AutoGen-style collaboration"
else:
    route = "plain functions first"

print("route:", route)
print("reason:", "choose the smallest abstraction that exposes state")
```

Expected output:

```text
route: LangGraph-style state graph
reason: choose the smallest abstraction that exposes state
```

Framework choice should be written into the README as a trade-off, not hidden inside dependencies.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Framework overview | Explain what a framework abstracts |
| 2 | LangChain / LangGraph | Model state, nodes, edges, branches, recovery |
| 3 | LlamaIndex | Connect documents, indexes, retrieval, evaluation |
| 4 | CrewAI / AutoGen | Compare role collaboration and multi-Agent conversation |
| 5 | Framework selection | Write a decision table and a no-framework baseline |

## Pass Check

You pass this chapter when you can implement the same small task with plain functions and with one framework, then explain which version is easier to debug and why.
