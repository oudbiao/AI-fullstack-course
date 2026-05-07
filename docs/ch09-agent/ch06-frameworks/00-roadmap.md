---
title: "9.6.1 Pre-Class Guide: What Exactly Will We Learn in the Agent Frameworks Chapter?"
sidebar_position: 0
description: "First build a learning map for the Agent frameworks chapter: how the framework overview, different framework styles, and selection logic work together to support real development decisions."
keywords: [Agent Framework Guide, LangGraph, LlamaIndex, CrewAI, AutoGen]
---

# 9.6.1 Pre-Class Guide: What Exactly Will We Learn in the Agent Frameworks Chapter?

## Chapter Focus

This chapter answers a simple question: with so many Agent frameworks on the market, what exactly does each one abstract, and how should you choose among them? Earlier, you have already learned about Agent goals, planning, tools, memory, MCP, and safety boundaries. If you jump straight into frameworks, it is very easy to fall into the trap of “learn whichever one is hottest.”

A framework is not magic. It simply abstracts away some repetitive engineering work. LangGraph emphasizes controllable state graphs and workflows, LlamaIndex emphasizes data and knowledge base applications, CrewAI emphasizes role-based collaboration, and AutoGen emphasizes multi-Agent conversational collaboration. Before choosing a framework, you first need to know whether your task is a fixed flow, a RAG application, open-ended exploration, or multi-role collaboration.

## Where This Chapter Fits in the Agent Learning Path

![Agent framework position map](/img/course/ch09-frameworks-position-map-en.png)

If you still cannot explain tool schemas, execution traces, stop conditions, and human confirmation, it is recommended that you do not rush into complex frameworks yet. A framework amplifies your design: when the boundaries are clear, it improves efficiency; when the boundaries are unclear, it makes problems harder to debug.

## What Different Frameworks Abstract

| Framework / Direction | Best-Suited Tasks | What You Should Focus On |
|---|---|---|
| LangGraph | Stateful, multi-step Agent workflows that need controllable branching and rollback | How state is defined, how nodes transition, and how recovery works after failures |
| LlamaIndex | Documents, knowledge bases, RAG, and data-connected applications | How data is connected, how indexes are built, and how retrieval and generation are evaluated |
| CrewAI | Multi-role collaboration, content production, research analysis, and workflow division | Whether roles are clear, whether task handoff is controllable, and who the final owner is |
| AutoGen | Multi-Agent conversations, code collaboration, and experimental automation | When the conversation stops, how tool permissions are constrained, and how loops are avoided |
| Low-code / platform tools | Rapid prototyping, business process demos, and collaboration with non-engineering teams | Observability, portability, version management, and deployment boundaries |

This table is not a ranking; it is a selection map. In real projects, you can also combine them. For example, you can use LlamaIndex to manage the knowledge base, LangGraph to orchestrate the workflow, MCP to connect tools, and your own logging system for evaluation and tracking.

## Learning Order for This Chapter

First, read the framework overview to understand why frameworks exist: not to make Agent “smarter,” but to make state, tools, workflows, memory, and logs easier to organize. Second, study LangChain/LangGraph, focusing on state graphs, nodes, edges, conditional branches, and resumable execution. Third, look at LlamaIndex and understand why it is closer to a “data application framework.” Fourth, study CrewAI and AutoGen to compare the strengths and weaknesses of role-based collaboration and multi-Agent conversations. Finally, read about framework selection and build your own decision table.

![Agent framework selection map](/img/course/ch09-framework-selection-map-en.png)

## When You Should Not Use a Framework

If the task only has three fixed steps, a regular function or workflow may be more stable than an Agent framework. If it is just a course Q&A system, basic RAG plus an evaluation set may be easier to ship than a multi-Agent system. If the task involves deleting files, sending messages, modifying databases, or making payments, you should first design permissions, confirmations, and rollback mechanisms rather than choosing a framework first.

A framework is suitable when complexity has already appeared: more and more state, more and more tools, workflows that need branching, failures that need recovery, execution traces that need to be saved, and multi-role collaboration that needs constraints. Do not create complexity just for the sake of using a framework.

## Small Project Exit for This Chapter

It is recommended that you do a small experiment: “two implementations of the same task.” The task can be “generate a one-week review plan based on course materials.” The basic version uses ordinary Python functions to implement a fixed flow: read the goal, look up materials, generate the plan, and output a checklist. The standard version uses LangGraph or a similar framework to implement the same flow, while recording state, nodes, and execution traces. For the challenge version, add a knowledge base retrieval node or a human confirmation node.

The project README should answer: why this task does or does not need a framework, what is stored in the state, what permissions the tools have, how to stop on failure, and whether the added complexity is worth it compared with the version without a framework.

## Framework Selection Self-Check List

Before choosing a framework, ask these five questions: does the task truly require multi-step state; does it need access to external data or a knowledge base; does it need multi-role collaboration; does it need resumable execution and logging; and are there high-risk actions that require human confirmation? The more answers are “yes,” the more worthwhile it is to introduce a framework. The more answers are “no,” the more you should keep things simple.

## Passing Criteria

After finishing this chapter, you should be able to explain the focus of LangGraph, LlamaIndex, CrewAI, and AutoGen; judge whether a task is worth introducing a framework for; build a minimal controllable Agent workflow; and clearly describe in the README the benefits, costs, and risk boundaries brought by the framework.
