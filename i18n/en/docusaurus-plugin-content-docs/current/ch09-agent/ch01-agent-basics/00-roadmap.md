---
title: "1.1 Pre-Class Guide: What Exactly Will You Learn in the Agent Basics Chapter?"
sidebar_position: 0
description: "First build a learning map for the Agent basics chapter: how Agent boundaries, goals, state, tools, memory, planning, and the system feedback loop fit together into the first smart-agent map."
keywords: [Agent guide, intelligent agent guide, Agent system architecture, tool calling, Agent loop]
---

# Pre-Class Guide: What Exactly Will You Learn in the Agent Basics Chapter?

This chapter is about clarifying the boundaries, capabilities, and system architecture of an Agent.

When many beginners first learn about Agent, they mix chatbots, workflows, RAG, tool calling, and multi-Agent together. Learning this way is easy to get carried along by frameworks and demos. The job of the Agent basics chapter is to first establish the smallest closed loop: goals, planning, action, observation, correction, and finally outputting results.

## Where This Chapter Fits in the Course

You have already learned LLM application development and RAG, and you know that large models can connect to documents, APIs, conversations, and content generation. At the Agent stage, the course begins to move from “model applications” to “systems that can keep acting toward a goal.”

The key change here is: a normal LLM application usually works as one user question, one system answer; an Agent places more emphasis on goal-driven behavior, state maintenance, tool calling, result observation, and multi-step execution.

![Agent basics position bridging diagram](/img/course/ch09-basics-position-bridge.png)

## The Real Problems This Chapter Solves

This chapter answers five questions: What is the difference between an Agent and a normal chatbot; what is the difference between an Agent and a fixed workflow; what roles do goals, state, tools, memory, and planning each play; why should a single Agent be made stable first before considering multiple Agents; and how does an Agent system record the process, handle failures, and avoid infinite loops.

The mistake beginners make most often is jumping straight into LangGraph, CrewAI, AutoGen, or multi-Agent collaboration without first understanding the execution loop of a single Agent. Frameworks can improve development efficiency, but they cannot replace your understanding of system boundaries.

## Recommended Learning Order for Beginners

It is recommended to first look at “What is an Agent” and clearly distinguish the boundaries between Agents, chatbots, RAG applications, tool-calling systems, and fixed workflows. Then study the development path to understand why Agent regained attention after LLMs. Next, look at the capability levels and place “can answer, can retrieve, can call tools, can plan, can use memory, can collaborate” on the same capability line. After that, study the system architecture to understand how goals, state, tools, memory, planners, and executors work together. Finally, read “From TD-Gammon to AlphaGo” to connect action, feedback, and planning in reinforcement learning with modern Agents.

![Agent basics chapter learning order diagram](/img/course/ch09-basics-chapter-flow.png)

## The Main Thread to Focus on While Studying This Chapter

The main thread of this chapter can be summarized as: Agent is not a model name, but a system approach that organizes models, tools, state, and feedback around a goal.

![Single-Agent execution loop diagram](/img/course/ch09-basics-execution-loop.png)

This loop helps you judge whether a system is an Agent. If the system only calls a model once in a fixed way, it is more like a normal LLM application. If the system can break a goal into steps, call tools, observe results, revise the plan, and continue execution when needed, it starts to resemble an Agent.

## The Relationship Between This Chapter and Later Chapters

This chapter is the entry point to Chapter 9, AI Agents and Intelligent Agent Systems. Later chapters on reasoning and planning will expand on how an Agent decides the next step; tool calling will explain how it connects to external capabilities; memory systems will explain how it stores context and experience; MCP will cover the tool ecosystem and protocols; multi-Agent will discuss how multiple roles collaborate; and evaluation, safety, and deployment will explain how to run Agents reliably.

If this chapter is not learned solidly, common problems later on are: assuming multi-Agent is always stronger than single-Agent; only demonstrating successful paths and not handling tool failures or looped execution; treating memory as “the more, the smarter,” without understanding that memory should serve the task; and learning many frameworks without knowing why the system is designed this way.

## How Beginners and Advanced Learners Should Read This Chapter

When beginners study this chapter for the first time, focus on the main thread and the smallest runnable example. You do not need to understand every detail at once. As long as you can clearly explain what problem this chapter solves, what the input and output are, and how the minimum project runs, you can continue forward.

Experienced learners can use this chapter as a chance to fill gaps and practice engineering: pay attention to boundary conditions, failure cases, evaluation methods, code reproducibility, and how it connects to the earlier and later stages. After reading, it is best to store the chapter content in your own project README or experiment notes.

## Suggested Study Time and Difficulty

| Study style | Suggested time | Goal |
|---|---|---|
| Quick overview | 20–30 minutes | Understand what problem this chapter solves and where it will be used later |
| Minimum completion | 1–2 hours | Run a minimum example and complete the chapter’s small project output |
| In-depth practice | Half a day to 1 day | Add error analysis, comparison experiments, or project README notes |

## Self-Check Questions for This Chapter

| Self-check question | Passing standard |
|---|---|
| What problem does this chapter solve? | You can explain its place in the whole course in one sentence |
| What are the minimum input and output? | You can clearly describe what input the example needs and what result it produces |
| Where are the common failure points? | You can list at least one reason for an error, poor result, or misunderstanding |
| What can you retain after learning it? | You can write this chapter’s output into a project README, experiment notes, or portfolio |

## Small Project Output for This Chapter

After finishing this chapter, it is recommended to build a minimum research assistant Agent. The user enters a topic, the Agent first breaks down the question, then decides whether to retrieve information or call tools, then organizes the observation results, and finally outputs a short summary with step records.

The focus of this project is not how rich the search results are, but how clearly the execution process is recorded: what the goal is, what the plan is, what tools were called, what was observed, when the system decided to continue, and when it decided to stop.

## Passing Criteria

By the end of this chapter, you should be able to explain the differences between an Agent and a chatbot, a RAG application, and a fixed workflow; you should be able to explain the roles of goals, state, tools, memory, planning, and observation in an Agent; and you should be able to draw the execution loop of a single Agent.

If you can build a single-Agent demo and record each tool call, observed result, failure handling step, and final output, then you are ready to move on to the chapters on reasoning, tools, memory, MCP, and multi-Agent.
