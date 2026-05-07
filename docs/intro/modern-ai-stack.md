---
sidebar_position: 7
title: "Modern AI Application Stack"
description: "A compact visual map of the 2025-2026 AI application stack: RAG, Agent, multimodal AI, model engineering, evaluation, observability, and deployment."
keywords: [modern AI applications, RAGOps, AgentOps, MCP, LLMOps, multimodal AI, AI engineering]
---

# Modern AI Application Stack

![Modern AI application technology stack overview](/img/course/intro-modern-ai-stack-map-en.png)

The second half of the course is not just about calling an LLM. It is about building systems that can use knowledge, take actions, handle real-world inputs, be evaluated, and keep running.

## 1. Remember the Five Blocks

| Block | Problem it solves | Main stage |
|---|---|---|
| RAG | The model needs private or updated knowledge | 8 |
| Agent | The task needs planning, tools, and multi-step execution | 9 |
| Multimodal AI | Inputs or outputs include images, PDFs, audio, video, or screenshots | 10-12 |
| Model engineering | The system must balance quality, latency, cost, and privacy | 6-8 |
| Ops | Prompts, evals, logs, traces, deployment, and rollback must be managed | 7-9 and capstone |

## 2. Minimal System View

| Layer | What to inspect |
|---|---|
| Input | User task, files, images, permissions |
| Context | documents, chunks, metadata, conversation state |
| Model or tool | model call, tool call, parameters, cost |
| Output | answer, citation, action, generated asset |
| Evaluation | fixed tests, human review, metrics |
| Observability | logs, traces, errors, latency |
| Improvement | changed prompt, retrieval, tool schema, model, or workflow |

If a project only has input and output, it is still a demo. If it can explain what data it used, what tool it called, why it failed, and how it improved, it is becoming engineering work.

## 3. When to Use What

| Need | Start with |
|---|---|
| Answer from documents | RAG with visible retrieval logs |
| Execute actions safely | Workflow first, Agent only when steps are uncertain |
| Understand screenshots or PDFs | Multimodal model plus source tracking |
| Reduce cost | smaller model, routing, caching, or shorter context |
| Keep quality stable | eval set, logs, traces, and versioned prompts |

First pass advice: remember the map, then learn each block through the smallest runnable project.
