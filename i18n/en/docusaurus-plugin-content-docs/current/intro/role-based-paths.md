---
sidebar_position: 12
title: "Choosing a Role-Based Path: Learn the AI Full-Stack by Your Goal"
description: "Different learning focuses designed for AI application engineers, RAG engineers, Agent developers, model engineering roles, and portfolio-based job seekers."
keywords: [AI application engineer, RAG engineer, Agent development, model engineering, AI portfolio path]
---

# Choosing a Role-Based Path: Learn the AI Full-Stack by Your Goal

![Role-based path selection map](/img/course/intro-role-based-paths-map.png)

## What This Section Is About

The same course should be read differently depending on your learning goal. New learners can follow lessons 1–9 in order; people with a clear career target can adjust how much they read in depth versus skim based on the role they want.

This page is not telling you to skip the fundamentals. It is helping you decide: which parts must be mastered, which parts only need to be understood at a high level for now, and which projects are best for a portfolio.

## Pick Your Role in 30 Seconds

| What do you most want to do | Priority path |
|---|---|
| Connect AI to products or business systems | AI application engineer path |
| Make models reliably read company docs, PDFs, and knowledge bases | RAG engineer path |
| Let AI break down tasks, use tools, and execute automatically | Agent developer path |
| Train, fine-tune, evaluate, and deploy models | Model engineering path |
| Prepare for job hunting, career switching, and showcasing skills | Portfolio job-seeker path |

## AI Application Engineer Path

The core skill of an AI application engineer is integrating models into real products: you can write backend APIs, call models, handle documents and user input, and add logging, evaluation, and deployment notes.

It is recommended to read lessons 1, 2, 3, 7, 8, and 9 carefully, and skim lessons 4–6 without skipping them entirely. Even if you do not train models, you still need to understand data, evaluation, vectors, Embedding, model cost, and failure analysis.

A representative project could be a course Q&A assistant with a login or simple frontend, supporting file upload, RAG retrieval, citation display, logging, and basic evaluation.

## RAG Engineer Path

The core skill of a RAG engineer is to reliably connect external knowledge to a large model. You need to understand document parsing, chunking, Embedding, vector databases, Hybrid Search, Reranking, Query Rewrite, GraphRAG, Multimodal RAG, and RAGOps.

It is recommended to read lessons 2, 3, 5, 7, and 8 carefully, and add Agentic RAG in lesson 9. The multimodal section in lesson 12 is also worth learning, because real knowledge bases often contain PDFs, tables, images, and screenshots.

A representative project could be an enterprise knowledge base demo with an evaluation set: it can show retrieved chunks, source citations, retrieval quality, answer faithfulness, failed examples, and optimization records.

## Agent Developer Path

The core skill of an Agent developer is not “letting the model think by itself,” but designing a controllable task execution system. You need to understand Prompt, tool calling, MCP, task planning, memory, workflows, multi-Agent systems, evaluation, safety, deployment, and AgentOps.

It is recommended to read lessons 7, 8, and 9 carefully, and also review the Python engineering skills in lesson 2 and the data processing skills in lesson 3. Agents often need to call code, databases, files, search, RAG, and external APIs, so basic engineering skills are very important.

A representative project could be a research assistant Agent: it can break down questions, retrieve materials, call tools, generate reports, record execution traces, and request human confirmation for high-risk steps.

## Model Engineering Path

The model engineering direction focuses more on model internals, training, fine-tuning, inference optimization, and deployment. You need a deeper understanding of mathematics, machine learning, deep learning, Transformer, fine-tuning, LoRA / QLoRA, quantization, distillation, small models, and hybrid deployment.

It is recommended to read lessons 4, 5, 6, and 7 carefully, then move on to the model deployment section in lesson 8. Lessons 10–12 can be chosen based on interest: vision, multimodal, or generative directions.

A representative project could be a model comparison experiment: compare different models, different quantization methods, and different Prompt or fine-tuning strategies in terms of performance, latency, and cost, and then write up an experiment report.

## Portfolio Job-Seeker Path

For the portfolio path, the most important thing is continuous output, not reading every chapter in maximum depth. At each stage, you should leave behind a project that can run, can be screenshotted, can be explained, and can be written into a README.

It is recommended to follow path one from lesson 1 to 9, while using the “end-to-end project: AI learning assistant growth path” to connect the results. For lessons 10–12, choose one direction as your capstone project; you do not need to go deep in every direction.

A representative portfolio could include: a Python utility, a data analysis report, an ML baseline, a deep learning experiment, a Prompt assistant, a RAG knowledge base, an Agent automation assistant, and a complete AI application deployment guide.

## Role Skill Radar

When choosing a role path, you can judge what you lack most across six dimensions: programming implementation, data processing, model understanding, AI application engineering, evaluation and review, and deployment delivery. Different roles emphasize different things, but no path can completely skip evaluation and engineering.

| Role | Must-have skills | Content you can skim first | Most important project evidence to add |
|---|---|---|---|
| AI application engineer | API, Prompt, RAG, backend, logging, deployment | Deep model training details | Runnable app, deployment notes, error handling |
| RAG engineer | document processing, retrieval, citations, evaluation, RAGOps | multi-Agent collaboration details | chunks, retrieval logs, eval questions |
| Agent developer | tool schemas, traces, permissions, failure recovery | advanced model training | tool logs, agent traces, safety boundaries |
| Model engineering path | mathematics, ML, DL, Transformer, fine-tuning | frontend presentation details | experiment logs, metric comparisons, training curves |
| Portfolio job seeker | README, demo, review, project communication | niche advanced topics | screenshots/GIFs, evaluation tables, failure examples, demo script |

## Minimum Project Set for Each Path

If you have limited time, you do not need to make every project large. Below is the minimum project set for different goals. After completing it, you can upgrade to the standard version based on your energy and time.

| Goal | Minimum project set | Upgrade direction |
|---|---|---|
| Switch to AI application development | Python API mini project + Prompt assistant + RAG Q&A + Agent tool calling | Add frontend, deployment, logging, and an evaluation set |
| Switch to RAG / knowledge base work | data cleaning project + document chunking demo + RAG Q&A + RAG evaluation report | Add hybrid search, rerank, and citation check |
| Switch to Agent / automation work | tool-calling demo + multi-step task Agent + safe confirmation examples | Add trace replay, MCP, and permission auditing |
| Go into model engineering | ML baseline + DL training experiment + Transformer/fine-tuning comparison | Add training diagnostics, cost analysis, and model deployment |
| Build a comprehensive portfolio | data analysis + ML baseline + RAG assistant + Agent assistant + capstone project | Unify README style and the project demo storyline |

These combinations are not about cutting the course content down; they are about prioritizing. The fundamentals decide whether you can debug on your own, the project sections decide whether you can demonstrate ability, and evaluation and review decide whether the project is trustworthy.

## How to Choose

If you are still unsure about your target, start with the AI application engineer path. It has the most stable coverage and is the easiest way to build projects. Once you discover that you like knowledge bases, Agents, model training, or multimodal creation more, switch to the corresponding path and go deeper.

No matter which path you choose, do not just collect tool names. For every technology you learn, you should be able to answer: what problem does it solve, when should you not use it, how do you build the smallest project with it, how do you evaluate it, and how do you write it into your portfolio.
