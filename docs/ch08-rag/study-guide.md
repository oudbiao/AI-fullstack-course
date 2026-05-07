---
title: "8.0 Study Guide and Task Sheet: How to Learn LLM Application Development and RAG Without Getting Confused"
sidebar_position: 1
description: "A learning guide for AI full-stack beginners on LLM applications: RAG, model deployment, application development, engineering practices, project roadmap, and acceptance criteria."
keywords: [LLM application study guide, how to learn RAG, how to learn vector databases, how to learn LangChain, large model engineering]
---

# 8.0 Study Guide and Task Sheet: How to Learn LLM Application Development and RAG Without Getting Confused

If you reach `8 LLM Application Development and RAG` and feel that frameworks, databases, deployment, and application logic are all mixed together, first separate the system into layers. An LLM application is not just calling an API. It is about connecting knowledge, models, applications, and engineering.

## Overall Principles for This Stage

For LLM applications, the first thing to understand is the four layers: the knowledge layer is responsible for bringing data into the system, the model layer is responsible for generation and understanding, the application layer is responsible for organizing features, and the engineering layer is responsible for stable operation.

![LLM application four-layer learning map](/img/course/ch08-study-guide-four-layer-map-en.png)

## Tasks You Must Complete in This Stage

Use these tasks to keep RAG practical. A RAG project is not complete when it returns an answer; it is complete when you can inspect the retrieved evidence, cite sources, evaluate failures, and rerun the system.

| Task | Deliverable | Passing Criteria |
|---|---|---|
| Get an LLM API call working | A minimal calling script | Can handle keys, requests, responses, exceptions, and retries |
| Complete document processing | A chunking and cleaning record | Can explain chunk size, overlap, metadata, and source |
| Build vector retrieval | A minimal retrieval demo | Can retrieve relevant document fragments from a question |
| Complete RAG Q&A | A course Q&A prototype | Answers include source citations and can explain the retrieval basis |
| Complete RAG evaluation | A test set and evaluation table | Can record hit rate, answer quality, citation quality, and failure samples |
| Complete the hands-on workshop | `rag_app_workshop.py` output and one added eval case | Can reproduce the expected output, explain each stage, and extend it safely |

## Recommended Learning Order

In the first round, learn RAG first. You need to understand how documents are parsed, chunked, vectorized, retrieved, reranked, and then passed into the model context.

In the second round, learn model deployment and unified interfaces. You need to understand the differences between cloud APIs, local models, inference services, and a unified calling layer.

In the third round, learn application development. This includes LLM API, Function Calling, dialog systems, document parsing, template generation, and AI-assisted coding.

In the fourth round, learn engineering practices. Async, API design, logging, monitoring, error handling, Docker, and deployment should be added gradually after you have a minimal working application.

In the fifth round, build a comprehensive project that connects the knowledge base, model calls, application features, and engineering practices.

## Decision Map: Prompt, RAG, Fine-tuning, or Agent?

Beginners often try to solve every problem with a single technique. A better rule is to first ask what kind of gap you are trying to close.

| Problem type | First choice | Why |
|---|---|---|
| Need to express a task clearly or constrain the output format | Prompt | The model already knows the task; it mainly needs better instructions |
| Need fresh, citable, or organization-specific knowledge | RAG | The answer should come from external documents, not model memory |
| Need stable behavior, tone, or formatting across many examples | Fine-tuning | The pattern should be learned into the model itself |
| Need repeated goal-driven actions, tool use, and state updates | Agent | The system must decide the next action after each observation |

A simple RAG failure diagnosis ladder is:

1. Did the documents get parsed and chunked well?
2. Did retrieval actually find the right pieces?
3. Did reranking or context packing drop the evidence?
4. Did the model ignore the evidence when answering?
5. Did the evaluation data reflect the real user task?

If the answer is still vague after all five checks, the issue is usually not “RAG is broken.” More often, the current problem should first be solved by prompt design, data cleanup, or a narrower task scope.

![Prompt, RAG, Fine-tune, and Agent decision map](/img/course/ch08-study-guide-method-choice-map-en.png)

The safest rule is to start with the simplest tool that closes the gap, then upgrade only when the simpler choice cannot solve the real problem.

## Suggested Learning Pace

| Content Type | Suggested Time | Learning Goal |
|---|---|---|
| RAG basics | 8–16 hours | Run through the full path from documents to answers |
| Model deployment | 4–8 hours | Understand how models are called reliably |
| Application development | 8–16 hours | Be able to wrap chat, tools, and structured outputs |
| Engineering practices | 8–16 hours | Be able to add logging, error handling, and deployment notes |
| Comprehensive project | 16–32 hours | Complete an LLM application that can be showcased |

## Project Roadmap for Each Stage

For your first project, it is recommended to build a minimal RAG system: prepare a few documents, chunk them, vectorize them, retrieve them, and let the model answer based on the materials.

Before expanding the project, complete [8.5.6 Hands-on: Full Chapter 8 RAG App Workshop](./ch05-projects/05-stage-hands-on-workshop.md). It is the recommended practical bridge between reading concepts and building a portfolio knowledge base assistant.

For your second project, it is recommended to build a course Q&A assistant or a personal knowledge base, adding source citations, retrieval result display, and feedback records.

For your third project, it is recommended to build an enterprise knowledge base demo, adding multi-turn dialogue, permission design ideas, logs, and evaluation examples.

## Common Sticking Points

The most common mistake is thinking that connecting a vector database means RAG is finished. What really affects the results is document quality, chunking strategy, retrieval recall, reranking, prompt organization, and evaluation data.

The second sticking point is chasing frameworks right away. It is recommended to hand-write a minimal RAG system first, and only then learn frameworks such as LangChain and LlamaIndex.

The third sticking point is only building successful demos and not handling failures. Real applications must consider no-answer cases, retrieval failures, model timeouts, output format errors, and excessive costs.

## Stage Portfolio Deliverables

![RAG evaluation loop map](/img/course/ch08-rag-evaluation-loop-map-v2-en.png)

If you want this stage to become portfolio material, keep at least these files or equivalent evidence.

| Deliverable | Description |
|---|---|
| `chunks.jsonl` | Document chunking results, including text, source, section, page, and content type |
| `retrieval_logs.jsonl` | top-k, score, source, and retrieved text summaries for each query |
| `eval_questions.csv` | Fixed evaluation questions, ground-truth answers, expected documents to hit, and key citations |
| `failure_cases.md` | Retrieval failures, generation failures, citation failures, metadata failures, and similar samples |
| `rag_config.md` | Configuration records such as chunk size, overlap, top-k, rerank, and prompt version |
| `rag_app_workshop_output.txt` | Output from [8.5.6 Hands-on: Full Chapter 8 RAG App Workshop](./ch05-projects/05-stage-hands-on-workshop.md), plus notes about one change you made |
| `README.md` | Run commands, example inputs and outputs, evaluation results, and improvement plans |

These files turn your RAG project from “can answer” into “can explain, can evaluate, and can review.”

## Stage Completion Questions

After finishing this stage, you should be able to independently complete a RAG application, explain the full path from documents to answers, and identify which layer may be causing poor results.

Before moving to Chapter 9, check that you can answer these questions:

- What limitation of large models does RAG solve?
- What does chunk size affect?
- What is the difference between embedding retrieval and keyword retrieval?
- Why must answers have citations?
- How do you determine whether a RAG failure is a retrieval problem or a generation problem?

If you can run the hands-on workshop, add one document and one evaluation case, and then build a knowledge base assistant with source citations, basic logs, and simple evaluation examples, you are ready to move on to the AI Agent stage.
