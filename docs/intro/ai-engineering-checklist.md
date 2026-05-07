---
sidebar_position: 13
title: "AI Engineering Launch Checklist"
description: "A compact checklist for moving an LLM, RAG, Agent, or multimodal demo toward a reproducible and reviewable project."
keywords: [AI engineering, LLMOps, RAGOps, AgentOps, AI evaluation, launch checklist]
---

# AI Engineering Launch Checklist

![AI project quick reference map](/img/course/appendix-project-quick-reference-map-en.png)

A demo is not enough. Before calling a project “ready,” check whether it can be reproduced, evaluated, debugged, and bounded.

## Demo to Usable System

| Layer | Check |
| --- | --- |
| Problem boundary | User, input, output, non-goals, refusal conditions |
| Evaluation | Fixed questions/tasks, expected results, failure samples |
| Logs/traces | Request, retrieval, prompt version, tool calls, errors |
| Cost/latency | Token count, calls, retries, slow steps |
| Security | API keys, sensitive data, high-risk tools, human confirmation |
| Deployment | `.env.example`, startup command, clean-run notes |
| Portfolio | README, screenshots, evaluation, limitations, next steps |

## Project-Specific Checks

| Project type | Must show |
| --- | --- |
| RAG | document sources, chunking, retrieval logs, citation checks, no-answer handling |
| Agent | tool schema, max steps, stop condition, trace, permission boundary |
| Multimodal | input assets, generated versions, human review, export constraints |
| Deployment | environment variables, health/error logs, rollback or reset command |

## Final Launch Question

If the system answers incorrectly tomorrow, retrieves the wrong source, calls the wrong tool, or becomes too expensive, can you identify which layer failed?

If not, add logs, evaluation, and boundaries before adding more features.
