---
sidebar_position: 15
title: "Failure Samples Library for AI Applications"
description: "A compact failure sample index for LLM API, Prompt, RAG, Agent, safety, and deployment issues."
keywords: [AI failure samples, RAG debugging, Agent debugging, Prompt debugging, LLM applications]
---

# Failure Samples Library for AI Applications

![AI project debug index map](/img/course/appendix-quick-ref-debug-index-map-en.png)

A failure sample records one real input where the system did not behave as expected. It helps you debug and prevents the same issue from returning.

## Failure Layers

| Layer | Common symptom | Check first |
| --- | --- | --- |
| LLM API | timeout, rate limit, empty response, high cost | request_id, raw response, tokens, latency |
| Prompt/schema | invalid JSON, missing fields, label drift | schema, examples, parser, fixed tests |
| RAG | wrong source, weak citation, missed document | retrieved chunks, metadata, citation_ok |
| Agent/tool | wrong tool, bad parameters, loop, missing trace | tool schema, max steps, action/observation |
| Safety | over-permission, sensitive logs, unsafe action | allowlist, human confirmation, audit log |
| Deployment | works locally only, bad secrets, unstable runtime | `.env.example`, dependency versions, startup logs |

## Failure Sample Template

```md
## Failure Sample

User input:
Expected:
Actual:
Layer:
Related logs:
Likely cause:
Fix:
Regression test:
Resolved:
```

Keep at least three failure samples for each portfolio project. A good project does not hide failures; it shows how failures are located and repaired.
