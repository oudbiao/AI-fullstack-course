---
title: "8.5.1 Project Roadmap: Build a Cited Knowledge Assistant"
description: "A concise hands-on roadmap for the Chapter 8 capstone: build a cited RAG or LLM application with retrieval logs, failure handling, evaluation, and deployment notes."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "LLM project guide, enterprise knowledge base, intelligent assistant, RAG project, SOP document assistant"
---
This capstone proves you can connect knowledge, model calls, application flow, and engineering evidence into one reproducible LLM application.

## See the Project Evidence Loop First

![LLM application capstone project roadmap](/img/course/ch08-projects-route-map-en.webp)

![LLM application project learning order diagram](/img/course/ch08-project-learning-order-map-en.webp)

![LLM application project delivery loop diagram](/img/course/ch08-project-delivery-loop-en.webp)

The project is not “connect a vector database.” It is a traceable loop: documents, chunks, retrieval, context, answer, citations, logs, evaluation, and improvement.

## Run a Project Readiness Check

Use this checklist before calling the project done.

```python
project = {
    "project_type": "knowledge-base assistant",
    "documents": 5,
    "eval_questions": 10,
    "citations": True,
    "empty_retrieval_handled": True,
    "failure_cases": 3,
}

ready = (
    project["documents"] >= 3
    and project["eval_questions"] >= 10
    and project["citations"]
    and project["empty_retrieval_handled"]
    and project["failure_cases"] >= 1
)

print("ready:", ready)
print("project_type:", project["project_type"])
print("evidence:", "docs, eval, citations, failures")
```

Expected output:

```text
ready: True
project_type: knowledge-base assistant
evidence: docs, eval, citations, failures
```

If `ready` is `False`, do not add another feature yet. Complete the evidence loop first.

## Learn in This Order

| Step | Project | What It Trains |
|---|---|---|
| 1 | Enterprise or course knowledge base | Retrieval, permissions, citations, traceable answers |
| 2 | Intelligent assistant | Retrieval, session state, and tool calling as product features |
| 3 | RAG + finetuning system | Separate missing knowledge from unstable behavior |
| 4 | SOP document assistant | Document parsing, structured output, and template rendering |
| 5 | Full hands-on workshop | A minimum reproducible loop before adding real APIs or databases |

If you need a guided baseline, start with [8.5.6 Hands-on: Full Chapter 8 RAG App Workshop](./05-stage-hands-on-workshop.md).

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
project_goal: user task and business boundary
baseline: simplest prompt/RAG/app version first
evaluation: fixed cases, retrieval evidence, answer quality, and citation check
failure_log: at least one failed case with likely cause
deliverable: README, run command, screenshots/logs, next step
```

## Project Deliverable Standards

| Deliverable | Minimum Requirement | Stronger Portfolio Version |
|---|---|---|
| README | Goal, run command, dependencies, and examples | Add architecture diagram, design trade-offs, cost, and retrospective |
| Knowledge base sample | Raw documents, chunks, metadata, and source fields | Add permission rules, document version, and update notes |
| Retrieval logs | Matched passages, scores, and ranking | Add failure-type statistics and before/after comparison |
| Answer citations | Final answers show supporting sources | Add citation faithfulness checks |
| Failure cases | At least one documented failure | Add 3 or more cases with cause, fix, and regression check |
| Evaluation | Fixed questions with pass/fail rules | Add baseline, metrics, and regression testing |
| Deployment note | How to run and required environment variables | Add Docker, monitoring, and fallback notes |

## Pass Check

You pass this chapter when the project can answer with citations, show retrieval logs, handle empty retrieval, keep evaluation cases, and explain at least one failure.

The strongest portfolio version is not the largest one. It is the version where another developer can reproduce the run, inspect the evidence, and understand how you would improve the next iteration.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer traces the full path from query to chunks, retrieval scores, cited evidence, answer, and fallback behavior.
2. The evidence should include retrieved passages, source metadata, a cited answer, and at least one empty-retrieval or wrong-retrieval case.
3. A good self-check explains whether a failure came from chunking, retrieval, ranking, prompt assembly, missing sources, or unsupported generation.

</details>
