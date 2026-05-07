---
title: "8 LLM Application Development and RAG"
sidebar_position: 0
description: "Build a practical RAG application loop: documents, chunks, retrieval, citations, evaluation, API wrapping, and engineering logs."
keywords: [LLM applications, RAG, Prompt Engineering, LangChain, vector databases, large model deployment]
---

# 8 LLM Application Development and RAG

![Main visual for LLM applications and RAG](/img/course/ch08-rag-engineering-en.png)

Chapter 7 explained how an LLM produces text. Chapter 8 turns that model into a useful application: **connect documents, retrieve evidence, answer with citations, log failures, and improve with an evaluation set.**

Think of RAG as "read before answering." The model should not guess from memory when the answer must come from your course notes, company documents, product manuals, or private knowledge base.

## 8.0.1 See the RAG Application Loop

![RAG application loop](/img/course/ch08-rag-app-loop-en.svg)

Use this loop as the chapter map.

| Layer | Job | What you should print or save |
|---|---|---|
| Knowledge | Parse documents, clean text, split chunks, keep metadata | `chunks.jsonl`, source, section, page, version |
| Retrieval | Find the chunks most relevant to a question | query, top-k chunks, scores, source IDs |
| Generation | Ask the LLM to answer only from retrieved context | final prompt, answer, citations, no-answer reason |
| Application | Wrap the flow as CLI, API, chat UI, or internal tool | request, response, error handling, user feedback |
| Operations | Compare quality, cost, latency, and failures over time | eval set, logs, token cost, latency, failure cases |

## 8.0.2 Learning Order And Task List

Do the workshop after the basics. First make the retrieval chain visible; then replace simple parts with stronger components.

| Step | Read | Do | Evidence to keep |
|---|---|---|---|
| 8.1 | RAG basics, document processing, retrieval, evaluation | Build a tiny document-to-answer loop | chunks, top-k output, cited answer |
| 8.2 | Deployment and unified APIs | Understand cloud API, local model, and unified calling layer | one calling note or config comparison |
| 8.3 | LLM app development | Wrap the RAG loop with API, tools, dialog, or document parsing | request/response sample and error path |
| 8.4 | Engineering practices | Add async, logging, monitoring, API design, or Docker notes | logs, config, deployment checklist |
| 8.5 | Stage project | Run [8.5.6 Hands-on: Full Chapter 8 RAG App Workshop](./ch05-projects/05-stage-hands-on-workshop.md) | workshop output, one added doc, one added eval case |

## 8.0.3 First Runnable Loop: Tiny RAG Without a Framework

Before LangChain, LlamaIndex, or a vector database, run the smallest possible chain. The goal is not a powerful retriever; the goal is to see every step.

Create `ch08_tiny_rag.py` and run it with Python 3.10 or later.

```python
import re

docs = [
    {
        "id": "ragops",
        "source": "study-guide.md#ragops",
        "text": "A RAG app needs an evaluation set with fixed questions, expected sources, ideal answers, and failure labels.",
    },
    {
        "id": "chunking",
        "source": "rag-basics.md#chunking",
        "text": "A RAG app splits documents into chunks and keeps source metadata so answers can cite evidence.",
    },
    {
        "id": "agentops",
        "source": "agent-guide.md#trace",
        "text": "Agent systems record tool calls, observations, permissions, and recovery steps.",
    },
]

question = "Why does a RAG app need an evaluation set?"
STOPWORDS = {"a", "an", "the", "why", "does", "with", "and", "so", "can", "be"}


def tokenize(text: str) -> set[str]:
    return set(re.findall(r"[\w\u4e00-\u9fff\u3040-\u30ff]+", text.lower())) - STOPWORDS


query_tokens = tokenize(question)
ranked = sorted(
    (
        (len(query_tokens & tokenize(doc["text"])), doc)
        for doc in docs
    ),
    key=lambda item: item[0],
    reverse=True,
)

print("question:", question)
print("top chunks:")
for score, doc in ranked[:2]:
    print(f"- {doc['id']} score={score} source={doc['source']}")

best = ranked[0][1]
answer = (
    "Use a fixed evaluation set so every RAG change can be compared "
    f"against the same questions and expected sources. [{best['source']}]"
)
print("answer:", answer)
```

Expected output:

```text
question: Why does a RAG app need an evaluation set?
top chunks:
- ragops score=4 source=study-guide.md#ragops
- chunking score=2 source=rag-basics.md#chunking
answer: Use a fixed evaluation set so every RAG change can be compared against the same questions and expected sources. [study-guide.md#ragops]
```

Operation tip: add one new document, ask one new question, and print the top-k chunks before reading the final answer. If the evidence is wrong, the answer cannot be trusted.

## 8.0.4 Debug Bad RAG Answers

![RAG debugging ladder](/img/course/ch08-rag-debug-ladder-en.svg)

When the answer is bad, locate the failing layer before changing the model.

| Symptom | Print first | Likely fix |
|---|---|---|
| The answer has no source | final prompt and retrieved chunks | keep source IDs in chunks and require citations |
| The source document has the answer but retrieval misses it | original text search and chunk text | adjust chunk size, add keywords, use hybrid search |
| Many chunks are recalled but the best one is not first | top-k scores and manual relevance labels | add reranking or rule-based filtering |
| The answer uses old information | document version and index build time | rebuild index and add regression tests |
| You cannot tell whether quality improved | before/after answers on the same questions | create a fixed evaluation set |

## 8.0.5 Common Failures

- Treating "connected a vector database" as "RAG is done." RAG quality also depends on document quality, chunking, ranking, Prompt, citations, and evaluation.
- Adding frameworks before understanding the chain. Frameworks are easier after you can print query, chunks, prompt, answer, and source.
- Letting the model answer when retrieval is empty. A useful RAG app must say "I do not know from the provided sources."
- Forgetting metadata. Without source, page, section, and version, citations and debugging become weak.
- Optimizing by feeling. Use the same evaluation questions every time you change chunking, retrieval, reranking, or Prompt.

## 8.0.6 Pass Check

Before entering Chapter 9, you should be able to:

- explain why RAG solves private, fresh, and citable knowledge problems;
- run the tiny RAG script and inspect top-k chunks before the answer;
- create chunks with source metadata and cite those sources in the answer;
- separate document, chunking, retrieval, generation, citation, and deployment failures;
- run the full Chapter 8 workshop, add one document, add one evaluation case, and record the result in a README.

For a printable checklist, use [8.0 Learning Checklist](./study-guide.md). For the guided project, start with [8.5.6 Hands-on: Full Chapter 8 RAG App Workshop](./ch05-projects/05-stage-hands-on-workshop.md).
