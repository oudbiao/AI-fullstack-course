---
title: "8 LLM Application Development and RAG"
sidebar_position: 0
description: "Learn large model application development, RAG (Retrieval-Augmented Generation), document processing, vector databases, Function Calling, LLM deployment, and engineering practices."
keywords: [LLM applications, RAG, Prompt Engineering, LangChain, vector databases, large model deployment]
---

# 8 LLM Application Development and RAG

![Main visual for LLM applications and RAG](/img/course/ch08-rag-engineering-en.png)

This stage is about “how to connect a large model to a real system.” You will move from model calls to document processing, knowledge bases, RAG, tool calling, dialogue systems, deployment, logs, and engineering practices.

## Story-based introduction: connect your knowledge base to the model

If you only chat with a general-purpose large model, it knows public knowledge learned during training. If you want it to answer questions from company documents, course materials, product manuals, or personal notes, you need to bring in external information. RAG is like giving the model a librarian: first find the materials, then answer based on them, and finally leave citations and logs so it’s easy to check where the answer came from.

## Learning quest map

![RAG learning quest map](/img/course/ch08-learning-quest-map-en.png)

## One picture for the main loop

![How RAG works from question to evidence](/img/course/ch08-rag-basics-workflow-map-v2-en.png)

This picture shows the loop we will keep reusing in this chapter: ask, retrieve, and answer with evidence.

## Interactive exercise: start by improving a “bad answer”

When doing RAG, deliberately pick a question the system answers badly, then trace the cause: is the answer missing from the original document, was important information split apart during chunking, did retrieval fail to recall it, did the Prompt fail to request citations, or did the model ignore the context? Each time you locate a problem, you are learning how to debug real LLM applications.

## Project bonus

A bonus project for this stage could be a “course materials Q&A assistant” or a “personal knowledge base assistant.” If you put in your earlier notes, project README files, and error logs, it can become a private teaching assistant that helps you keep learning, and it can also naturally lead into the next stage: Agent.

## Stage positioning

| Info | Description |
|---|---|
| Suitable for | Learners who already understand the basics of large models and want to build LLM applications, knowledge bases, or enterprise AI tools |
| Estimated time | 90–120 hours |
| Prerequisites | Complete the large model principles, Prompt, and fine-tuning stages |
| Stage output | RAG knowledge base, intelligent Q&A assistant, course materials assistant, or enterprise knowledge base demo |

## Minimum beginner path

Beginners should first hand-code a minimal RAG: prepare documents, chunk text, generate embeddings, retrieve chunks, organize the Prompt, and generate answers with sources. As long as you can identify whether poor RAG performance comes from the documents, chunking, retrieval, prompting, or the model, you’ve completed the minimum path.

## Advanced learning path

Experienced learners can go deeper into reranking, multi-path retrieval, query rewriting, evaluation set construction, logging and monitoring, access control, and deployment. You can also try packaging RAG as an API or a small app, and add feedback collection and cost statistics.

## What beginners should do first, and what advanced learners should do later

When you first learn this stage, don’t rush into complex frameworks. A safer order is to use a few dozen lines of code to get the minimal loop working: “document → chunking → vector → retrieval → answer → citation,” then gradually replace pieces with more reliable components. You should first be able to see what happens at each step, and only then understand why vector databases, reranking, logs, and evaluation are needed.

Experienced learners can focus on engineering quality: how to degrade gracefully when retrieval fails, how to ensure citations are trustworthy, how logs support postmortems, how evaluation sets are built, how APIs are deployed, and how costs are tracked. Your goal is not just a Demo that “can answer,” but a knowledge base system that people are willing to use long term and that can be debugged when something goes wrong.

## Why RAG and application engineering are needed

A large model itself cannot naturally access your private documents, and it cannot guarantee that its knowledge is always up to date. RAG puts external knowledge retrieval results into the model context so the model can answer based on the materials. Application engineering is responsible for connecting model capabilities to a product: handling user input, calling the model, organizing state, recording logs, controlling costs, and evaluating results.

![Main backbone diagram of a RAG application system](/img/course/ch08-rag-system-backbone-en.png)

## Modern RAG deep dive: from can answer to maintainable

By 2025–2026, RAG is no longer just “vector database + Prompt.” Real knowledge bases face problems such as keyword misses, unstable chunk ranking, outdated documents, untrustworthy citations, and uncontrollable cost and latency. So this stage treats RAG as a long-running engineering system.

| Technical direction | Problem it solves | What to focus on |
|---|---|---|
| Hybrid Search | Pure vector retrieval may miss exact keywords, IDs, terms, and names | Combine keyword retrieval with vector retrieval, then merge and rank |
| Reranking | Initial recall returns many results but ranking is unstable | Use a reranking model or rules to move the most relevant chunks to the top |
| Query Rewrite | User questions are too short, too colloquial, or missing context | Rewrite the question into a query that is better for retrieval |
| Multi-query Retrieval | A single query does not cover enough | Generate queries from multiple angles to improve recall |
| GraphRAG | Answers depend on entities and relationships across documents | Extract entity relationships and organize context around a graph structure |
| Agentic RAG | One retrieval is not enough; the system must keep checking and deciding | Let the system decide whether to keep retrieving, change the query, or stop |
| Multimodal RAG | Knowledge sources include PDFs, screenshots, tables, and images | Combine document parsing, visual understanding, and text retrieval |

When learning these techniques, don’t treat them as components you must all stack on top of each other. Each technique should answer a failure mode: why is plain RAG not enough? Does it improve recall, ranking, context organization, citation trustworthiness, or operations and updates?

## RAGOps deep dive: how to keep improving after launch

RAGOps focuses on quality maintenance after a RAG system is online. A qualified RAG project should be able to see at least: document sources, chunking method, index version, recalled chunks, reranking scores, answer citations, no-answer handling, user feedback, token cost, response latency, and failure logs.

![RAGOps continuous improvement loop](/img/course/ch08-ragops-improvement-loop-en.png)

Minimal RAGOps does not need to be complicated at the start, but it must have a fixed evaluation set. For example, prepare 20–50 course questions, annotate the expected document, ideal answer, and boundaries for what must not be fabricated. Every time you change the Prompt, chunking strategy, Embedding model, or reranking approach, compare results using the same question set instead of guessing by feeling.

## Learning path for this stage

Chapter 1 teaches RAG. You will understand document parsing, chunking, Embedding, vector databases, retrieval strategies, RAG optimization, and evaluation.

Chapter 2 teaches local LLM deployment and unified interfaces. You will understand local models, inference services, and the meaning of a unified API.

Chapter 3 teaches large model application development, including LLM API, LangChain basics, Function Calling, dialogue systems, document parsing, AI-assisted coding, and template document generation.

Chapter 4 teaches engineering practices, including asynchronous programming, API design, logging and monitoring, and Docker deployment.

Chapter 5 moves into a comprehensive project, combining the knowledge base, model calls, application interfaces, and engineering practices.

## What you should be able to do after finishing

- Design a basic RAG workflow
- Complete document parsing, chunking, Embedding, and vector retrieval
- Determine whether poor RAG performance comes from documents, chunking, retrieval, prompting, or the model
- Call LLM APIs and wrap them as application interfaces
- Use Function Calling or tool calling to organize simple tasks
- Add logs, error handling, and basic deployment notes to LLM applications

## Common misconceptions

Don’t think that “connecting a vector database” means RAG is done. The real factors that affect performance are document quality, chunking strategy, retrieval recall, reranking, prompt organization, citation trustworthiness, and evaluation methods.

Also, don’t rely on complex frameworks too early. Frameworks can improve efficiency, but only after you understand the underlying flow. It’s recommended to hand-code the minimal RAG first, then learn frameworks like LangChain and LlamaIndex.

## RAG error theater: where bad answers usually get stuck

If RAG answers are inaccurate, don’t rush to change the model. First check whether the answer exists in the original document, then see whether chunking split important information apart, whether the retrieval results recalled relevant chunks, whether the Prompt required answers based on sources, and only then decide whether the model ignored the context.

## How to read the first time: must-read, project reference, and optional deep dives

| Reading label | Recommended chapters | Learning goal |
|---|---|---|
| Must-read | RAG basics, document processing, retrieval strategies, RAG evaluation, LLM API practice | Get the minimal Q&A loop working for a knowledge base |
| Project reference | Vector databases, RAG optimization, Function Calling, logging and monitoring, API design | Focus on when building a course Q&A assistant or enterprise knowledge base |
| Optional deep dive | Advanced RAG, local model deployment, LangChain, Docker deployment, template document generation | Deepen only when you need performance tuning, deployment, or app expansion |

For the first pass, it’s best to hand-code the minimal RAG first, then connect a framework later. As long as you can print the query, top-k chunks, sources, and answer, you’ve already captured the main thread of this stage.

## Small runnable RAG experiment: understand the retrieval chain without a framework

When learning this stage, it’s a good idea to first build a minimal runnable experiment and not rush into LangChain or a complex vector database. Prepare 5–10 course text snippets, simulate retrieval with keyword overlap or a simple Embedding approach, then print each step: user question, rewritten query, recalled chunks, ranking scores, final Prompt, and answer sources.

```python
import re

questions = ["Why does a RAG project need an evaluation set?"]
docs = [
    {"id": "ragops", "text": "RAGOps needs to record document sources, retrieval chunks, citations, cost, and failure logs."},
    {"id": "agentops", "text": "AgentOps focuses on execution traces, tool permissions, failure recovery, and human confirmation."},
]

def tokenize(text):
    return re.findall(r"[\w\u4e00-\u9fff\u3040-\u30ff]+", text.lower())

query = questions[0]
hits = sorted(
    docs,
    key=lambda d: len(set(tokenize(query)) & set(tokenize(d["text"]))),
    reverse=True,
)

for hit in hits[:2]:
    print(hit["id"], hit["text"])
```

The point of this experiment is not how powerful the algorithm is, but to let learners see for the first time that “retrieval results can be inspected.” Once the minimal chain is working, you can replace it with vector models, Hybrid Search, Reranking, Query Rewrite, and evaluation sets.

## RAG failure case library: locate problems by symptom

| Symptom | Common cause | How to locate it | Fix direction |
|---|---|---|---|
| The answer sounds fluent but has no source | The Prompt does not force citations, or the context does not preserve source IDs | Print the final Prompt and recalled chunks | Keep document name, page number, and paragraph ID in the chunks |
| The document clearly has the answer, but retrieval misses it | Chunking is too fragmented, keywords are missing, or vector retrieval drifted | Search the original text directly with keywords, then inspect the chunk contents | Adjust chunks, add Hybrid Search, or use Query Rewrite |
| Many chunks are recalled but the ranking is off | Initial recall only finds “possibly relevant” items | Print top-k scores and manually labeled relevance | Add Reranking or rule-based filtering |
| The document was updated but the answer is still old | The index version was not refreshed | Record document versions and index time | Add index rebuilding, expiration flags, and regression evaluation |
| You don’t know whether optimization helped | There is no fixed question set | Compare before/after answers using the same question set | Build an evaluation set and a failure sample table |

| Review question | What you should be able to answer |
|---|---|
| Source of knowledge | Which documents, web pages, or databases does the system rely on? |
| Knowledge processing | How are documents parsed, cleaned, chunked, and written into the index? |
| Retrieval quality | Which chunks did the user question hit, and are the ranking and scores reasonable? |
| Answer citations | Can the user see which sources the answer is based on? |
| Error handling | How does the system handle empty retrievals, model timeouts, or format errors? |
| Evaluation iteration | Is there a fixed question set and logs to compare results before and after optimization? |

The real outcome of this stage is a knowledge base assistant with sources, logs, error handling, and evaluation samples—not just one successful model call.


## Fun task card for this stage

| Play mode | Task for this stage |
|---|---|
| Story quest | Let the assistant answer based on materials: import documents, retrieve chunks, generate answers, and check whether citations support the conclusion. |
| Boss fight | **Citation Hallucination Dragon** |
| Unlockable badges | RAG citation police, retrieval archaeologist |
| Easy beginner mode | Only complete a minimal input-to-output loop and keep a run screenshot or command output |
| Portfolio evidence | eval questions, retrieval logs, citation_check |

If this stage feels like a lot, first treat this task card as your minimum goal. If you can finish the easy beginner mode, you can keep learning; later, when preparing your portfolio, come back and upgrade to the standard and challenge versions.

## Stage deliverables

| Deliverable | Minimum version | Portfolio version |
|---|---|---|
| RAG prototype | Can retrieve and answer for 5–10 document snippets | Supports course document import, source citations, and no-answer handling |
| Document processing record | Explain chunk size and source fields | Save `chunks.jsonl`, metadata, and index version |
| Retrieval logs | Print top-k chunks | Record query, score, source, rerank, and matched text summary |
| Evaluation set | 10 fixed questions | Annotate gold_doc, gold_answer, citation_ok, and failure type |
| Failure samples | Record 1–3 failed questions | Separate retrieval, context, generation, citation, and deploy issues |
| README | Clearly explain run commands and sample output | Show architecture, config, evaluation results, limitations, and next steps |

## Stage evaluation rubric

| Level | Evaluation standard | Portfolio evidence |
|---|---|---|
| Basic pass | Can complete document reading, chunking, retrieval, answering, and source display | Run screenshots, sample questions, recalled chunks |
| Standard pass | Can add an evaluation question set, failure samples, logs, and error handling | `evals/questions.jsonl`, failure case table, log samples |
| Excellent work | Can compare different retrieval strategies and explain cost, latency, and citation trustworthiness | Hybrid Search / Reranking comparison, cost records, deployment notes |

When presenting in an interview or portfolio, don’t just say “I built a RAG.” A better way to say it is: I first built the minimal RAG loop, then found that recall was unstable for certain questions, so I added an evaluation set, printed top-k chunks, compared chunking and reranking strategies, and finally made the system provide sources and record failure reasons.

## Stage project

The basic version is to build a personal knowledge base Q&A assistant that supports answering questions from local documents and provides sources. The standard version needs document preprocessing, vector indexing, retrieval evaluation, logging, and a simple Web API. The challenge version can be an enterprise knowledge base demo with permissions, feedback, reranking, multi-turn dialogue, and online deployment notes.

If you want a more detailed learning rhythm, you can read [Study Guide: How to Learn LLM Applications and Engineering Without Getting Lost](./study-guide.md).

## Relationship with the AI Learning Assistant capstone project

This stage can correspond to AI Learning Assistant v0.8: reading course Markdown, supporting retrieval, answering, source citations, and an evaluation question set. If you are following the capstone project path, it’s recommended that by the end of this stage you submit at least one version record: what capabilities were added in this stage, how to run it, what the sample input/output looks like, what problems you encountered, and what you plan to change next.

## Stage completion criteria

| Completion level | What you need to do |
|---|---|
| Minimum completion | Build a course Q&A or knowledge base assistant and implement retrieval, citations, and evaluation. |
| Recommended completion | Finish at least one runnable small project in this stage, and document the run method, sample input/output, and encountered issues in the README. |
| Portfolio completion | Connect this stage’s output to the “AI Learning Assistant” capstone project, and leave screenshots, logs, evaluation samples, and a next-step plan. |

After finishing this stage, you do not need to memorize every detail. What matters more is that you can clearly explain: what problem this stage solves, how it relates to the previous stage, and how it will support later learning. The next stage will upgrade the system from Q&A to an Agent that can call tools.
