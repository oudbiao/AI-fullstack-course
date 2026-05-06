---
title: "Phase Learning Task Sheet"
description: "Break the LLM application development and RAG phase into actionable learning tasks, practice deliverables, and completion criteria."
keywords: [RAG, LLM applications, vector database, learning task sheet, AI application engineering]
---

# Phase Learning Task Sheet: LLM Application Development and RAG

The goal of this phase is to help you integrate a large model into a real application, rather than only asking questions in a chat box. You need to master API calls, prompts, document processing, vector retrieval, answer citations, evaluation, and engineering deployment.

## Required Tasks for This Phase

| Task | Deliverable | Passing Criteria |
| --- | --- | --- |
| Get an LLM API call working | A minimal calling script | Can handle key, requests, responses, exceptions, and retries |
| Complete document processing | A chunking and cleaning record | Can explain chunk size, overlap, metadata, and source |
| Build vector retrieval | A minimal retrieval demo | Can retrieve relevant document fragments from a question |
| Complete RAG Q&A | A course Q&A prototype | Answers include source citations and can explain the retrieval basis |
| Complete RAG evaluation | A test set and evaluation table | Can record hit rate, answer quality, citation quality, and failure samples |
| Complete the hands-on workshop | `rag_app_workshop.py` output and one added eval case | Can reproduce the expected output, explain each stage, and extend it safely |

## Recommended Learning Order

First get the model API working, then process documents and embeddings, then build vector retrieval, and finally connect the retrieval results to the generation model. Don’t introduce complex frameworks at the beginning; first use minimal code to understand the input/output boundaries of RAG.

The key to RAG is not “connecting a vector database,” but being able to explain why a document fragment was retrieved, whether the answer is truly supported by the source, which questions cannot be retrieved, and which answers may hallucinate.

## Relationship to the AI Learning Assistant Project

This phase corresponds to the v0.8 course Q&A assistant in the AI Learning Assistant project. It should be able to read course Markdown, build an index, answer learner questions, and provide citation sources. This version is the key milestone that upgrades the project from a “study log tool” to an “AI assistant.”

Recommended minimum features include: importing course documents, chunking by headings and body text, saving metadata, retrieving relevant fragments, generating answers, displaying citation paths, and recording questions and answers. The standard version then adds an evaluation set, failure sample analysis, and configurable parameters.

## Common Sticking Points

Common issues include document chunks that are too small or too large, lost metadata, mismatches between the embedding model and the language, retrieval hits where the answer does not use the source, answers that look correct but are not supported by citations, and context windows that are too long, causing excessive cost and latency. When debugging, separate the retrieval results from the generated answer and inspect them independently.


## Easy / Standard / Challenge Tasks

| Difficulty | What you need to complete | Who it is for |
|---|---|---|
| Easy | Finish 5 answers with sources | First-time learners, those with limited time, or beginners |
| Standard | Complete 10 evaluation questions and citation_ok checks | Learners who want to include this phase in their portfolio |
| Challenge | Compare failure types across chunk, top-k, or rerank strategies | Learners with some foundation who want stronger project evidence |

## This Phase’s Badge and Boss Fight

| Type | Content |
|---|---|
| Boss Fight | Citation Hallucination Dragon |
| Unlockable Badges | RAG Citation Police, Retrieval Archaeologist |
| Minimum Completion Slogan | Get it working first, then explain it, then record failures |
| Evidence Saving Suggestion | Save screenshots, logs, failure samples, or evaluation tables to `reports/`, `evals/`, or `logs/` |

Once you complete the Easy version, you can move on. Only after completing the Standard version is it recommended to include it in your portfolio. Do the Challenge version only if you have extra capacity.

## Phase Portfolio Deliverables

If you want to turn this phase’s results into portfolio material, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `chunks.jsonl` | Document chunking results, including fields such as text, source, section, page, and content_type |
| `retrieval_logs.jsonl` | top-k, score, source, and retrieved text summaries for each query |
| `eval_questions.csv` | Fixed evaluation questions, ground-truth answers, expected documents to hit, and key citations |
| `failure_cases.md` | Samples of retrieval failures, generation failures, citation failures, metadata failures, etc. |
| `rag_config.md` | Configuration records such as chunk_size, overlap, top-k, rerank, and prompt_version |
| `rag_app_workshop_output.txt` | Output from [5.6 Hands-on: Full Chapter 8 RAG App Workshop](./ch05-projects/05-stage-hands-on-workshop.md), plus notes about one change you made |
| `README.md` | Run commands, example inputs and outputs, evaluation results, and improvement plans |

These files do not need to be complete at the very beginning, but they will help turn your RAG project from “can answer” into “can explain, can evaluate, and can review.”

## Phase Completion Questions

After learning this phase, you should be able to answer these questions: what limitations of large models does RAG solve, what does chunk size affect, what is the difference between embedding and keyword retrieval, why must answers have citations, and how to determine whether a RAG failure is a retrieval problem or a generation problem.

## Completion Status Checklist

- [ ] I can independently complete an LLM API call and handle exceptions and retries.
- [ ] I can explain the relationship between document chunking, embedding, vector retrieval, and answer generation.
- [ ] I can inspect the retrieved original text fragments, rather than only the final answer.
- [ ] I have completed a RAG Q&A prototype with source citations.
- [ ] I have completed the Chapter 8 hands-on workshop and added at least one new document or evaluation case.
- [ ] I have a fixed set of evaluation questions and have recorded samples of retrieval failures, citation failures, or generation failures.
