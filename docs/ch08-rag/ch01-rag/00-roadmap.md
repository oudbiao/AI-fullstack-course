---
title: "1.1 Pre-Class Guide: What Exactly Will We Learn in This RAG Chapter?"
sidebar_position: 0
description: "First build a learning map for the RAG chapter: how document processing, chunking, vectorization, retrieval, reranking, generation, and evaluation connect into the main knowledge-enhanced workflow."
keywords: [RAG guide, retrieval-augmented generation, vector database, document chunking, reranking, RAG evaluation]
---

# Pre-Class Guide: What Exactly Will We Learn in This RAG Chapter?

This chapter solves this question: when the model’s knowledge is not new enough, not complete enough, or not well aligned with your business data, how can you reliably bring external knowledge into the answer pipeline?

RAG is not just “connect a vector database.” What it really teaches is a way of thinking about LLM application development: which knowledge should be stored in model parameters, which knowledge should live in external documents, how to retrieve relevant materials when the user asks a question, and how to make the model answer based on the materials instead of making things up.

## Where This Chapter Fits in the Overall Course

You have already learned LLM principles, Prompting, fine-tuning, and alignment in Chapter 7. By Chapter 8, the course moves from “understanding model capabilities” to “organizing application systems.” RAG is the first key foundation of this application path.

If Prompt helps the model understand tasks better, and fine-tuning adjusts the model’s behavior, then RAG is more like connecting the model to a knowledge base that can be updated, traced, and managed.

![Bridge diagram showing RAG’s position in LLM applications](/img/course/ch08-rag-position-bridge-en.png)

## The Real Problems This Chapter Solves

This chapter answers five questions: why you cannot just feed an entire document directly to the model; why chunking, cleaning, and metadata affect retrieval quality; what Embedding and vector databases actually do; how retrieval, reranking, filtering, and context assembly affect the final answer; and how to judge whether a RAG system is good or not.

The easiest mistake beginners make with RAG is thinking that “if it can retrieve something, then the answer is reliable.” In reality, RAG performance often gets stuck because of document quality, chunk size, retrieval coverage, reranking accuracy, prompt organization, and answer evaluation.

## Recommended Learning Order for Beginners

It is recommended to first understand the smallest RAG loop: after the user asks a question, the system retrieves relevant materials, then sends the materials and the question to the model for answering. Then learn document processing and chunking, so you understand that before knowledge enters the system, it needs to be cleaned, split into chunks, and given metadata. Next, learn Embedding and vector databases to understand why similarity search can find relevant passages. Finally, look at retrieval optimization, reranking, and evaluation, because these determine whether the system can move from a demo to a usable product.

![Flow diagram of the core chapter learning order for RAG](/img/course/ch08-rag-core-chapter-flow-en.png)

## The Main Thread to Focus on in This Chapter

The main thread of this chapter can be summarized as: turn “materials” into “retrievable knowledge,” then turn “retrieval results” into “answerable responses.”

![Pipeline diagram from materials to answers in RAG](/img/course/ch08-rag-data-to-answer-pipeline-en.png)

The first half focuses on turning raw materials into a retrievable knowledge index; the second half focuses on retrieval, reranking, answering, and citation based on the user’s question.

Once you understand this chain, you will know that tuning a RAG project is not just about changing the prompt. If nothing is retrieved, you should check document parsing, chunking, and retrieval coverage; if something is retrieved but the answer is wrong, you should check reranking, context organization, and generation prompts; if the answer looks correct but cannot be verified, you should add source citations and evaluation examples.

## How This Chapter Relates to Later Chapters

RAG is the foundation for later course topics such as question-answering assistants, personal knowledge bases, enterprise knowledge bases, and Agent tool calling. The model deployment chapter will explain how the model can be called reliably, the application development chapter will put RAG into chat, file upload, and API workflows, and the engineering chapter will add logging, error handling, evaluation, and deployment.

If you do not build a solid foundation in this chapter, common problems later will be: the knowledge base demo runs, but the system answers off-topic for slightly more complex questions; source citations are unstable; when users ask a question with no answer, the model hallucinates; and the system cannot tell whether the failure came from retrieval, the model, or poor document quality.

## How Beginners and Advanced Learners Should Read This Chapter

When beginners read this chapter for the first time, they should first focus on the main thread and the smallest runnable example. You do not need to understand every detail at once. As long as you can explain what problem this chapter solves, what the inputs and outputs are, and how the smallest project runs, you can move forward.

More experienced learners can treat this chapter as a chance to fill gaps and practice engineering: focus on edge cases, failure examples, evaluation methods, code reproducibility, and how it connects to the previous and next stages. After reading, it is best to store the chapter’s key points in your own project README or experiment notes.

## Suggested Study Time and Difficulty

| Study Method | Suggested Time | Goal |
|---|---|---|
| Quick skim | 20–30 minutes | Understand what problem this chapter solves and where it will be used later |
| Minimum pass | 1–2 hours | Run a minimal example and complete the chapter’s small project exit |
| Deep practice | Half a day to 1 day | Add error analysis, comparison experiments, or project README notes |

## Chapter Self-Check Questions

| Self-Check Question | Passing Standard |
|---|---|
| What problem does this chapter solve? | You can explain its position in the whole course in one sentence |
| What are the minimum inputs and outputs? | You can clearly explain what inputs the example needs and what result it produces |
| Where are the common failure points? | You can list at least one cause of errors, poor results, or misunderstanding |
| What can you retain after learning it? | You can write the chapter output into a project README, experiment notes, or portfolio |

## Small Project Exit for This Chapter

After finishing this chapter, it is recommended to build a minimal course knowledge-base Q&A system. Prepare 3 to 5 course documents, and complete document loading, chunking, vectorization, retrieval, context assembly, model answering, and source display. The project does not need a complex interface, but it must show the retrieved passages and answer sources.

A suggested minimum deliverable includes: 3 to 5 Markdown documents in `docs/`, a `rag_demo.py` file or notebook, at least 10 fixed test questions, printed results for top-k retrieved chunks, and an evaluation table for “question → matched document → whether the answer is correct.”

```python
query = "Why does RAG need source citations?"
for i, chunk in enumerate(top_k_chunks, start=1):
    print(i, chunk["source"], chunk["text"][:80])
```

If you want to go further, you can add a small evaluation set, such as 10 questions, with the expected matching document and ideal answer labeled for each one, so you can observe the changes before and after adjusting chunking and retrieval strategies.

## Passing Criteria

By the end of this chapter, you should be able to explain the full path from documents to answers in RAG, clearly describe what chunk, embedding, vector database, retrieval, reranking, context, and citation each do, and judge roughly which layer is causing poor RAG performance.

If you can build a minimal knowledge-base assistant with source citations, retrieved passage display, and simple evaluation examples, then you have reached the basic requirement for moving on to LLM application development and the Agent stage.
