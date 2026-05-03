---
title: "5.2 Project: Enterprise Knowledge Base Q&A"
sidebar_position: 21
description: "Build a portfolio-grade enterprise knowledge base Q&A system, covering document chunking, retrieval, permission filtering, source citation, and error analysis."
keywords: [enterprise knowledge base, RAG project, retrieval, metadata, source citation, permissions]
---

# Project: Enterprise Knowledge Base Q&A

:::tip Section positioning
The reason enterprise knowledge base Q&A is a great portfolio project is not because it sounds fancy, but because it is very real:

- There are documents
- There are permissions
- There are versions
- There are citations
- And there is pressure that “getting the answer wrong affects the business”

So the most important thing for this kind of project is not “whether it looks like it is answering,” but:

> **Does the answer come from the correct document, within the correct permission scope, and can it be traced back to the source?**
:::

## Learning Objectives

- Learn how to organize enterprise documents into searchable knowledge units
- Learn how to design the two enterprise-grade constraints of permissions and citations
- Learn how to build a showcaseable project loop with a minimal retriever
- Learn how to present the project around error analysis and traceability

---

## 1. Why is enterprise knowledge base Q&A harder than ordinary FAQ?

### 1.1 Documents are longer

Enterprise knowledge is often not just a few Q&As,
but rather:

- Policy documents
- Internal SOPs
- Training manuals
- Product documentation

### 1.2 Permissions are more complex

For the same question, there may be:

- An external version
- An internal version

### 1.3 Trust requirements are higher

Users will often ask:

- Where did this rule come from?
- Which file are you citing?

So enterprise knowledge base Q&A is more like a combination of:

- A retrieval system
- A citation system
- A permission system

![Enterprise knowledge base permission and citation loop diagram](/img/course/ch08-enterprise-kb-permission-citation-map-en.png)

:::tip Reading guide
An enterprise knowledge base cannot rely only on “semantic relevance.” First filter candidates by user permissions, then retrieve and rerank, and finally make sure the answer includes source citations; otherwise, the system may sound correct while leaking internal documents or becoming impossible to trace.
:::

---

## 2. Define the project scope first

A very suitable minimum scope for a portfolio project is:

> **Build a “refund / invoice / certificate / internal customer support SOP” knowledge base Q&A system for an internal help center on a course platform.**

It should at least answer four types of questions:

1. External policy questions
2. Internal process questions
3. Questions whose answers differ by permission
4. Questions that require source citations

### Why is this scope good?

- The document topics are focused
- The permission boundary is realistic
- It is easy to explain whether the result is good or bad

---

## 3. Design the knowledge units first, not the model first

The following example does three things:

1. Splits documents into the smallest knowledge units
2. Adds metadata to each chunk
3. Distinguishes between public and internal visibility

```python
kb = [
    {
        "id": "doc_001",
        "section": "Refund Policy",
        "department": "support",
        "visibility": "public",
        "text": "A refund can be requested within 7 days of purchase if learning progress is below 20%.",
        "keywords": {"refund", "7 days", "progress", "20%"},
    },
    {
        "id": "doc_002",
        "section": "Certificate Guide",
        "department": "teaching",
        "visibility": "public",
        "text": "A completion certificate can be issued after finishing all required projects and passing the course final test.",
        "keywords": {"certificate", "final test", "project"},
    },
    {
        "id": "doc_003",
        "section": "Internal Customer Support SOP",
        "department": "internal",
        "visibility": "internal",
        "text": "When handling a refund request, customer support must first verify the order number, learning progress, and payment channel.",
        "keywords": {"refund", "customer support", "SOP", "verify"},
    },
]

for item in kb:
    print(item)
```

### 3.1 Why add so much metadata here?

Because enterprise knowledge base retrieval is not only about “does the content seem similar,”
but also about deciding:

- Whether the current user is allowed to see it
- Which business domain it belongs to
- How the source should be shown in the answer

This is also the fundamental difference between an enterprise project and a normal Q&A demo.

---

## 4. Build an explainable retriever first

To make the example runnable in the current environment, we will not use an external embedding library yet,
but instead use a pure Python keyword-overlap retriever to get the project skeleton in place first.

```python
def retrieve(query, allowed_visibility, top_k=2):
    candidates = []

    for item in kb:
        if item["visibility"] not in allowed_visibility:
            continue
        score = sum(keyword in query for keyword in item["keywords"])
        candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in candidates[:top_k] if score > 0]


print("public user:")
print(retrieve("What is the refund policy?", allowed_visibility={"public"}))

print("\ninternal support:")
print(retrieve("What is the customer verification process?", allowed_visibility={"public", "internal"}))
```

### 4.1 Although this retriever is simple, why is it very suitable for teaching?

Because it makes three things very clear:

1. How query terms affect recall
2. How permissions affect the candidate set
3. Why the results are different

### 4.2 Why deliberately not use embeddings directly here?

Because this lesson first needs to explain clearly:

- Permissions
- Source citations
- Structured knowledge units

These enterprise-grade key points.
Once the skeleton is clear, switching to a stronger retrieval method will be much more stable.

---

## 5. Make “answer + sources” together

```python
def answer_with_sources(query, allowed_visibility):
    hits = retrieve(query, allowed_visibility=allowed_visibility, top_k=2)

    if not hits:
        return {
            "answer": "No sufficiently relevant information was found within the current permission scope.",
            "sources": [],
        }

    top = hits[0]
    return {
        "answer": top["text"],
        "sources": [
            {
                "id": top["id"],
                "section": top["section"],
                "department": top["department"],
                "visibility": top["visibility"],
            }
        ],
    }


print(answer_with_sources("What is the refund policy?", {"public"}))
print(answer_with_sources("What is the customer verification process?", {"public", "internal"}))
```

### 5.1 Why is “returning sources” a highlight of a portfolio project?

Because it makes the system do more than just “give you an answer,”
and also answer:

- Where did this answer come from?
- Why should I trust it?

This significantly increases the credibility of the project.

### 5.2 Why do enterprise scenarios need sources more than ordinary Q&A?

Because enterprise users often really use the answer to carry out a process.
Without sources, trust is hard to build.

---

## 6. How should this project be evaluated?

### 6.1 It is not enough to only check “whether it answered”

An enterprise knowledge base project should be evaluated in at least three layers:

1. Whether the retrieval is relevant
2. Whether the permissions are correct
3. Whether the citations are traceable

### 6.2 A minimal evaluation set

```python
eval_cases = [
    {
        "query": "What is the refund policy?",
        "visibility": {"public"},
        "expected_doc": "doc_001",
    },
    {
        "query": "What is the customer verification process?",
        "visibility": {"public"},
        "expected_doc": None,
    },
    {
        "query": "What is the customer verification process?",
        "visibility": {"public", "internal"},
        "expected_doc": "doc_003",
    },
]

for case in eval_cases:
    result = answer_with_sources(case["query"], case["visibility"])
    got = result["sources"][0]["id"] if result["sources"] else None
    print({
        "query": case["query"],
        "expected_doc": case["expected_doc"],
        "got": got,
        "match": got == case["expected_doc"],
    })
```

### 6.3 Why is this kind of evaluation valuable?

Because it directly covers the two most important risks in an enterprise knowledge base:

- It should answer, but fails to answer correctly
- It should not be visible, but internal documents are exposed

---

## 7. How can you take this project one step closer to portfolio quality?

### 7.1 Upgrade rule-based retrieval to vector retrieval

### 7.2 Add document chunking and reranking

### 7.3 Build a user interface for source display

The most recommended items to show are:

- User question
- Matched document
- Final answer
- Source citation

### 7.4 Show a few “permission-related failure examples”

This will be very convincing.

---

## 8. The most common pitfalls

### 8.1 Only building “can answer,” not “can be traced”

### 8.2 Only looking at semantic relevance, not permission boundaries

### 8.3 Making document chunks too coarse

When chunks are too coarse, both answers and sources often become vague.

---

## Summary

The most important thing in this lesson is to build a portfolio-grade judgment:

> **What makes enterprise knowledge base Q&A feel like a real project is not that it connects to a retriever, but that it organizes knowledge units, permission boundaries, answer generation, and source traceability into a trustworthy closed loop.**

Once this closed loop is clear, this project will look very much like a real enterprise system.

---



## Suggested version roadmap

| Version | Goal | Delivery focus |
|---|---|---|
| Basic version | Get the minimal loop running | Can accept input, process it, and output results, while keeping a set of examples |
| Standard version | Form a showcaseable project | Add configuration, logs, error handling, README, and screenshots |
| Advanced version | Close to portfolio quality | Add evaluation, comparison experiments, failure-case analysis, and a next-step roadmap |

It is recommended to finish the basic version first, and do not pursue an all-in-one solution from the beginning. Each time you upgrade a version, write into the README what new capability was added, how it was verified, and what problems still remain.

## Exercises

1. Add two more “public documents” and one “internal document” to `kb` to make query competition more realistic.
2. Why is “correct permissions” sometimes more important than “a beautiful answer” in an enterprise knowledge base project?
3. Think about it: if document chunks are cut too coarsely, how will that affect the answer and the citation?
4. If you turn this project into a portfolio piece, which 4 blocks of information would be most worth showing on the homepage?
