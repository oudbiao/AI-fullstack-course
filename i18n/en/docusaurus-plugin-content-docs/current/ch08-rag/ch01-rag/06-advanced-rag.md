---
title: "1.7 Advanced RAG Architectures"
sidebar_position: 6
description: "Understand how systems keep evolving when basic RAG is no longer enough, from routing and multi-hop retrieval to Agentic RAG and structured retrieval."
keywords: [advanced RAG, router, multi-hop, agentic rag, graph rag, structured retrieval]
---

# Advanced RAG Architectures

## Learning Objectives

After completing this section, you will be able to:

- Understand why basic RAG is not enough in complex scenarios
- Recognize common architectures such as routing-based, multi-hop, and Agentic RAG
- Run a toy example of “multi-knowledge-base routing”
- Know when to upgrade a RAG architecture, and when not to

---

## 1. Why Does Basic RAG Eventually Hit Its Limits?

### 1.1 Basic RAG Is Suitable for “One Question -> One Retrieval -> One Answer”

This is already enough for many FAQs and simple Q&A tasks.  
But when the problem gets more complex, bottlenecks start to appear.

For example:

- Need to search across multiple knowledge bases
- Need to check policies first, then product docs
- Need to break the task into multiple sub-questions

### 1.2 Common Complex Scenarios

For example, a user asks:

> “Can this learner get a refund? If not, is there an extension option?”

This actually implies multiple actions:

1. Check the refund policy
2. Determine whether the current conditions are met
3. Then check the extension option

At this point, “retrieving only once” is often not enough.

---

## 2. Routing-Based RAG: Decide Where to Search First

### 2.1 When One Knowledge Base Is Not Enough, Route First

Many systems do not have just one document store. They may have:

- Policy knowledge base
- Product knowledge base
- Technical documentation knowledge base
- FAQ knowledge base

If all queries go into the same store, the noise can be very high.  
A better approach is:

> First determine which knowledge base the question belongs to, then retrieve from there.

### 2.2 A Runnable Multi-Store Routing Example

```python
policy_docs = [
    "Refund policy: You can apply for a refund within 7 days after purchasing the course.",
    "Certificate policy: You can get a certificate after passing the test."
]

tech_docs = [
    "If login fails, first check your account password and network connection.",
    "A 401 error from API calls usually indicates authentication failure."
]

def route_query(query):
    if "refund" in query or "certificate" in query:
        return "policy"
    if "login" in query or "API" in query or "401" in query:
        return "tech"
    return "default"

def retrieve_simple(query, docs):
    return [doc for doc in docs if any(word in doc for word in query)]

queries = ["how to get a refund", "how to handle a 401 error"]

for q in queries:
    route = route_query(q)
    if route == "policy":
        hits = retrieve_simple(q, policy_docs)
    elif route == "tech":
        hits = retrieve_simple(q, tech_docs)
    else:
        hits = []
    print(q, "-> routed to", route, "->", hits)
```

This is the simplest version of “Router RAG.”

---

## 3. Multi-hop RAG: Break the Problem into Multiple Steps

### 3.1 Some Questions Cannot Be Answered in One Step

For example:

> “What conditions has this person completed, and what is still missing for them to get certified?”

This kind of question usually requires:

1. Check the certification rules
2. Check the user’s completion status
3. Compare the two

### 3.2 Multi-hop RAG Is More Like Solving a Problem Step by Step

Instead of finding all the materials at once, it works like this:

- Solve the first sub-question first
- Then continue retrieving based on the intermediate result

This feels closer to an Agent.

---

## 4. Agentic RAG: Retrieval Is No Longer a Fixed Pipeline

### 4.1 What Is the Difference from Normal RAG?

Normal RAG is more like a fixed flow:

1. Retrieve
2. Assemble context
3. Answer

Agentic RAG, on the other hand, may:

1. Decide whether retrieval is needed
2. Decide how many times to retrieve
3. Decide whether to rewrite the query or switch data sources
4. Then decide whether to continue acting

### 4.2 Advantages and Trade-offs

Advantages:

- More flexible
- Can handle complex tasks

Trade-offs:

- Harder to debug
- Slower
- Higher cost

So not every RAG system should be made agentic.

---

## 5. Structured Retrieval: Not All Knowledge Should Go into a Pure Text Store

### 5.1 When the Data Itself Has Structure

For example:

- Order table
- User status
- Ticketing system
- Grade table

These kinds of data are often better handled by:

- SQL queries
- API queries
- Graph databases

rather than forcing them into plain text and then retrieving from that.

### 5.2 A Common Upgrade Path

Real systems may combine:

- Unstructured document RAG
- Structured database queries
- Tool calling

This is also why “advanced RAG” is often closely tied to Agents.

---

## 6. Graph RAG and Knowledge Graph Thinking

### 6.1 What Problem Does It Solve?

When knowledge points have obvious relationships, plain text chunking may not be enough.

For example:

- Person relationships
- Company organizational structure
- Product dependency relationships

In these cases, a graph structure makes it easier to express the connections between nodes.

### 6.2 When Is It Worth Considering?

When your questions often require:

- Jumping across multiple entities
- Following relationship chains
- Structured reasoning

you can consider graph-style retrieval.

![Advanced RAG architecture selection map](/img/course/ch08-advanced-rag-architecture-selection-map.png)

:::tip Reading guide
Start by choosing an architecture based on the question type: if multiple knowledge bases interfere with one another, consider Router RAG first; if the question needs multiple steps, consider Multi-hop RAG; if you need autonomous decision-making, consider Agentic RAG; if relationship chains are obvious, then consider Graph RAG.
:::

---

## 7. When Should You Upgrade to Advanced RAG?

### 7.1 Signs That It Is Worth Upgrading

If you are already facing these problems:

- Multiple knowledge bases interfere with each other
- One retrieval is often not enough
- Structured data needs to work together with retrieval
- The question clearly needs step-by-step reasoning

then it may be time to upgrade the architecture.

### 7.2 Signs That It Is Not Worth Upgrading

If you have not even stabilized basic RAG yet:

- Chunking is unreasonable
- There is no evaluation set
- You have not tuned top-k

then do not rush into advanced architectures.

---

## 8. Common Beginner Mistakes

### 8.1 Wanting to Use Agentic RAG as Soon as a Task Looks Complex

In many cases, getting routing and retrieval strategies right already solves most of the problem.

### 8.2 Thinking “More Components” Means “More Advanced”

More components do not necessarily mean a better system; they may just make maintenance harder.

### 8.3 Upgrading Architectures Without Evaluation

Without evaluation, you cannot tell whether the upgrade is a real improvement or just “something that looks more complicated.”

---

## Summary

The most important takeaway from this section is:

> Advanced RAG is not about showing off. It is about giving the system a smarter way to organize retrieval when basic RAG cannot cover complex questions.

Polishing a simple architecture first, and then deciding whether to upgrade, is usually the more mature engineering path.

---

## Exercises

1. Add a “course content knowledge base” to the routing example and extend the `route_query()` rules.
2. Think about your own project: is there any data that would actually be better suited to SQL / API queries rather than pure text retrieval?
3. Try to come up with a question that can only be answered with multi-hop retrieval.
