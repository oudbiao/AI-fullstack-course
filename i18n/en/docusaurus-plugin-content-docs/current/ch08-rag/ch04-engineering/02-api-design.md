---
title: "4.3 API Design and Serviceization"
sidebar_position: 18
description: "From request structure, response structure, idempotency, and error handling to version management, understand how LLM service APIs can be designed more robustly."
keywords: [API design, service design, idempotency, request schema, response schema, versioning]
---

# API Design and Serviceization

:::tip Section Focus
When building LLM applications, many people can write a local script, but once they move to serviceization, things quickly get messy.  
The real question is not “Can you write an endpoint?”, but:

> **Can this endpoint be called reliably by others for a long time?**

This section is here to answer that question.
:::

## Learning Goals

- Understand the most basic things an LLM service API should define
- Learn how to design clear request and response structures
- Understand key service concepts such as idempotency, error returns, `trace_id`, and version management
- Read and understand a minimal API processing loop

---

## 1. Why API design is not “just wrapping some JSON”

### 1.1 What does a bad interface look like?

```python
bad_request = {
    "msg": "What is the refund policy?"
}

bad_response = {
    "text": "Refunds are available within 7 days"
}
```

What’s wrong here?

- What is `msg`? User message? System message?
- No `trace_id`
- No error structure
- No version information
- No context field

### 1.2 What is a good API design doing?

At its core, it answers:

- What does the input look like?
- What does the output look like?
- How should errors be represented?
- Can it stay stable when called 10 times or 100,000 times?

In other words, API design is not “just writing an entry point”; it is defining:

> **The contract between the system and the outside world.**

---

## 2. First, design the request structure

### 2.1 A minimal request structure usually needs at least these fields

- `query`
- `user_id` (optional)
- `session_id` (for multi-turn scenarios)
- `metadata` (optional)

### 2.2 A clearer request object

```python
request = {
    "query": "What is the refund policy?",
    "user_id": 1,
    "session_id": "sess_001",
    "metadata": {
        "channel": "web"
    }
}

print(request)
```

Here, you can already feel:

- What the query is
- Who sent it
- Which session it belongs to
- What extra context is included

This is much better than “passing only a string.”

---

## 3. Then, design the response structure

### 3.1 Why must the response also be standardized?

Because real consumers are often not just people, but also:

- Frontend applications
- Other services
- Logging systems
- Evaluation systems

They all need to consume the result reliably.

### 3.2 A more robust response structure

```python
response = {
    "trace_id": "trace_001",
    "answer": "A refund can be requested within 7 days after purchase, provided the learning progress is below 20%.",
    "sources": [
        {"id": "doc_001", "section": "Refund Policy"}
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 35
    }
}

print(response)
```

### 3.3 Why are these fields valuable?

- `trace_id`: makes it easy to trace the request path
- `answer`: the actual business output
- `sources`: helps with citation and verification
- `usage`: helps with cost analysis

---

## 4. Error responses must also be designed

### 4.1 Many systems only design successful responses

But in real engineering, the more common issues are actually:

- Invalid parameters
- Upstream timeouts
- Insufficient permissions
- Empty knowledge base

### 4.2 A unified error structure

```python
error_response = {
    "trace_id": "trace_002",
    "error": {
        "code": "INVALID_ARGUMENT",
        "message": "query cannot be empty"
    }
}

print(error_response)
```

This step is very important because it makes the caller clearly understand:

- What went wrong
- What category the error belongs to
- Whether it is worth retrying

![API contract, error structure, and version management diagram](/img/course/ch08-api-contract-error-version-map.png)

:::tip Reading Guide
An API is a system contract, not just JSON. When reading the diagram, focus on request schema, response schema, error object, `trace_id`, and version, because they determine whether the interface can be consumed stably over time by the frontend, evaluation systems, and other services.
:::

---

## 5. A minimal runnable service handling function

### 5.1 Simulate an API handler with pure Python first

```python
def handle_chat(request):
    trace_id = "trace_demo_001"

    if "query" not in request or not request["query"].strip():
        return {
            "trace_id": trace_id,
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "query cannot be empty"
            }
        }

    answer = f"System reply: {request['query']}"
    return {
        "trace_id": trace_id,
        "answer": answer,
        "sources": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8}
    }

print(handle_chat({"query": "What is the refund policy?"}))
print(handle_chat({"query": ""}))
```

### 5.2 What is this code actually teaching?

It teaches you:

1. Validate the request first
2. Every request should have a `trace_id`
3. Both success and failure need a unified structure

This is already the most important layer of service design.

---

## 6. Why is idempotency important?

### 6.1 What is idempotency?

Simply put:

> Repeated calls with the same request should produce the same or a controlled result.

This is especially important in these scenarios:

- Retries
- Re-sending after a timeout
- Network instability

### 6.2 Which APIs need idempotency more?

Especially:

- Ticket creation
- Payment initiation
- Order changes

A pure question-answering API is usually more like a “read-only operation,” so idempotency is easier to handle.

---

## 7. Why can’t version management be added later?

### 7.1 Once others integrate with your API, changing fields casually becomes hard

If today the response returns:

- `answer`

and tomorrow it changes to:

- `response_text`

the caller will break immediately.

### 7.2 A simple versioning strategy

```python
api_info = {
    "version": "v1",
    "endpoint": "/api/v1/chat"
}

print(api_info)
```

Even for a small project, it is best to build version awareness early.

---

## 8. A FastAPI example closer to a real service

If you want to see a style closer to a real backend, take a look at this minimal version.

:::info Runtime Environment
```bash
pip install fastapi uvicorn
```
:::

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/v1/chat")
def chat(payload: dict):
    if "query" not in payload or not payload["query"].strip():
        return {
            "trace_id": "trace_demo_002",
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "query cannot be empty"
            }
        }

    return {
        "trace_id": "trace_demo_002",
        "answer": f"System reply: {payload['query']}"
    }
```

Although this code is simple, it is already very close to the smallest real-world service prototype.

---

## 9. If your goal is a “knowledge-base-driven courseware generation assistant,” what should the minimal API look like?

These systems usually need more than just a `/chat` endpoint.  
At minimum, they often have interfaces like these:

| Endpoint | What it is responsible for |
|---|---|
| `/courseware/generate` | Generate courseware structure or document based on a topic |
| `/courseware/preview` | Preview structured results first |
| `/documents/ingest` | Upload and parse PDF / Word / PPT |
| `/retrieval/search` | Debug retrieval results |

When building for the first time, a more stable default approach is usually:

1. Start with only one `generate` endpoint
2. Return structured results or an export link first
3. Then add debugging and batch interfaces

A very small request structure can be defined like this first:

```python
generate_request = {
    "topic": "Discount word problems",
    "audience": "Upper elementary school",
    "doc_format": "word",
    "style": "classroom explanation",
    "exercise_count": 3,
}

print(generate_request)
```

The value of this object is:

- It turns the slots collected during multi-turn conversation into actual service API parameters

## 10. Common mistakes beginners make most often

### 10.1 The request structure is too loose

It may feel convenient at first, but it becomes very painful later.

### 10.2 The error structure is inconsistent

This makes it increasingly difficult for the frontend and other services to integrate.

### 10.3 No `trace_id`

When something goes wrong, it becomes hard to trace the request path.

### 10.4 Binding the API too tightly to a single business logic from the start

This makes future expansion very difficult.

---

## Summary

The most important thing in this section is not getting the API to run, but understanding:

> **The core of API design is turning input, output, errors, and traceability into a stable system contract.**

Once the contract is clear, the service can truly be relied on by others for the long term.

---

## Exercises

1. Add support for a `session_id` field to `handle_chat()`.
2. Design a unified error code enum, such as `INVALID_ARGUMENT`, `TIMEOUT`, and `NOT_FOUND`.
3. Think about it: if this were a “ticket creation” API, how would you consider idempotency?
4. Explain in your own words: why is API design essentially about defining a system contract?
