---
title: "E.B.3 Concurrent Programming (Including asyncio)"
sidebar_position: 10
description: "Start with I/O-bound tasks, understand the boundaries of threads, coroutines, and asyncio in service code, and learn a minimal concurrency controller."
keywords: [asyncio, concurrency, async, semaphore, gather, Python]
---

# E.B.3 Concurrent Programming (Including asyncio)

![asyncio concurrency control flowchart](/img/course/elective-asyncio-concurrency-control-en.png)

![async task timeout cancellation and rate limiting diagram](/img/course/elective-asyncio-timeout-cancel-rate-limit-map-en.png)

:::tip Reading the Diagram
More concurrency is not always better. When reading the diagram, focus on how the event loop, semaphore, timeout, cancellation, retry, and rate limit work together to protect upstream services, especially for LLM API calls, RAG scraping, and Agent tool invocations.
:::

:::tip Section Overview
Concurrency is one of the easiest Python topics to turn into an “API memorization exercise.”
But in real engineering, the more important question is actually:

> **When do you need concurrency, and when does it just make things more complicated?**

Especially in AI applications and service-side code, many tasks are fundamentally I/O-bound, which is exactly the kind of scenario `asyncio` is best at.
:::

## Learning Objectives

- Understand the difference between I/O-bound and CPU-bound tasks
- Understand why `asyncio` is suitable for many service scenarios
- Learn to use `gather`, `Semaphore`, and timeout control to organize concurrency
- Build the mindset that “concurrency is a tool, not the default answer”

---

## Why do so many Python projects end up using asyncio?

### Because many tasks are just “waiting”

For example:

- waiting for an HTTP response
- waiting for a database response
- waiting for file reads

For these tasks, the real time cost is not CPU computation,
but waiting for external I/O.

### The core value of asyncio

It lets you move on to other tasks
while one task is waiting.

This is especially suitable for:

- scraping
- API orchestration
- multi-tool services
- batch requests

### An analogy

Synchronous code is like one counter serving one person at a time.
Asynchronous code is more like taking a ticket and waiting in line: while the counter is waiting for one person’s documents, it can handle someone else’s request first.

---

## Let’s first look at a minimal asynchronous concurrency example

```python
import asyncio


async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name} done"


async def main():
    results = await asyncio.gather(
        fetch("task_a", 0.2),
        fetch("task_b", 0.1),
    )
    print(results)


asyncio.run(main())
```

### What is this code really trying to show?

It shows that:

- two waiting tasks can proceed concurrently

If you switch to synchronous serial execution,
the total time will be closer to:

- `0.2 + 0.1`

instead of:

- `max(0.2, 0.1)`

### Why is this so common in AI applications?

Because many applications do all of these at the same time:

- retrieval
- multiple API calls
- reading and writing multiple services

These tasks are not CPU-heavy; they are waiting-heavy.

---

## Why is more concurrency not always better?

### Too much concurrency can overwhelm upstream services

If you send 1000 requests at once,
it may not be faster; instead, you may get:

- rate limited
- more timeouts
- upstream cascading failures

### So you often need a concurrency limit

One of the simplest tools is:

- `Semaphore`

It limits how many tasks can run at the same time.

```python
import asyncio


semaphore = asyncio.Semaphore(2)


async def bounded_fetch(name, delay):
    async with semaphore:
        print("start", name)
        await asyncio.sleep(delay)
        print("end", name)
        return name


async def main():
    tasks = [bounded_fetch(f"task_{i}", 0.2) for i in range(5)]
    results = await asyncio.gather(*tasks)
    print(results)


asyncio.run(main())
```

### What is the most important thing to learn from this code?

Concurrency is not only about “can they run together,”
but also about:

- how many should run together at once

This is exactly one of the key control points in many production services.

---

## Why are timeout and cancellation also important?

### Without a timeout, slow tasks can hang forever

This is very dangerous when you depend on many external systems.
A common approach is:

- `asyncio.wait_for(...)`

```python
import asyncio


async def slow_task():
    await asyncio.sleep(2)
    return "done"


async def main():
    try:
        result = await asyncio.wait_for(slow_task(), timeout=0.5)
        print(result)
    except asyncio.TimeoutError:
        print("timeout")


asyncio.run(main())
```

### Why is this especially important for Agent systems?

Because Agents often depend on:

- external tools
- upstream models
- retrieval systems

Without timeouts, the whole chain can easily get stuck.

---

## When should you not prioritize asyncio?

### Pure CPU-bound tasks

For example:

- heavy numerical computation
- large-scale image transformations

These tasks are better suited for:

- multiprocessing
- native high-performance libraries

### Your team is not ready for asynchronous complexity yet

Async code introduces:

- debugging complexity
- state management difficulty

If the scenario does not need it, there is no need to force it.

### Synchronous code is already simple enough and stable enough

For small scripts and small tasks,
synchronous code can actually be clearer.

---

## The most common misconceptions

### Misconception 1: concurrency always means faster

Not necessarily.
The key is whether the task is I/O-bound.

### Misconception 2: `async` should be added everywhere

Async is a technique, not a style badge.

### Misconception 3: knowing `gather` means you know asyncio

In real projects, what often matters more is:

- rate limiting
- timeouts
- error handling

---

## Summary

The most important takeaway from this section is not to memorize `asyncio` as a list of APIs,
but to build a practical judgment:

> **If a task is mainly waiting on I/O, asynchronous concurrency can usually improve throughput significantly; but in production, it must also be paired with concurrency limits, timeouts, and error control.**

Once you have this judgment in place, service-side concurrency code will make much more sense later on.

---

## Exercises

1. Change `Semaphore(2)` to `Semaphore(1)` and `Semaphore(5)`, and compare how the log order changes.
2. Think about why many Agent / API orchestration tasks are naturally suitable for asyncio.
3. Why is timeout control in asynchronous systems just as important as `gather`?
4. Give one example of a task that you think is **not** suitable for solving with asyncio first.
