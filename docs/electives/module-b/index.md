---
title: "E.B Advanced Python Roadmap"
sidebar_position: 0
description: "A concise hands-on roadmap for the Advanced Python elective: decorators, generators, asyncio, and metaprogramming for traceable engineering pipelines."
---

# E.B Advanced Python Roadmap

Use this elective when your prototype starts repeating logic, waiting on slow calls, streaming data, or registering tools dynamically.

## See the Engineering Path First

![Advanced Python Topics Module Map](/img/course/elective-python-advanced-module-map-en.webp)

![Generator stream pipeline](/img/course/elective-generator-stream-pipeline-en.webp)

Advanced Python is useful when it makes code more observable, reusable, and easier to control.

## Run the Smallest Async Trace

```python
import asyncio

async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name}:done"

async def main():
    results = await asyncio.gather(
        fetch("retrieval", 0.1),
        fetch("rerank", 0.05),
    )
    print(results)

asyncio.run(main())
```

Expected output:

```text
['retrieval:done', 'rerank:done']
```

This is the smallest async habit: launch independent work, wait for all results, then keep a trace.

## Learn in This Order

| Step | Lesson | Practice Output |
|---|---|---|
| 1 | [E.B.1 Decorators](./01-decorators-advanced.md) | Add timing or logging without changing business code |
| 2 | [E.B.2 Iterators and Generators](./02-iterators-advanced.md) | Stream rows without loading everything at once |
| 3 | [E.B.3 Concurrency](./03-concurrency.md) | Run async tasks with timeout and cancellation thinking |
| 4 | [E.B.4 Metaprogramming](./04-metaprogramming.md) | Register tools or handlers explicitly |

## Pass Check

You pass this module when you can build one traceable pipeline that uses a decorator, generator, async call, or registry, and can explain why the code became easier to debug.
