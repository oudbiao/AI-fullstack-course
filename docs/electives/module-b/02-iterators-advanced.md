---
title: "E.B.2 Advanced Iterators and Generators"
sidebar_position: 9
description: "From lazy evaluation and streaming processing to generator pipelines and `yield from`, understand why iterators and generators are especially well-suited for data and service code."
keywords: [iterator, generator, yield, yield from, lazy evaluation, streaming]
---

# E.B.2 Advanced Iterators and Generators

:::tip Section Overview
Iterators and generators are often misunderstood as just “syntax tricks.”
But in real-world engineering, their most important value is actually this:

> **They let data be produced and consumed step by step, instead of loading everything into memory at once.**

This is very common in data processing, log streams, batch jobs, and server-side code.
:::

![Generator streaming pipeline diagram](/img/course/elective-generator-stream-pipeline-en.png)

## Learning Objectives

- Understand the core value of iterators and generators in engineering
- Understand why lazy evaluation can significantly reduce memory pressure
- Learn how to build simple generator pipelines
- Use runnable examples to master when to use `yield` and `yield from`

---

## Why do engineering teams like generators so much?

### Because much data is a “stream,” not a “chunk”

For example:

- Log streams
- Reading files line by line
- Network request results
- Large-scale sample processing

If you always read everything into a list first,
it can easily lead to:

- Memory waste
- Increased latency

### The core value of generators

They let you:

- Produce the next value only when needed

This is lazy evaluation.

### An analogy

A list is like preparing a large table of dishes all at once.
A generator is like serving dishes one by one to each table.

If there are many guests and many dishes, the second approach usually uses fewer resources.

---

## First, look at a sliding-window generator

```python
def sliding_window(nums, size):
    for i in range(len(nums) - size + 1):
        yield nums[i : i + size]


for window in sliding_window([1, 2, 3, 4, 5], 3):
    print(window)
```

### Why is this code valuable?

Because it already shows the essence of a generator:

- It does not return all windows at once
- It produces them one by one

### Where is this kind of pattern common?

For example:

- Time-series windows
- NLP chunking
- Batch slicing

---

## Generator pipelines: chaining multiple steps together

In engineering, what is more common is not a single generator,
but a pipeline made of multiple generators.

```python
def read_lines():
    lines = [
        "INFO request ok",
        "ERROR db timeout",
        "INFO cache hit",
        "ERROR auth failed",
    ]
    for line in lines:
        yield line


def filter_errors(lines):
    for line in lines:
        if "ERROR" in line:
            yield line


def normalize(lines):
    for line in lines:
        yield line.lower()


pipeline = normalize(filter_errors(read_lines()))

for item in pipeline:
    print(item)
```

### What is this example mainly trying to teach?

A lot of data processing in engineering can be broken into:

- Reading
- Filtering
- Transforming

If each step generates a full list,
the pipeline becomes heavier;
using a generator pipeline is more natural.

### Why is this useful for AI engineering too?

Because you often work with:

- Sample streams
- Log streams
- Retrieval result streams

These scenarios are naturally suited to generator pipelines.

---

## Why is `yield from` worth learning?

### What problem does it solve?

When a generator simply wants to forward another iterable outward,
`yield from` makes the code clearer.

```python
def chunk_batches():
    yield [1, 2]
    yield [3, 4]


def flatten():
    for batch in chunk_batches():
        yield from batch


print(list(flatten()))
```

### Why is it more worth learning than a nested loop?

Because it expresses intent more clearly:

- “Continue yielding the contents of the sub-iterator outward”

---

## The easiest pitfalls to fall into

### Misconception 1: Generators are always faster

They usually save memory,
but that does not mean they are absolutely faster in every scenario.

### Misconception 2: Generators can only be iterated once

In many cases, this is a design feature, not a bug.
If you need to consume it repeatedly, you must create it again.

### Misconception 3: Using generators just for the sake of using generators

If the data size is small and the logic is simple,
a plain list may actually be easier to read.

---

## Summary

The most important thing in this section is to build an engineering intuition:

> **Generators and iterators are best suited for data flows that are produced step by step and consumed step by step. Their value is mainly in saving memory, reducing intermediate copies, and organizing pipelines.**

Once you clearly understand this layer,
you will naturally think of them when doing log processing, sample pipelines, and streaming services.

---

## Exercises

1. Modify `sliding_window` so it outputs data chunks with a fixed batch size.
2. Use `yield from` to write another example that flattens a nested list.
3. Think about when a list is more appropriate and when a generator is more appropriate.
4. Can you rewrite an existing data processing function as a generator pipeline?
