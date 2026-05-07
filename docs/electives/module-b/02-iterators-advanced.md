---
title: "E.B.2 Advanced Iterators and Generators"
sidebar_position: 9
description: "Use generators to process data streams step by step instead of loading everything into memory."
keywords: [iterator, generator, yield, yield from, lazy evaluation, streaming]
---

# E.B.2 Advanced Iterators and Generators

![Generator streaming pipeline diagram](/img/course/elective-generator-stream-pipeline-en.png)

Generators are useful when data arrives as a stream: logs, files, API pages, sample batches, retrieval results, or model outputs. They produce one item at a time, so you avoid building unnecessary intermediate lists.

## What You Need

- Python 3.10+
- No external packages
- Basic understanding of `for` loops

## Key Terms

- **Iterator**: an object that can produce the next value.
- **Generator**: a function that uses `yield` to produce values lazily.
- **Lazy evaluation**: compute the next value only when needed.
- **Pipeline**: small processing steps chained together.
- **`yield from`**: forward values from another iterable.

## Run A Streaming Pipeline

Create `generator_pipeline.py`:

```python
def read_events():
    events = [
        "INFO request ok",
        "ERROR db timeout",
        "INFO cache hit",
        "ERROR auth failed",
        "ERROR model busy",
    ]
    for event in events:
        yield event


def filter_errors(events):
    for event in events:
        if event.startswith("ERROR"):
            yield event


def normalize(events):
    for event in events:
        yield event.lower()


def batch(items, size):
    group = []
    for item in items:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


pipeline = batch(normalize(filter_errors(read_events())), size=2)

for group in pipeline:
    print(group)
```

Run it:

```bash
python generator_pipeline.py
```

Expected output:

```text
['error db timeout', 'error auth failed']
['error model busy']
```

The pipeline reads, filters, normalizes, and batches without creating a full list at every step.

## Use `yield from`

Add this helper:

```python
def flatten(groups):
    for group in groups:
        yield from group
```

Then change the final loop:

```python
for item in flatten(pipeline):
    print(item)
```

This expresses “send every item inside each group outward” more clearly than a nested loop.

## When Generators Help

Use generators when:

1. The input may be large.
2. You process records one by one.
3. You want to connect read/filter/transform/batch steps.
4. You do not need random access to all items.

Prefer a list when the data is small and repeated access makes the code simpler.

## Common Mistakes

- Expecting a generator to be reusable after it has been consumed.
- Assuming generators are always faster; their main benefit is often memory and structure.
- Making a simple list transformation harder to read by forcing `yield` everywhere.

## Practice

Modify `batch` so it also prints `batch_id`. Then change the input events and confirm the pipeline still works without changing the later steps.
