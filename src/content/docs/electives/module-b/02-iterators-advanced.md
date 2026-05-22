---
title: "E.B.2 Advanced Iterators and Generators"
description: "Use generators to process data streams step by step instead of loading everything into memory."
sidebar:
  order: 9
head:
  - tag: meta
    attrs:
      name: keywords
      content: "iterator, generator, yield, yield from, lazy evaluation, streaming"
---
![Generator streaming pipeline diagram](/img/course/elective-generator-stream-pipeline-en.webp)

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

Run this small standalone demo:

```python
def flatten(groups):
    for group in groups:
        yield from group

pipeline = [
    ["error db timeout", "error auth failed"],
    ["error model busy"],
]

for item in flatten(pipeline):
    print(item)
```

Expected output:

```text
error db timeout
error auth failed
error model busy
```

This expresses “send every item inside each group outward” more clearly than a nested loop.

## When Generators Help

Use generators when:

1. The input may be large.
2. You process records one by one.
3. You want to connect read/filter/transform/batch steps.
4. You do not need random access to all items.

Prefer a list when the data is small and repeated access makes the code simpler.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
python_pattern: decorator, iterator, generator, concurrency primitive, or metaprogramming hook
code_artifact: minimal runnable example plus printed output
use_case: where this pattern improves an AI app, pipeline, tool, or server
failure_check: hidden side effects, unreadable abstraction, race condition, or overengineering
Expected_output: small advanced-Python example with a practical AI-system use note
```

## Common Mistakes

- Expecting a generator to be reusable after it has been consumed.
- Assuming generators are always faster; their main benefit is often memory and structure.
- Making a simple list transformation harder to read by forcing `yield` everywhere.

## Practice

Modify `batch` so it also prints `batch_id`. Then change the input events and confirm the pipeline still works without changing the later steps.

<details>
<summary>Reference implementation and walkthrough</summary>

One acceptable answer is to enumerate batches at the output edge:

```python
for batch_id, group in enumerate(batch(normalized, size=2), start=1):
    print(batch_id, group)
```

This keeps the earlier reader, filter, and normalizer unchanged. If changing input events only changes the printed groups, while the pipeline structure stays intact, the exercise worked. The core lesson is that generator pipelines should let you swap data without rewriting every downstream step.

</details>
