---
title: "7.1.4 Pretrained Language Models at a Glance"
sidebar_position: 3
description: "Start from the idea of “learn general patterns on large-scale corpora first, then transfer to specific tasks,” and understand why pretrained models have become the shared foundation of modern NLP and large models."
keywords: [pretrained models, transfer learning, BERT, GPT, T5, foundation models]
---

# 7.1.4 Pretrained Language Models at a Glance

:::tip Section Overview
In the era of large models, the word “pretraining” appears almost everywhere.
But when many beginners hear it for the first time, they understand it as a very vague idea:

- First learn once on big data

That is of course correct, but not enough.

The real judgment to build is:

> **Why do people no longer start from scratch for every NLP task, but instead first train a general-purpose foundation and then transfer from it?**

This lesson is a quick entry into that intuition.
:::

## Learning Objectives

- Understand what pretraining means, and what transfer and downstream adaptation mean
- Understand why pretrained models can be used for “many tasks with one model”
- Distinguish the broad directions of encoder-only, decoder-only, and encoder-decoder models
- Understand the idea of “shared foundation + different task heads” through a runnable example

---

## Why Have Pretrained Models Become the Mainstream in Modern NLP?

### Because Many Tasks Essentially Share Language Ability

Whether you are doing:

- sentiment classification
- question answering
- summarization
- dialogue
- retrieval

they all rely on some common fundamentals:

- word meaning understanding
- syntactic relationships
- context modeling
- common sense and language patterns

If every task had to learn these abilities from scratch,
the cost would be very high.

### The Core Idea of Pretraining

So people began by doing one thing first:

- train a base model on massive general-purpose corpora

Let it first learn:

- general language regularities
- general representations
- basic knowledge distribution

Then transfer it to specific tasks.

It is like:

- first reading most general education textbooks
- then doing specialized training for specific subjects

### Why Is This Much Better Than “Training Each Task from Scratch”?

Because you do not need to relearn language itself every time.
Downstream tasks only need to do the following on top of an existing foundation:

- task head training
- fine-tuning
- Prompt adaptation
- retrieval augmentation

This greatly lowers the barrier.

---

## What Do Pretrained Models Actually Give Us?

### A Foundation That “Already Knows a Bit of Language”

A model initialized from random weights knows nothing at the beginning.
A pretrained model, however, has at least learned some of the following:

- grammar patterns
- collocation relationships
- high-frequency facts
- common task formats

This means downstream tasks are no longer starting from complete zero.

### Reusable Representations

What makes many pretrained models so valuable is not just that they “can answer,”
but that they can output a set of relatively good hidden representations.

These representations can be used by downstream tasks for:

- classification
- retrieval
- matching
- ranking

### The Possibility of Transfer Learning

The core of transfer learning is:

> **Learn general abilities on large tasks, and adapt with little effort on smaller tasks.**

This is also why, once pretrained models appeared,
the entire NLP workflow was rewritten.

---

## First Run a “Shared Foundation + Two Task Heads” Example

The code below does not train a real large model,
but it captures the most important structural intuition of pretrained models:

- there is a shared encoder
- the encoder learns general representations
- different tasks attach different heads on top of it

```python
from math import sqrt

word_vectors = {
    "refund": [0.9, 0.8, 0.1],
    "order": [0.8, 0.7, 0.2],
    "password": [0.1, 0.2, 0.9],
    "reset": [0.1, 0.1, 0.95],
    "great": [0.7, 0.2, 0.1],
    "bad": [0.2, 0.8, 0.1],
}


def encode(text):
    tokens = text.lower().split()
    valid = [word_vectors[token] for token in tokens if token in word_vectors]
    dim = len(valid[0])
    return [sum(vec[i] for vec in valid) / len(valid) for i in range(dim)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    exps = [2.71828 ** s for s in scores]
    total = sum(exps)
    return [x / total for x in exps]


# Same foundation, two different task heads
intent_head = {
    "refund_intent": [1.0, 0.9, 0.1],
    "password_intent": [0.1, 0.2, 1.0],
}

sentiment_head = {
    "positive": [1.0, 0.2, 0.0],
    "negative": [0.1, 1.0, 0.0],
}


def classify(vector, head):
    labels = list(head.keys())
    scores = [dot(vector, head[label]) for label in labels]
    probs = softmax(scores)
    best = max(zip(labels, probs), key=lambda x: x[1])
    return best, dict(zip(labels, [round(p, 3) for p in probs]))


text_a = "refund order"
text_b = "reset password"
text_c = "great refund"
text_d = "bad refund"

for text in [text_a, text_b]:
    vec = encode(text)
    best, probs = classify(vec, intent_head)
    print("intent:", text, "->", best, probs)

for text in [text_c, text_d]:
    vec = encode(text)
    best, probs = classify(vec, sentiment_head)
    print("sentiment:", text, "->", best, probs)
```

### What Real Idea Does This Code Correspond To?

It corresponds to one of the most important workflows in the pretraining era:

1. first have a shared language foundation
2. reuse this foundation across different tasks
3. replace only the head on top, or do a small amount of adaptation

This is why one pretrained model can be used for many tasks.

### Why Is This Better Than “Relearning Word Vectors for Every Task”?

Because the foundation has already learned a lot of general information.
Downstream tasks do not need to start from scratch to understand:

- `refund` is probably related to after-sales service
- `reset password` is probably related to login problems

They only need to perform a directed mapping on top of the foundation.

### What Would a “Head” Be in the Real World?

In a real model, it might be:

- a classification layer
- a generation head
- a retrieval projection layer
- a token-level prediction head

The idea is always the same:

- shared foundation
- specialized task heads

---

## What Are the Main Pretrained Model Directions?

### Encoder-only: More Focused on Understanding and Representation

Representative models:

- BERT

These models are usually more suitable for:

- classification
- extraction
- matching
- retrieval encoding

### Decoder-only: More Focused on Generation

Representative models:

- GPT
- LLaMA
- Qwen

These models are usually more suitable for:

- dialogue
- writing
- code generation
- open-ended completion

### Encoder-Decoder: Better for Input-to-Output Tasks

Representative models:

- T5
- BART

These models are naturally suitable for:

- summarization
- translation
- paraphrasing
- answer generation

---

## After Pretraining, How Else Can We Adapt to Tasks?

### Linear Probing / Task-Head Fine-Tuning

The lightest approach is:

- freeze the foundation
- train only the top head

This is very common for small tasks.

### Full Fine-Tuning

Let the entire model update together.
The advantage is flexibility, but the disadvantage is high cost.

### Parameter-Efficient Fine-Tuning

For example:

- LoRA
- Adapter

This is a very important direction in the large-model era,
because it greatly lowers the barrier to adapting a task on a large foundation.

### Prompt and RAG

Not every task needs the model parameters to be changed.
Many problems can also be solved through:

- Prompt
- RAG
- tool calling

So the value of pretrained models is not just “giving you a model that can be fine-tuned,”
but also “giving you a reusable foundation.”

---

## The Most Common Misunderstandings

### Misunderstanding 1: Pretrained Models Know Everything

They have a strong foundation, but that does not mean:

- their knowledge is always up to date
- their behavior is always stable
- they are perfectly adapted to your business right away

### Misunderstanding 2: Once You Use a Pretrained Model, You No Longer Need to Care About Data

That is not true.
Whether you are fine-tuning or evaluating, data quality still determines the final result.

### Misunderstanding 3: As Long as the Model Is Bigger, It Must Be Better Than a Smaller Model for the Current Task

Sometimes the task is very simple,
or the cost is highly sensitive,
and a large model is not necessarily the best solution.

---

## Summary

The most important thing in this lesson is not to memorize how many model names there are,
but to build a core modern NLP judgment:

> **The value of pretrained models lies in first learning a general language foundation from large corpora, and then transferring that foundation to many different tasks.**

Once this main line is established,
when you later learn:

- fine-tuning
- Prompt
- RAG
- Agent

you will better understand that you are “using an existing foundation,” rather than starting from scratch every time.

---

## Exercises

1. Explain in your own words: why can a pretrained model be reused for multiple different tasks?
2. Referring to the example, add a new task head to the shared foundation, such as “topic classification.”
3. Why do we say a pretrained model provides a foundation, not a magic button that automatically solves every task?
4. Think about it: if your task data is very limited, what is the biggest advantage of a pretrained model compared with training from scratch?
