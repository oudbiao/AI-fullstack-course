---
title: "7.1.3 Word Embeddings and Semantic Representation"
sidebar_position: 2
description: "From one-hot to dense vectors, and then to sentence representations and contextual representations, understand why models can turn “semantically similar” into distance relationships in vector space."
keywords: [embedding, semantic representation, cosine similarity, sentence embedding, contextual embedding]
---

# 7.1.3 Word Embeddings and Semantic Representation

![Embedding semantic space diagram](/img/course/embedding-semantic-space-en.png)

:::tip Section focus
Tokenizer solves:

- How to split text

Embedding solves:

- How to turn the split tokens into semantic vectors

When many people first encounter embedding, they think of it as:

- Assigning each word a string of numbers

That is not enough.
The real key is:

> **These numbers are not assigned arbitrarily. They gradually form a “semantic space,” bringing similar words and similar sentences closer together in the space.**
:::

## Learning Goals

- Understand the essential difference between one-hot and dense embedding
- Understand why similar semantics can be reflected in vector distance
- Understand the step-by-step progression from word vectors to sentence vectors to contextual representations
- Use a runnable example to see how embedding supports similarity calculation

---

## Why can’t we directly use one-hot to represent words?

### one-hot is clean, but it does not express semantic relationships

Suppose the vocabulary contains four words:

- `refund`
- `return`
- `password`
- `banana`

The one-hot representations would look like this:

- `refund` -> `[1, 0, 0, 0]`
- `return` -> `[0, 1, 0, 0]`
- `password` -> `[0, 0, 1, 0]`

The problem is:

- `refund` and `return` are semantically close
- `refund` and `banana` are semantically far apart

But in one-hot space, they are equally “far” from each other.

This means:

> **one-hot can distinguish identity, but it cannot express similarity.**

### The core value of dense embedding

What embedding tries to do is:

- Make semantically similar words have similar vectors

For example:

- `refund` and `return`
- `reset` and `recover`

They can be placed closer together in vector space.

This is what really matters about embedding:

- Not just encoding
- But representation

### An analogy: putting words on a map

You can think of embedding as map coordinates.

- one-hot is more like an ID number: it can only tell people apart
- embedding is more like a map location: it can not only distinguish, but also show who is closer to whom

Once you have this semantic map,
the model can more easily discover:

- Which words often appear in similar contexts
- Which sentences express similar meanings

![Semantic space map from one-hot to dense embedding](/img/course/ch07-embedding-onehot-dense-map-en.png)

:::tip Reading tip
The key point of this diagram is the contrast: one-hot is like an ID number, which can only tell whether two words are the same; dense embedding is like map coordinates, which can express who is closer to whom. From here on, text truly enters a computable semantic space.
:::

---

## Why do word vectors have semantics?

### Because they are learned in context

Embedding is not manually defined.
It is usually learned gradually during training.

If two words often appear in similar contexts,
the model will tend to learn them as nearby vectors.

This is the classic distributional hypothesis:

> **A word’s meaning is largely determined by the context in which it appears.**

### Similar semantics does not mean exact synonymy

Being close in vector space only means:

- Similar usage
- Similar contextual distribution

It does not mean:

- Fully interchangeable

For example:

- `doctor` and `hospital`

may also be close, because they often appear together.
So “close” in embedding is more about being close in distributional meaning.

### From words to sentences, representations can keep building upward

When you combine multiple token vectors,
you can get:

- Phrase vectors
- Sentence vectors
- Paragraph vectors

That is also why embedding is not only used for word similarity,
but is also widely used in:

- Retrieval
- Clustering
- Classification
- RAG

---

## First, run a truly semantic comparison example

The code below does three things:

1. Assign a small embedding to several words
2. Compute cosine similarity between words
3. Use the “average token vector” method to get sentence vectors, then compare sentence similarity

```python
from math import sqrt

embeddings = {
    "refund": [0.90, 0.80, 0.10],
    "return": [0.88, 0.78, 0.12],
    "password": [0.10, 0.20, 0.95],
    "reset": [0.12, 0.18, 0.92],
    "order": [0.75, 0.70, 0.15],
    "banana": [0.05, 0.95, 0.10],
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def sentence_embedding(tokens, embedding_table):
    valid = [embedding_table[token] for token in tokens if token in embedding_table]
    dim = len(valid[0])
    return [sum(vec[i] for vec in valid) / len(valid) for i in range(dim)]


print("refund vs return  :", round(cosine(embeddings["refund"], embeddings["return"]), 3))
print("refund vs password:", round(cosine(embeddings["refund"], embeddings["password"]), 3))

query_a = ["reset", "password"]
query_b = ["password", "reset"]
query_c = ["refund", "order"]

vec_a = sentence_embedding(query_a, embeddings)
vec_b = sentence_embedding(query_b, embeddings)
vec_c = sentence_embedding(query_c, embeddings)

print("query_a vs query_b:", round(cosine(vec_a, vec_b), 3))
print("query_a vs query_c:", round(cosine(vec_a, vec_c), 3))
```

### What is this code showing?

It shows two levels of meaning:

First level:

- Similar words such as `refund` and `return` are also closer in vector space

Second level:

- After aggregating token vectors, sentences can also be compared by similarity in vector space

This is why embedding can support semantic retrieval and recall.

### Why are `query_a` and `query_b` so close?

Because they differ only in word order,
and after averaging the vectors, the resulting representation is basically the same.

This also reveals a limitation of simple averaging:

- It hardly cares about order

So early static sentence vectors were useful,
but their expressive power was limited.

### Why is this code still valuable?

Because it captures the most essential intuition of embedding:

> **“Semantic closeness” can be turned into “vector closeness.”**

No matter how much more complex later sentence embedding models, dual-encoder retrieval models, or LLM embedding APIs become,
they still fundamentally rely on this idea.

---

## From word vectors to contextual representations

### Early embedding: one word usually has one fixed vector

For example, in traditional word vectors:

- `bank`

no matter whether it appears in:

- river bank
- bank account

it usually has the same vector.

This creates ambiguity problems.

### Contextual representations: the same word can change in different contexts

By the Transformer era,
word representations are no longer completely fixed, but can change according to context.

In other words:

- the vector of `bank` in a financial context
- the vector of `bank` in a riverbank context

can be different.

This is one of the most important advances of contextual representations.

![Contextual representation disambiguates polysemy diagram](/img/course/ch07-contextual-embedding-sense-map-en.png)

:::tip Reading tip
When reading this diagram, focus only on the word `bank`: in `bank account` it moves closer to financial concepts, while in `river bank` it moves closer to geographic concepts. Transformer contextual representations let the same token no longer have only one fixed coordinate.
:::

### A simple context simulation

The example below is not a real Transformer,
but it can help you build the intuition of “the same word, different vectors.”

```python
base_bank = [0.50, 0.50, 0.50]
finance_context = [0.30, -0.10, 0.20]
river_context = [-0.20, 0.25, -0.10]

bank_in_finance = [a + b for a, b in zip(base_bank, finance_context)]
bank_in_river = [a + b for a, b in zip(base_bank, river_context)]

print("bank in finance:", [round(x, 2) for x in bank_in_finance])
print("bank in river  :", [round(x, 2) for x in bank_in_river])
```

Its purpose is not to simulate a real model,
but to help you remember:

- Static embedding: one word, one vector
- Contextual representation: the same word changes with context

---

## What are embeddings used for in real projects?

### Retrieval and RAG

After encoding both questions and documents into vectors,
you can perform:

- similarity-based retrieval

This is the foundation of many RAG systems.

### Semantic clustering and deduplication

If the vectors of two pieces of text are very close,
they often mean the texts are semantically similar too.

This can be used for:

- Text clustering
- FAQ merging
- Near-duplicate detection

### As input features for downstream tasks

Many classification, matching, and ranking tasks first convert text into embeddings,
then train heads on top of them or use them for similarity scoring.

---

## The most easily misunderstood parts of embedding

### Misconception 1: If vectors are close, they must be synonyms

Not necessarily.
It more likely means:

- Similar distribution
- Related usage

### Misconception 2: The simpler the sentence vector, the better

Averaging word vectors is intuitive,
but it easily loses:

- Order
- Negation
- Long-range dependencies

### Misconception 3: Having embeddings means language is understood

Embedding is only a representation layer,
and true understanding still requires:

- Context modeling
- Task objectives
- Training data

---

## Summary

The most important thing in this section is not memorizing a few similarity formulas,
but building this judgment:

> **The core value of embedding is turning discrete tokens into a vector space that is comparable, composable, and capable of reflecting semantic relationships.**

As long as you hold on to this main thread,
later topics such as:

- Sentence embeddings
- Retrieval models
- RAG
- Contextual representations

will feel much more natural.

---

## Exercises

1. Change the word vectors in the example and observe which words become closer or farther apart.
2. Why can averaging word vectors build an initial intuition, but is not suitable for expressing all semantic phenomena?
3. Explain in your own words: what is the biggest difference between static embedding and contextual representation?
4. Think about it: if you are building FAQ retrieval, what problem can embedding help you solve first?
