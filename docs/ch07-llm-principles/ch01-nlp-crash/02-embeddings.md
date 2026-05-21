---
title: "7.1.3 Word Embeddings and Semantic Representation"
sidebar_position: 2
description: "Use small runnable labs to turn tokens into dense vectors, compare cosine similarity, build a tiny semantic retriever, and understand contextual representations."
keywords: [embedding, semantic representation, cosine similarity, sentence embedding, contextual embedding, retrieval]
---

# 7.1.3 Word Embeddings and Semantic Representation

![Embedding semantic space diagram](/img/course/embedding-semantic-space-en.webp)

:::tip One Sentence
Tokenization gives the model discrete IDs. Embedding turns those IDs into vectors so the model can compare, combine, and move meaning through layers.
:::

## The Mental Model

One-hot IDs can tell words apart, but they cannot tell which words are related. Dense embeddings place tokens in a vector space:

```text
token id -> embedding table lookup -> dense vector
```

In that space:

- nearby vectors often mean related usage;
- cosine similarity measures direction similarity;
- sentence vectors are usually produced by pooling token vectors;
- contextual models can make the same token move depending on nearby words.

## From One-Hot to Dense Vectors

![Semantic space map from one-hot to dense embedding](/img/course/ch07-embedding-onehot-dense-map-en.webp)

With one-hot vectors, every different word is equally different:

```text
refund   -> [1, 0, 0, 0]
return   -> [0, 1, 0, 0]
password -> [0, 0, 1, 0]
banana   -> [0, 0, 0, 1]
```

Dense vectors can encode useful geometry:

```text
refund  and return   -> close
password and reset   -> close
refund  and password -> far
```

This geometry is learned from data, not hand-written. Words that appear in similar contexts tend to get similar vectors.

## Lab 1: Compare Word Similarity

Run this tiny embedding table. The numbers are hand-made for learning, but the operations are the same as real embedding retrieval.

```python
from math import sqrt

embeddings = {
    "refund": [0.90, 0.80, 0.10],
    "return": [0.88, 0.78, 0.12],
    "password": [0.10, 0.20, 0.95],
    "reset": [0.12, 0.18, 0.92],
    "order": [0.75, 0.70, 0.15],
    "banana": [0.05, 0.95, 0.10],
    "policy": [0.82, 0.74, 0.18],
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


print("refund vs return  :", round(cosine(embeddings["refund"], embeddings["return"]), 3))
print("refund vs password:", round(cosine(embeddings["refund"], embeddings["password"]), 3))
print("password vs reset :", round(cosine(embeddings["password"], embeddings["reset"]), 3))
```

Expected output:

```text
refund vs return  : 1.0
refund vs password: 0.293
password vs reset : 1.0
```

Interpretation:

- high cosine means similar direction, not identical meaning;
- `refund` and `return` are close because this toy table puts them in the same customer-service region;
- `password` and `reset` are close for the same reason;
- `refund` and `password` are far because they serve different intents.

## Lab 2: Build a Tiny Semantic Retriever

Now average token vectors to create sentence vectors, then rank three documents for a query.

```python
def mean_embedding(tokens):
    vectors = [embeddings[token] for token in tokens if token in embeddings]
    dim = len(vectors[0])
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(dim)]


query = mean_embedding(["reset", "password"])
documents = {
    "A refund policy": ["refund", "policy"],
    "B password reset": ["password", "reset"],
    "C banana return": ["banana", "return"],
}

ranked = sorted(
    (
        (name, cosine(query, mean_embedding(tokens)))
        for name, tokens in documents.items()
    ),
    key=lambda item: item[1],
    reverse=True,
)

for name, score in ranked:
    print(f"{name}: {score:.3f}")
```

Expected output:

```text
B password reset: 1.000
C banana return: 0.335
A refund policy: 0.333
```

This is the core of vector retrieval:

```text
query text -> query vector -> compare with document vectors -> top-k results
```

Real RAG systems use stronger embedding models and vector databases, but the logic is still similarity ranking.

## Why Averaging Is Useful but Limited

Mean pooling is easy to understand, but it loses important information:

- word order;
- negation;
- emphasis;
- long-range dependency;
- which token should matter most.

For example, `reset password` and `password reset` become identical in the toy retriever. That is acceptable for a first intuition, but not enough for reasoning-heavy tasks.

## Contextual Representations

![Contextual representation disambiguates polysemy diagram](/img/course/ch07-contextual-embedding-sense-map-en.webp)

Static embeddings usually give one word one vector. Contextual models make the vector depend on surrounding words:

```text
bank account -> bank moves toward finance
river bank   -> bank moves toward geography
```

Run this small simulation:

```python
base_bank = [0.50, 0.50, 0.50]
finance_context = [0.30, -0.10, 0.20]
river_context = [-0.20, 0.25, -0.10]

bank_in_finance = [a + b for a, b in zip(base_bank, finance_context)]
bank_in_river = [a + b for a, b in zip(base_bank, river_context)]

print("bank in finance:", [round(x, 2) for x in bank_in_finance])
print("bank in river  :", [round(x, 2) for x in bank_in_river])
```

Expected output:

```text
bank in finance: [0.8, 0.4, 0.7]
bank in river  : [0.3, 0.75, 0.4]
```

![Embedding lab output result map](/img/course/ch07-embedding-cosine-retrieval-context-result-map-en.webp)

This is not a real Transformer. It is a memory hook: the same token can end up with different representations after context is mixed in.

## Project Uses

| Use case | What embedding provides | Watch out for |
|---|---|---|
| RAG retrieval | find semantically related chunks | bad chunks or stale metadata still hurt answers |
| FAQ clustering | merge similar questions | close does not always mean duplicate |
| Deduplication | find near-duplicate content | paraphrases and templates can confuse scores |
| Classification | turn text into features | labels and calibration still matter |
| Recommendation | compare users, items, or queries | popularity bias can dominate similarity |

## Debugging Checklist

- Normalize vectors before cosine similarity if your library does not do it.
- Print top-k scores, not only top-1; a weak margin means retrieval is uncertain.
- Inspect false positives: related terms are not always correct answers.
- Compare static, sentence, and contextual embeddings for the same data.
- For multilingual projects, test cross-language pairs before assuming the embedding model aligns languages well.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
vectors: at least three text embeddings or toy vectors
similarity_check: closest pair and score
retrieval_result: top match for one query
limitation: averaging or similarity misses context/negation/order
next_use: this becomes retrieval evidence in Chapter 8
```

## Exercises

1. Move `banana` closer to `password` in the toy table. How does retrieval break?
2. Add a document `D recover account` and create vectors for `recover` and `account`.
3. Make a query `refund order`. Which document should rank first?
4. Explain why `doctor` and `hospital` may be close even though they are not synonyms.
5. In a RAG project, what evidence would you collect to prove your embedding model is good enough?

<details>
<summary>Project reference and review notes</summary>

1. If `banana` moves close to `password`, similarity search may retrieve fruit-related text for account-recovery queries. The failure is not random; it comes from bad geometry.
2. `recover` and `account` should be placed near password/account-support concepts, not near unrelated commerce or fruit concepts. The added document should become a plausible match for account-recovery queries.
3. `refund order` should rank the refund/order document first if the embedding space captures both commerce and refund intent.
4. `doctor` and `hospital` are close because they often appear in the same domain. Similarity can mean topical relation, not strict synonymy.
5. Useful evidence includes a fixed query set, expected top-k documents, retrieval scores, known failure cases, latency, cost, and examples where wording changes but intent stays the same.

</details>

## Summary

Embedding turns discrete token IDs into geometry:

```text
identity -> vector -> distance -> retrieval / clustering / model input
```

The deeper idea is not the formula. It is that meaning becomes something you can compare, rank, and pass through a neural network.
