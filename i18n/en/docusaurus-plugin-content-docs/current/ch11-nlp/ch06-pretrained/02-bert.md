---
title: "6.3 BERT Series"
sidebar_position: 17
description: "From bidirectional context and Masked Language Modeling to fine-tuning methods, truly understand what BERT solves in modern NLP."
keywords: [BERT, MLM, bidirectional encoder, pretraining, Transformer Encoder, NLP]
---

# BERT Series

![BERT Masked Language Model](/img/course/bert-masked-language-model-en.png)

:::tip Section focus
BERT is one of the key milestones that brought modern NLP into the “pretraining for everything” era.  
Many concepts in today’s large models have evolved in form, but quite a few of the underlying ideas can be traced back to BERT.
:::

## Learning Objectives

- Understand why BERT became a milestone in NLP
- Clearly explain the core difference between BERT and autoregressive models like GPT
- Master key concepts such as `[CLS]`, `[SEP]`, `[MASK]`, and bidirectional context
- Read a minimal BERT input example
- Understand common fine-tuning methods for BERT

## Background: Which paper did BERT come from?

The most important historical milestone in this section is:

| Year | Paper | Key Authors | What it solved most importantly |
|---|---|---|---|
| 2018 | *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding* | Devlin et al. | Made bidirectional Transformer pretraining + fine-tuning the main path for modern NLP understanding tasks |

For beginners, the most important thing to remember first is:

- BERT is not “just another model name”
- It represents a very important paradigm shift:

> **First do general pretraining on massive text, then fine-tune the same base model for different tasks.**

That is also why, when you learn large models today, many ideas about “pretrain first, then adapt” can be seen very clearly in BERT as an early prototype.

---

## 1. What problem did BERT actually solve?

### 1.1 First, the old problem: word meaning depends on context

Words do not always have a fixed meaning.

For example, the English word `bank`:

- “river bank” means the riverbank
- “bank account” means a financial bank

The same is true in another English example:

- “I ate an apple” — apple means the fruit
- “Apple released a new device” — Apple means the company

If a model can only give each word a fixed vector, it will struggle.

### 1.2 BERT’s key breakthrough

One of BERT’s core contributions is:

> **Making a word’s representation truly depend on context.**

In other words, the same word can get different representations in different sentences.

This is called a “contextual representation.”

### 1.3 A better analogy for beginners

You can think of BERT as:

- a “careful reader” that looks both before and after when reading a sentence

Unlike early static word vectors, which only give a word one fixed business card, it is more like:

- when the same word appears in different sentences, BERT re-understands the role it is playing now

That is why BERT is especially suitable for understanding tasks.

---

## 2. Why is BERT called a “bidirectional” model?

### 2.1 What does bidirectional mean?

Consider the sentence:

> “I took a walk next to the bank yesterday”

When understanding “bank,” people do not only look at the left context “I took a walk next to the,” but also the right context “yesterday.”

BERT’s important feature is:

> The representation of the current token uses both left and right context at the same time.

### 2.2 The core difference from GPT

Roughly speaking:

- **BERT**: more focused on understanding, reads context bidirectionally
- **GPT**: more focused on generation, looks only at left history

So:

- For classification, extraction, and matching tasks, BERT is very strong
- For continuation, dialogue, and generation, the GPT route is more natural

---

## 3. What does BERT input actually look like?

### 3.1 Three very common special tokens

| token | role |
|---|---|
| `[CLS]` | Aggregation position for sentence-level tasks |
| `[SEP]` | Sentence separator |
| `[MASK]` | Position hidden during pretraining |

### 3.2 A minimal input example

```python
tokens = ["[CLS]", "I", "love", "natural", "language", "processing", "[SEP]"]
print(tokens)
print("sequence length:", len(tokens))
```

For sentence-pair tasks, such as question matching:

```python
tokens = [
    "[CLS]", "How", "is", "the", "weather", "today", "[SEP]",
    "Will", "it", "rain", "in", "Beijing", "today", "[SEP]"
]
print(tokens)
```

### 3.3 A beginner-friendly input structure table

| Component | Most important thing to remember |
|---|---|
| `[CLS]` | Aggregation position for sentence-level tasks |
| `[SEP]` | Sentence boundary separator |
| `[MASK]` | Position to be recovered during pretraining |

This table is especially helpful for beginners because it turns BERT input from a “mysterious token string” into a few understandable parts.

---

## 4. What does BERT do during pretraining?

### 4.1 The classic task: Masked Language Modeling

BERT’s most classic training objective is MLM, which means:

> Hide some tokens in a sentence and let the model guess them back from context.

For example:

> “I love [MASK] language processing”

The model must infer what `[MASK]` is from the surrounding context.

### 4.2 A minimal runnable example

```python
tokens = ["[CLS]", "I", "love", "[MASK]", "language", "processing", "[SEP]"]
mask_index = tokens.index("[MASK]")

candidates = ["natural", "machine", "deep"]

print("tokens =", tokens)
print("mask index =", mask_index)
print("candidate fill-ins =", candidates)
```

This example is not actually training a model, but it already teaches you:

- the `[MASK]` position is explicit
- the model’s job is to recover the hidden information
- predictions at the current position depend on bidirectional context

### 4.3 Why is this important?

Because it forces the model to truly understand:

- what is said on the left
- what is said on the right
- what should go in the hidden position

This makes BERT very good at understanding tasks.

### 4.4 The safest default learning order for BERT

A more stable order is usually:

1. First understand what bidirectional context is filling in
2. Then look at the most common tokens: `[CLS] / [SEP] / [MASK]`
3. Then see what MLM asks the model to learn during training
4. Finally look at how fine-tuning attaches a classification head

This is easier than jumping straight into paper details and large-model parameters.

---

## 5. BERT input is not just tokens

### 5.1 Token Embedding

Each token is first turned into a vector.

### 5.2 Position Embedding

The model also needs to know the order, so positional encoding must be added.

### 5.3 Segment Embedding

For sentence-pair tasks, the model also needs to know “which tokens belong to sentence A” and “which tokens belong to sentence B”.

You can think of BERT input as the sum of three parts:

> `final input representation = token embedding + position embedding + segment embedding`

This step is important because Transformer itself does not inherently contain sequence-order awareness.

---

## 6. A truly runnable offline BERT example

The example below does not require downloading pretrained weights. You only need to install `transformers` and `torch`, and you can initialize a small random BERT locally. It is mainly here to help you understand input/output shapes.

:::info Runtime environment
```bash
pip install torch transformers
```
:::

```python
import torch
from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=100,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64
)

model = BertModel(config)

input_ids = torch.tensor([
    [1, 5, 8, 9, 2, 0, 0],   # a shorter sample, padded with 0s at the end
    [1, 7, 6, 3, 4, 2, 0]
])

attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 0]
])

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("last_hidden_state shape:", outputs.last_hidden_state.shape)
print("pooler_output shape    :", outputs.pooler_output.shape)
```

### 6.2 How should we understand the outputs?

- `last_hidden_state`
  - shape: `[batch, seq_len, hidden_size]`
  - each token has a contextual representation

- `pooler_output`
  - shape: `[batch, hidden_size]`
  - usually can be understood as one kind of whole-sentence summary representation

This also explains why BERT is suitable for:

- token-level tasks: use `last_hidden_state`
- sentence-level tasks: use `[CLS]` or sentence-level representations

---

## 7. How do we use BERT for classification?

### 7.1 Typical workflow

The most common approach is:

1. Input a sentence
2. Pass it through BERT
3. Take `[CLS]` or a sentence representation
4. Attach a linear classification head

This is the classic fine-tuning approach.

### 7.2 A small conceptual example

```python
import torch
from torch import nn

# Assume this is the [CLS] representation output by BERT
cls_embedding = torch.randn(4, 32)  # batch=4, hidden=32

# Attach a classification head
classifier = nn.Linear(32, 2)
logits = classifier(cls_embedding)

print("logits shape:", logits.shape)
```

This code is very simple, but it teaches you something very important:

> BERT is often not the end of a task, but a powerful representation layer.

### 7.3 What is most worth showing when BERT is used in a project

What is usually most worth showing is not:

- “I used BERT”

but rather:

1. What the input text looks like
2. How the `[CLS]` representation connects to the classification head
3. What it does better than traditional representations or lighter models
4. Which failure cases it still gets wrong

That way, other people can more easily see:

- you understand BERT’s role in the task pipeline
- you did more than just change the model name

---

## 8. What tasks is BERT good for?

### 8.1 Especially suitable for

- Text classification
- Sentence-pair matching
- Named entity recognition
- Extractive question answering

### 8.2 Where it is less natural

BERT itself was not designed for free-form long-text generation.  
If the task focus is:

- long conversation generation
- continuation
- large-scale text creation

then the GPT route is usually more natural.

---

## 9. Why is BERT no longer the only main character?

### 9.1 The reason is not that BERT is useless, but that the ecosystem kept moving forward

Later NLP and LLM development brought:

- larger-scale pretraining
- stronger generative models
- more unified task interfaces

So today, many applications more often discuss GPT, T5, and Llama-style routes.

### 9.2 But BERT is still very worth learning

Because it helps you truly understand:

- contextual representations
- encoder-only models
- the pretraining + fine-tuning paradigm
- the difference between token-level and sentence-level tasks

These are all important foundations for learning large models later.

---

## 10. Common beginner mistakes

### 10.1 Mixing up BERT and GPT as if they were the same thing

They are both important, but their training objectives and strengths are not the same.

### 10.2 Thinking `[CLS]` is “naturally the best sentence vector”

It is useful in many tasks, but it is not a universal best choice.

### 10.3 Only knowing “use BERT for classification” without knowing what it actually learns

What you really need to master is:

- why it is bidirectional
- why MLM works
- why it is better suited to understanding tasks

---

## Summary

The most important thing in this section is not memorizing BERT’s full name, but grasping these three points:

1. BERT is a representative model of bidirectional context modeling
2. It learns to “understand tokens from context” through MLM
3. It is very suitable for understanding tasks and the fine-tuning paradigm

Once you understand these three points, many differences will become naturally clear when you later learn GPT, T5, and LLMs.

---

## Exercises

1. Create a sentence with `[MASK]` by yourself, and write the candidate word(s) you think are most reasonable.
2. Change `hidden_size` in the offline BERT example to 64, then see how the output shape changes.
3. Think about this: why can a training objective like “I love [MASK] language processing” help the model learn bidirectional understanding?
4. Explain in your own words: what is the core difference between BERT and GPT in the way they “look at context”?
