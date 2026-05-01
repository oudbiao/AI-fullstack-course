---
title: "6.6 Practical Transformers Library Usage"
sidebar_position: 20
description: "From tokenizer, config, and model to the smallest pipeline, truly learn how to use the core HuggingFace Transformers interfaces offline."
keywords: [transformers, HuggingFace, tokenizer, AutoModel, pipeline, config]
---

# Practical Transformers Library Usage

![Transformers library call chain diagram](/img/course/ch11-transformers-library-call-chain-map-en.png)

:::tip Reading the diagram
When you first use the Transformers library, it is easy to get confused by all the API names. First, look at the call chain in this order: Tokenizer, Config, Model, Task Head, Pipeline. Understand what each object is responsible for, and then look up the specific class names. That will make things much clearer.
:::

:::tip Where this section fits
When learning pretrained models, it is easy to stay at the conceptual level and end up “able to talk about it but not use it.”  
This section is meant to solve a very practical question:

> **When facing the `transformers` library, where should I even start?**

We will try to walk through the most important interfaces in a way that does not depend on downloading anything from the internet.
:::

## Learning objectives

- Understand what the most common objects in the `transformers` library do
- Learn to distinguish the roles of tokenizer, config, model, and pipeline
- Run a minimal offline example with tokenizer + model
- Understand how real projects move from “it runs” to “it is maintainable”

---

## 1. First, let’s separate the main characters in the library

### 1.1 `Tokenizer`

Responsible for turning text into a sequence of numbers that the model can process.

### 1.2 `Config`

Responsible for describing model architecture parameters, such as:

- hidden size
- number of layers
- number of heads

### 1.3 `Model`

Responsible for the actual forward computation.

### 1.4 `Pipeline`

A higher-level wrapper that helps you combine:

- tokenization
- forward pass
- post-processing

into a more convenient interface.

Remember this in one sentence:

> tokenizer handles entry, model handles computation, and pipeline stitches the whole thing together. 

---

## 2. Why do many people get confused the first time they use `transformers`?

Because it has two layers:

### 2.1 Concept layer

You know things like:

- BERT is encoder-only
- GPT is decoder-only

### 2.2 Tool layer

Then you run into:

- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `pipeline`
- `from_pretrained`

Beginners often get stuck here:

> There are too many names, but I do not know what each interface actually solves. 

So the key goal of this section is not memorizing APIs, but building a map of the call sequence.

---

## 3. First, build a minimal tokenizer offline

### 3.1 Why not just download an existing model?

Because the tutorial should try to ensure that you can still run it even without internet access.  
So here we manually prepare a tiny `vocab.txt` so that you can truly understand what a tokenizer does.

### 3.2 Runnable example

```python
from pathlib import Path
from transformers import BertTokenizer

vocab_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "I", "love", "natural", "language", "processing", "Beijing"
]

vocab_path = Path("mini_vocab.txt")
vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

tokenizer = BertTokenizer(vocab_file=str(vocab_path))

encoded = tokenizer("Ilovenatural language processing", return_tensors="pt")

print(encoded)
print("tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
```

### 3.3 What is this code teaching you?

It is teaching you that:

- a tokenizer is not a magical black box
- it is essentially “vocabulary + splitting rules + encoding rules”
- the most important outputs are `input_ids` and `attention_mask`

---

## 4. Next, build a minimal BERT model offline

### 4.1 Why use a randomly initialized model?

Because our goal right now is not to chase performance, but to:

> truly understand how a model object in the `transformers` library receives input and produces output. 

### 4.2 Runnable example

```python
import torch
from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=15,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64
)

model = BertModel(config)

input_ids = torch.tensor([[2, 5, 6, 7, 8, 9, 10, 11, 12, 3]])
attention_mask = torch.ones_like(input_ids)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("last_hidden_state shape:", outputs.last_hidden_state.shape)
print("pooler_output shape    :", outputs.pooler_output.shape)
```

### 4.3 What should you really understand here?

- `input_ids` are token IDs
- `attention_mask` tells the model which positions are valid
- `last_hidden_state` is the contextualized representation of each position
- `pooler_output` is one kind of sentence-level representation

Once you understand these, it becomes much easier to add a classification head, matching head, or generation head later.

---

## 5. Connect tokenizer and model together

### 5.1 Runnable example

```python
from pathlib import Path
import torch
from transformers import BertTokenizer, BertConfig, BertModel

vocab_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "I", "love", "natural", "language", "processing"
]

vocab_path = Path("mini_vocab_2.txt")
vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

tokenizer = BertTokenizer(vocab_file=str(vocab_path))

config = BertConfig(
    vocab_size=len(vocab_tokens),
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64
)
model = BertModel(config)

batch = tokenizer(["Ilovenatural language processing", "I love language"], padding=True, return_tensors="pt")
outputs = model(**batch)

print("input_ids shape        :", batch["input_ids"].shape)
print("attention_mask shape   :", batch["attention_mask"].shape)
print("last_hidden_state shape:", outputs.last_hidden_state.shape)
```

### 5.2 This is the most basic real call chain

In real projects, the most common low-level flow is:

1. text -> tokenizer
2. tokenizer -> tensor
3. tensor -> model
4. model output -> post-processing or task head

You have already run through this chain.

---

## 6. What are the `Auto*` interfaces for?

### 6.1 Why does the library have so many `AutoModel` variants?

Because `transformers` wants to save you from writing lots of model-type conditionals by hand.

For example:

- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `AutoModelForCausalLM`

Their design goal is:

> Give a model name or config, and automatically pick the right class. 

### 6.2 An offline `AutoModel.from_config` example

```python
from transformers import AutoModel, BertConfig

config = BertConfig(
    vocab_size=20,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    intermediate_size=32
)

model = AutoModel.from_config(config)
print(type(model))
```

This shows that:

- `AutoModel` does not necessarily need to download anything online
- it is essentially “automatically instantiate the correct model based on the config”

---

## 7. Is `pipeline` worth learning?

### 7.1 Yes, but you need to know when it is appropriate

Advantages of `pipeline`:

- quick to get started
- fast for demos
- less boilerplate code

It is better suited for:

- learning
- quick validation
- small experiments

### 7.2 But in engineering, you cannot rely on it alone

Because real projects often still need control over:

- batch
- device
- output format
- logging
- error handling

So the mature approach is usually:

- learn `pipeline`
- but also understand the underlying tokenizer + model call chain

---

## 8. The most common task heads in the Transformers library

Here are some high-frequency interfaces to remember:

| Interface | Suitable for |
|---|---|
| `AutoModel` | Getting basic representations only |
| `AutoModelForSequenceClassification` | Text classification |
| `AutoModelForTokenClassification` | Sequence labeling |
| `AutoModelForQuestionAnswering` | Extractive QA |
| `AutoModelForCausalLM` | Generation tasks |

The logic behind this is actually simple:

> the same backbone model + different task heads. 

---

## 9. Common pitfalls for beginners

### 9.1 Only knowing `pipeline`, not the underlying calls

This makes it easy to get stuck in engineering scenarios.

### 9.2 Not understanding tokenizer output fields

At minimum, you should understand:

- `input_ids`
- `attention_mask`

### 9.3 Mixing up “model concepts” and “library interfaces”

You need to clearly distinguish:

- BERT / GPT are model families
- `AutoModel` / `pipeline` are library interfaces

---

## Summary

The most important thing in this section is not memorizing APIs, but understanding:

> **The core call chain of the Transformers library is: tokenizer encodes text into tensors, model performs the forward computation on those tensors, and then a task head or post-processing step produces the final result.**

Once this chain is clear, your thinking will become much more stable when you later do classification, extraction, generation, or fine-tuning.

---

## Exercises

1. Modify the mini vocab in this section, add a few of your own words, and see how the tokenizer output changes.
2. Change `hidden_size` in `BertConfig` to 64 and see how the output shape changes.
3. Explain in your own words: why can’t you just learn `pipeline` when studying the `transformers` library?
4. Think about this: if you want to do text classification, should you start with `AutoModel` or `AutoModelForSequenceClassification`?
