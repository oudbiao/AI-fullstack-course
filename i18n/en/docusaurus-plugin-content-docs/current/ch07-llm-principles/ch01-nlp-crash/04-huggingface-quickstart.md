---
title: "1.5 HuggingFace Quick Start"
sidebar_position: 4
description: "From tokenizer, config, model, batch, to forward outputs, understand HuggingFace’s most common workflow, and get a runnable beginner example that does not rely on network downloads."
keywords: [HuggingFace, transformers, tokenizer, model, config, forward, batch]
---

# HuggingFace Quick Start

:::tip Section Overview
When many beginners first encounter HuggingFace, they get confused by these names:

- `AutoTokenizer`
- `AutoModel`
- `pipeline`
- `config`
- `forward`

They look like a lot of APIs,  
but if you extract the core workflow, it is actually very stable:

> **text -> tokenizer -> input ids / mask -> model.forward -> hidden states / logits**

The goal of this lesson is to make this chain clear.
:::

## Learning Goals

- Understand HuggingFace’s most common input-to-output workflow
- Distinguish what tokenizer, config, model, and batch are responsible for
- Read a minimal `transformers` example that does not depend on online downloads
- Build an initial sense of familiarity for reading official examples and repository code later

---

## 1. What Exactly Does HuggingFace Help Us Do?

### 1.1 It is not a “model,” but an entire ecosystem

Many people misunderstand HuggingFace as:

- a very powerful model platform

More accurately, it is a whole ecosystem of tools for model development, commonly including:

- `transformers`
- `datasets`
- `tokenizers`
- `peft`

What you will encounter most often are:

- tokenizer
- model
- config

### 1.2 The most common workflow only has a few steps

Whether you are doing classification, generation, or feature extraction,  
the most important call path usually looks like this:

1. Use the tokenizer to convert text into `input_ids`
2. Prepare `attention_mask`
3. Feed the batch into the model
4. Get hidden states, logits, or generated results from the output

![HuggingFace standard workflow object relationship diagram](/img/course/ch07-huggingface-workflow-object-map.png)

:::tip Reading the diagram
When reading this diagram, think of HuggingFace as a standard lab bench: `tokenizer` turns text into tensors, `config` describes the model structure, `model.forward` performs the computation, and the output is then turned into hidden states, logits, or generation results. There are many API names, but the main line is actually very stable.
:::

Once this chain becomes clear in your mind,  
many examples will no longer feel messy.

### 1.3 An analogy: like assembling a standardized lab setup

You can think of HuggingFace as standardized lab components:

- tokenizer is like a sample preprocessor
- config is like the model blueprint
- model is the machine that actually performs computation
- batch is a tray of samples sent in at once

Its value lies in:

- unified interfaces
- reduced repetitive work
- helping you try models and tasks faster

---

## 2. First, Separate the Most Common Objects

### 2.1 Tokenizer: turns text into model input

It usually handles:

- tokenization
- token -> id
- padding
- truncation

The most common fields in its output are:

- `input_ids`
- `attention_mask`

### 2.2 Config: the model structure blueprint

config mainly describes:

- hidden size
- number of layers
- number of heads
- vocabulary size

You can think of it as the instruction manual for “what the model looks like.”

### 2.3 Model: the part that actually performs forward

The model builds the neural network according to config,  
then accepts tensor inputs and outputs:

- `last_hidden_state`
- `pooler_output`
- `logits`

Outputs differ a bit across tasks,  
but the core idea is the same.

### 2.4 Batch: why padding is always needed

Because a batch contains texts of different lengths.  
Models usually require input tensors to have a unified shape,  
so you need to:

- pad shorter sentences
- use a mask to tell the model which positions are real tokens

---

## 3. First Look at a Zero-Download, Runnable `transformers` Example

This code has a few very important characteristics:

- It does not rely on downloading a model from the internet
- It directly uses a locally defined `BertConfig` to randomly initialize a small model
- It prepares a very small vocabulary by itself
- It encodes two sentences as a batch and feeds them into the model

In other words, it lets you run through the main HuggingFace workflow completely.

:::info Run tip
```bash
pip install torch transformers
```
:::

```python
import torch
from transformers import BertConfig, BertModel

vocab = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "reset": 4,
    "password": 5,
    "refund": 6,
    "order": 7,
    "please": 8,
    "help": 9,
}


def tokenize(text):
    return text.lower().split()


def encode(text, max_length=6):
    tokens = ["[CLS]"] + tokenize(text) + ["[SEP]"]
    input_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens][:max_length]
    attention_mask = [1] * len(input_ids)

    if len(input_ids) < max_length:
        pad_count = max_length - len(input_ids)
        input_ids += [vocab["[PAD]"]] * pad_count
        attention_mask += [0] * pad_count

    return input_ids, attention_mask


texts = [
    "please help reset password",
    "refund order",
]

encoded = [encode(text) for text in texts]
input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.long)

config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
)

model = BertModel(config)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("input_ids shape        :", tuple(input_ids.shape))
print("attention_mask shape   :", tuple(attention_mask.shape))
print("last_hidden_state shape:", tuple(outputs.last_hidden_state.shape))
print("pooler_output shape    :", tuple(outputs.pooler_output.shape))
```

### 3.1 In what order should you read this code?

The best order is:

1. Look at `encode` first to understand how text becomes `input_ids`
2. Then look at `BertConfig` to see how the model structure is defined
3. Finally, look at the output shapes from `model(...)`

This way, you will quickly connect:

- text format
- model structure
- forward outputs

These three things together.

### 3.2 Why not use `from_pretrained` here?

Because `from_pretrained` often requires downloading weights from the internet.  
To make sure this example can run offline, we intentionally use:

- `BertConfig(...)`
- `BertModel(config)`

That means:

- the model is randomly initialized

It cannot be used for real task prediction,  
but it is very suitable for understanding HuggingFace’s basic calling workflow.

### 3.3 What is the easiest detail to overlook here?

The easiest thing to overlook is:

- a batch does not exist naturally
- you first encode each text, then stack them into tensors

If you understand this layer clearly,  
it will be much easier to read wrappers such as `DataCollator` and `Trainer` later.

---

## 4. What Does a Real-World `from_pretrained` Usually Look Like?

If you have an internet connection,  
a more common pattern would be:

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

batch = tokenizer(
    ["please help reset password", "refund order"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

outputs = model(**batch)
print(outputs.last_hidden_state.shape)
```

This code does essentially the same thing as the offline version above.  
The difference is just that:

- the tokenizer and model weights are provided directly by the Hub

So you can think of the offline example above as:

- opening the black box and looking inside

And think of this `from_pretrained` version as:

- getting started quickly with the official wrapper

---

## 5. Why Is HuggingFace So Good for Beginners and Experiments?

### 5.1 Because the interfaces are unified

Even though many models differ internally,  
they usually follow a similar interface in HuggingFace:

- tokenizer handles text input
- model handles forward

This makes switching models much easier.

### 5.2 Because the ecosystem is rich

Later, you will also encounter:

- `AutoModelForSequenceClassification`
- `AutoModelForCausalLM`
- `Trainer`
- `DataCollator`

They are all built on top of this most basic chain.

### 5.3 Because it fits the pattern of “experiment first, then go deeper”

Often, you do not implement everything from scratch first,  
but instead run a standard interface to get things working,  
and then gradually understand:

- tokenizer
- attention mask
- logits
- generation config

This is also why HuggingFace is such a valuable learning entry point.

---

## 6. Common Pitfalls

### 6.1 Mistake 1: Thinking that making `from_pretrained` work means you truly understand the model

Getting it to run is only the beginning.  
Real understanding also requires knowing:

- what the input tensors look like
- what each output field means
- whether the tokenizer and model match

### 6.2 Mistake 2: Ignoring `attention_mask`

If there is padding but no mask,  
the model may treat padded positions as real content.

### 6.3 Mistake 3: Mixing up randomly initialized models with pretrained models

The offline example in this lesson is only for understanding the interface.  
Models with real task ability are usually:

- models loaded with pretrained weights

---

## Summary

The most important thing in this lesson is not remembering how many HuggingFace class names there are,  
but truly connecting the most basic workflow:

> **Text first goes through the tokenizer to become `input_ids` and `attention_mask`, then config defines the structure, model performs the forward pass, and finally the output is hidden states or task results.**

Once this chain is clear,  
you will no longer be intimidated by official examples, third-party repositories, or training scripts.

---

## Exercises

1. Change `max_length` in the example to a smaller value and observe how padding and truncation change.
2. Why does this lesson use `BertConfig + BertModel` instead of `from_pretrained` directly?
3. Explain in your own words what tokenizer, config, and model are responsible for.
4. If you see a batch with `input_ids` but no `attention_mask`, what problem would you suspect first?
