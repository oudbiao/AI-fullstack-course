---
title: "7.1.5 Hugging Face Quick Start"
sidebar_position: 4
description: "Run the core Hugging Face workflow: tokenizer output, config, batch tensors, model.forward, hidden states, logits, and common debugging checks."
keywords: [Hugging Face, transformers, tokenizer, model, config, forward, batch, logits]
---

# 7.1.5 Hugging Face Quick Start

![Hugging Face workflow object map](/img/course/ch07-huggingface-workflow-object-map-en.webp)

:::tip Core Chain
Most Hugging Face examples reduce to one chain:

```text
text -> tokenizer -> input_ids / attention_mask -> model.forward -> hidden states / logits / generated tokens
```

If you understand this chain, `pipeline`, `Trainer`, `DataCollator`, and `AutoModel...` become convenience layers instead of mysterious APIs.
:::

## The Four Objects

| Object | Responsibility | Common fields |
|---|---|---|
| tokenizer | text preprocessing and token-to-ID conversion | `input_ids`, `attention_mask` |
| config | model blueprint | `hidden_size`, `num_hidden_layers`, `vocab_size` |
| model | neural computation | `last_hidden_state`, `logits`, generated IDs |
| batch | stacked tensors with one shape | `[batch, seq_len]` inputs |

The important habit is to inspect shapes. If a shape is wrong, the model usually has not even reached the “AI” part yet.

## Lab 1: Run the Workflow Without Downloading Weights

Install dependencies:

```bash
python -m pip install torch transformers
```

This example uses a tiny random BERT model from `BertConfig`. It has no real language ability, but it lets you inspect the full call path without downloading pretrained weights.

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


def encode(text, max_length=6):
    tokens = ["[CLS]"] + text.lower().split() + ["[SEP]"]
    input_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens][:max_length]
    attention_mask = [1] * len(input_ids)

    while len(input_ids) < max_length:
        input_ids.append(vocab["[PAD]"])
        attention_mask.append(0)

    return input_ids, attention_mask


texts = ["please help reset password", "refund order"]
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

Expected output:

```text
input_ids shape        : (2, 6)
attention_mask shape   : (2, 6)
last_hidden_state shape: (2, 6, 32)
pooler_output shape    : (2, 32)
```

Read the shapes:

- `2` means two texts in the batch.
- `6` means each sequence was padded or truncated to length 6.
- `32` comes from `hidden_size=32`.
- `last_hidden_state` keeps one vector per token.
- `pooler_output` is one vector per sequence in this BERT-style model.

## Lab 2: Use a Real Pretrained Model

When internet access is available, use `from_pretrained`:

```python
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

batch = tokenizer(
    ["please help reset password", "refund order"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

outputs = model(**batch)

print(batch.keys())
print(batch["input_ids"].shape)
print(outputs.last_hidden_state.shape)
```

Expected shape-level output:

```text
dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
torch.Size([2, 6])
torch.Size([2, 6, 768])
```

![Hugging Face shape output result map](/img/course/ch07-huggingface-batch-shape-forward-result-map-en.webp)

Now the model has pretrained weights. The workflow is the same, but the tokenizer, config, and weights come from the Hub and must match each other.

## Object Map for Reading Real Code

![Hugging Face terms map](/img/course/ch07-huggingface-terms-map-en.webp)

When reading a repository, map unfamiliar names back to the core chain:

| Name | How to think about it |
|---|---|
| `pipeline` | high-level demo wrapper around tokenizer + model |
| `AutoTokenizer` | loads the tokenizer class that matches the model repo |
| `AutoModel` | loads the base model without a task head |
| `AutoModelForSequenceClassification` | base model plus classification head |
| `AutoModelForCausalLM` | decoder-style model for next-token generation |
| `DataCollator` | pads and stacks examples into a batch |
| `Trainer` | wraps training loop, evaluation, checkpoints, and logging |
| `logits` | raw scores before softmax or token selection |

## Debugging Checklist

- Tokenizer and model should come from the same model repo.
- Print `batch.keys()` and tensor shapes before calling the model.
- If you pad, you usually need `attention_mask`.
- A random `BertModel(config)` is only for interface learning; it is not pretrained.
- `AutoModel` outputs representations; task-specific classes output task logits.
- If CUDA memory fails, reduce batch size, sequence length, or model size before changing code logic.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
objects: tokenizer, model, config, pipeline or manual forward pass
offline_run: toy workflow output is saved
real_model_optional: model id and task are recorded if downloaded
shape_or_score: one output tensor shape or prediction score
debug_note: model path, device, and tokenizer/model mismatch checked
```

## Exercises

1. Change `max_length` in Lab 1 from `6` to `4`. Which token gets truncated?
2. Change `hidden_size=64`. Which output shape changes?
3. Add a third sentence and confirm the batch dimension changes from `2` to `3`.
4. In Lab 2, replace `AutoModel` with `AutoModelForSequenceClassification`. What new field appears?
5. Explain why `pipeline()` is useful for demos but not enough for debugging batch-shape problems.

<details>
<summary>Reference answers and explanation</summary>

1. The tail token is usually truncated first, so the exact missing token depends on the tokenizer output. This is why you should print tokens instead of guessing from raw text.
2. `hidden_size=64` changes the last dimension of hidden-state tensors. If the model is actually instantiated with that config, parameter shapes also change.
3. Adding a third sentence should change the first dimension of the batch tensors from `2` to `3`. Sequence length and hidden size should remain controlled by tokenization and config.
4. `AutoModelForSequenceClassification` exposes task-specific outputs, especially `logits`, because it adds a classification head on top of the base model.
5. `pipeline()` is great for quick demos, but it hides tokenization, tensor shapes, padding, device placement, and model outputs. Debugging needs those details.

</details>

## Summary

Hugging Face is easiest to learn by following tensors:

```text
tokenizer creates tensors -> model consumes tensors -> outputs expose states or logits
```

Once you can inspect that path, official examples become much less intimidating.
