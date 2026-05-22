---
title: "7.3.1 Transformer Deep Dive Roadmap: Blocks, Masks, Cost"
description: "A compact Transformer deep dive roadmap: architecture review, decoder blocks, variants, efficient attention, and scale cost."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Transformer deep dive, decoder block, efficient attention, KV cache, model variants"
---
This chapter looks inside the Transformer enough to debug LLM behavior and understand why context length, attention, KV cache, and model variants matter.

## Look at the Internal Flow First

![Transformer deep-dive chapter relationship diagram](/img/course/ch07-transformer-deep-chapter-flow-en.webp)

![Transformer information flow, computation cost, and task fit diagram](/img/course/ch07-transformer-cost-task-map-en.webp)

## Build a Causal Mask

```python
seq_len = 4
mask = []
for query_pos in range(seq_len):
    row = []
    for key_pos in range(seq_len):
        row.append("allow" if key_pos <= query_pos else "block")
    mask.append(row)

for row in mask:
    print(row)
```

Expected output:

```text
['allow', 'block', 'block', 'block']
['allow', 'allow', 'block', 'block']
['allow', 'allow', 'allow', 'block']
['allow', 'allow', 'allow', 'allow']
```

![Causal mask run result map](/img/course/ch07-causal-mask-result-map-en.webp)

Generation uses this "no future peeking" rule: a token can attend to earlier tokens, but not future tokens.

## Learn in This Order

| Order | Read | What to focus on |
|---|---|---|
| 1 | [7.3.2 Architecture Review](./01-architecture-review.md) | attention, residual, normalization |
| 2 | [7.3.3 Modern Decoder Block](./02-modern-decoder-block.md) | decoder-only LLM block |
| 3 | [7.3.4 Model Variants](./02-model-variants.md) | encoder, decoder, encoder-decoder |
| 4 | [7.3.5 Efficient Attention](./03-efficient-attention.md) | KV cache, MQA/GQA, long context |
| 5 | [7.3.6 Scale and Computation](./04-scale-computation.md) | cost, latency, memory |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
block_contract: [batch, seq, d_model] in and out
mask_check: causal mask blocks future positions
kv_cache_reason: inference reuses past keys and values
compute_note: attention cost grows with sequence length
bridge: these details explain latency and context limits in apps
```

## Pass Check

You pass this roadmap when you can explain why decoder-only models need a causal mask, why attention gets expensive as context grows, and why KV cache helps generation.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer explains how tokens, context, attention, prompts, and generation behavior connect in one request-response path.
2. The evidence should include at least one reproducible prompt or structured-output test, plus notes on why the output passed or failed.
3. A good self-check separates prompt design, RAG, fine-tuning, and alignment: use the lightest method that fixes the observed problem.

</details>
