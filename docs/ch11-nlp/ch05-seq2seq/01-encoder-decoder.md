---
title: "11.5.2 Seq2Seq Models"
sidebar_position: 1
description: "Start from the idea of “input one piece of text, output another piece of text,” and understand why the encoder-decoder structure became the foundation for translation, summarization, and paraphrasing tasks."
keywords: [seq2seq, encoder decoder, translation, summarization, text generation]
---

# 11.5.2 Seq2Seq Models

![Seq2Seq Encoder-Decoder Bottleneck Diagram](/img/course/ch11-seq2seq-encoder-decoder-bottleneck-map-en.webp)

:::tip Reading Guide
The core challenge of Seq2Seq is “compress the input sequence into a representation, then generate the output sequence step by step.” When reading the diagram, focus on why the context vector creates an information bottleneck and why Attention appears later.
:::

:::tip Section Focus
In the earlier classification and sequence labeling tasks, the output is usually still a “label.”
But starting from this chapter, we move into another kind of problem:

> **The input is a piece of text, and the output is also a piece of text.**

For example:

- Translation
- Summarization
- Paraphrasing
- Question-answer generation

The most classic starting point for these tasks is the encoder-decoder structure.
:::

## Learning Objectives

- Understand the fundamental difference between Seq2Seq and classification tasks
- Understand what the encoder and decoder each do
- Build an intuition for “encode first, then generate” through a runnable example
- Understand why Seq2Seq became the basic structure for many generation tasks

---

## First, Build a Map

For beginners, the best way to understand this section is not to “start with model details,” but to first see how the task shape changes:

```mermaid
flowchart LR
    A["Input a piece of text"] --> B["The encoder reads and understands it first"]
    B --> C["Internal representation"]
    C --> D["The decoder generates the output step by step"]
```

So what this section really wants to solve is:

- Why are “text-to-text” tasks fundamentally different from classification tasks?
- Why do we split the system into encoder and decoder?

## What Problem Does Seq2Seq Solve?

### It is not “assigning a label to the whole sentence”

It is more like:

- Input a sequence of tokens
- Output another sequence of tokens

For example:

- “I love studying” -> “I enjoy learning”

### Why are ordinary classifiers not suitable for this kind of task?

Because classifier outputs are usually one label from a fixed set.
But the output of a Seq2Seq task:

- Has variable length
- Has variable content
- Has sequential dependencies during generation

### An Analogy

Classification is like giving an essay a score.
Seq2Seq is more like rewriting a Chinese essay into an English one.

---

## What Do the Encoder and Decoder Do?

### Encoder

It is responsible for:

- Reading the input sequence
- Compressing the input into an internal representation

### Decoder

It is responsible for:

- Using the encoded result
- Generating the output sequence step by step

### Why split it into two parts?

Because these tasks naturally work in this order:

- First understand the input
- Then construct the output

This is different from plain classification.

---

## Run a Minimal “Encode Then Generate” Example

```python
translation_memory = {
    "I": "I",
    "love": "love",
    "study": "study",
}


def encode(source_tokens):
    return {"source_tokens": source_tokens, "length": len(source_tokens)}


def decode(encoded):
    output = []
    for token in encoded["source_tokens"]:
        output.append(translation_memory.get(token, "<unk>"))
    return output


source = ["I", "love", "study"]
encoded = encode(source)
target = decode(encoded)

print("encoded:", encoded)
print("decoded:", target)
```

Expected output:

```text
encoded: {'source_tokens': ['I', 'love', 'study'], 'length': 3}
decoded: ['I', 'love', 'study']
```

This is only a toy dictionary decoder, but it makes the Seq2Seq shape visible: encode the whole input first, then let the decoder produce an output sequence.

### What is the most important insight from this example?

It shows that the core flow of Seq2Seq is:

1. The input does not directly become the final answer
2. There is first an intermediate encoded representation
3. Then the decoder generates the output

### What should beginners remember first when learning Seq2Seq?

The most important things to remember first are:

1. The encoder is more like “reading and understanding the input first”
2. The decoder is more like “writing the output step by step based on the understanding”
3. The output is not a fixed label, but a sequential generation process

---

## What Is the Most Common Difficulty in Seq2Seq?

### The input is compressed too crudely

A classic problem in early encoder-decoder models is:

- The entire input is compressed into a single fixed-length vector

When the input is long, information is easily lost.

### The output is generated step by step

This means:

- If the previous step is wrong
- The later steps are also likely to go wrong

### This is also why Attention was introduced later

One of the core goals of Attention is to let the decoder, when generating, not rely only on one fixed vector,
but dynamically look at different positions in the input.

### How is this section related to the Attention mechanism later?

What this section should first establish is:

- The main “encode -> decode” flow of Seq2Seq

And the next section on Attention is meant to solve the core bottleneck here:

- Fixed-length representations lose information too easily

---

## What Tasks Is Seq2Seq Suitable For?

### Translation

A classic input-output mapping task.

### Summarization

Input a long article, output a short one.

### Paraphrasing and Question-Answer Generation

The input and output are not the same text, but there is a clear correspondence between them.

---

## The Easiest Pitfalls to Fall Into

### Misconception 1: Seq2Seq is just a “translation model”

Translation is only the most classic example.
In essence, it applies to a broader set of “text-to-text” tasks.

### Misconception 2: Having an encoder and decoder is already enough

Without Attention and stronger representations, long-input problems become very obvious.

### Misconception 3: For generation tasks, being able to produce output is all that matters

What is really hard in Seq2Seq tasks is:

- Staying faithful to the input
- Generating reasonable content
- Preserving structure

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
source_target: source text, target text, and task type
decoded_output: generated summary, translation, transcript, or sequence result
alignment_note: attention, CTC path, coverage, or copied source evidence
failure_check: omission, repetition, hallucination, wrong alignment, or weak evaluation
Expected_output: generated text with factual or alignment review notes
```

## Summary

The most important thing in this section is to understand Seq2Seq as:

> **A structure that first encodes the input and then generates the output step by step. It is the basic paradigm behind translation, summarization, and many text generation tasks.**

As long as this main structure is clear, learning Attention and T5 later will feel very natural.

---

## What You Should Take Away

- Seq2Seq is the basic structure for “input sequence -> output sequence”
- The encoder / decoder design exists to solve “understand first, then generate”
- The emergence of Attention was precisely to address the core information bottleneck in Seq2Seq

---

## Exercises

1. Expand the dictionary in the example to 5–10 words, then try a few more sentences.
2. Why is the output of Seq2Seq not fixed-length, and not from a fixed label set?
3. Think about this: if the input is very long, why is “compressing it into only one vector” difficult?
4. Explain in your own words: what does the encoder do, and what does the decoder do?

<details>
<summary>Solution approach and explanation</summary>

1. Expanding the dictionary should make the toy generator cover more input words, but it will still fail on unseen words and word order.
2. Seq2Seq output is variable-length because the decoder emits tokens step by step until a stop condition, not from a fixed label set.
3. Compressing a long input into one vector is difficult because details, order, and long-range dependencies compete for limited representation space.
4. The encoder reads the input into a representation; the decoder uses that representation and previous outputs to generate the target sequence.

</details>
