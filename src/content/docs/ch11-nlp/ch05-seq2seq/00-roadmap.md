---
title: "11.5.1 Seq2Seq Roadmap: Input Sequence to Output Sequence"
description: "A concise hands-on roadmap for Seq2Seq and Attention: understand encoder-decoder, bottlenecks, attention, decoding, and text-to-text tasks."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Seq2Seq guide, attention guide, machine translation"
---

# 11.5.1 Seq2Seq Roadmap: Input Sequence to Output Sequence

Seq2Seq handles tasks where both input and output are sequences: translation, summarization, rewriting, dialogue, and error correction.

## See the Generation Bridge First

![Seq2Seq and Attention chapter learning order diagram](/img/course/ch11-seq2seq-chapter-flow-en.webp)

![Seq2Seq encoder decoder bottleneck map](/img/course/ch11-seq2seq-encoder-decoder-bottleneck-map-en.webp)

![T5 text-to-text task unification map](/img/course/ch11-t5-text-to-text-task-unification-map-en.webp)

The bridge to modern LLMs is clear: generation happens step by step, and attention helps the decoder look back at useful input positions.

## Run an Input-Output Pair Check

```python
source = ["I", "love", "NLP"]
target = ["J'aime", "le", "NLP"]

for step, token in enumerate(target, start=1):
    print(f"decode_step_{step}:", token)
print("source_length:", len(source))
print("target_length:", len(target))
```

Expected output:

```text
decode_step_1: J'aime
decode_step_2: le
decode_step_3: NLP
source_length: 3
target_length: 3
```

Generation projects should record decoding strategy, failure cases, and whether important input information was lost.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Encoder-Decoder | Explain why input and output can have different lengths |
| 2 | Attention | Explain dynamic alignment during generation |
| 3 | Machine translation | Connect teacher forcing, decoding, BLEU/error analysis |
| 4 | CTC and speech | See what changes when input/output are not frame-aligned |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
source_target: source text, target text, and task type
decoded_output: generated summary, translation, transcript, or sequence result
alignment_note: attention, CTC path, coverage, or copied source evidence
failure_check: omission, repetition, hallucination, wrong alignment, or weak evaluation
Expected_output: generated text with factual or alignment review notes
```

## Pass Check

You pass this chapter when you can explain encoder-decoder, attention, greedy/beam decoding, and one generation failure.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer starts from the text unit and output type: token, span, sentence label, sequence, embedding, or generated text.
2. The evidence should include a small dataset example, model or pipeline choice, metric, and at least one inspected error case.
3. A good self-check distinguishes preprocessing issues from model issues, such as tokenization mistakes, label ambiguity, data imbalance, or hallucinated generation.

</details>
