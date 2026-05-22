---
title: "11.4.1 Sequence Labeling Roadmap: One Label per Token"
description: "A concise hands-on roadmap for sequence labeling: understand BIO tags, NER, HMM/CRF history, BiLSTM-CRF, and span-level evaluation."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "sequence labeling guide, NER, BiLSTM-CRF"
---

# 11.4.1 Sequence Labeling Roadmap: One Label per Token

Sequence labeling predicts one label for each token. NER, word segmentation, part-of-speech tagging, and slot filling all use this idea.

## See the Label Path First

![Sequence labeling chapter learning flowchart](/img/course/ch11-sequence-labeling-chapter-flow-en.webp)

![HMM CRF sequence history map](/img/course/ch11-hmm-crf-sequence-history-map-en.webp)

![BiLSTM CRF label path map](/img/course/ch11-bilstm-crf-label-path-map-en.webp)

The key output is not one sentence label, but aligned token-level tags such as `B-PER`, `I-PER`, and `O`.

## Run a BIO Tag Check

```python
tokens = ["Ada", "Lovelace", "wrote", "notes"]
tags = ["B-PER", "I-PER", "O", "O"]

for token, tag in zip(tokens, tags):
    print(token, tag)
```

Expected output:

```text
Ada B-PER
Lovelace I-PER
wrote O
notes O
```

If tokenization changes, labels must stay aligned. Many sequence-labeling bugs are alignment bugs.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | NER and BIO | Create token-level labels and entity spans |
| 2 | HMM/CRF history | Understand sequence constraints and label transitions |
| 3 | BiLSTM-CRF | Connect contextual features with valid label paths |
| 4 | Project practice | Evaluate precision, recall, F1, boundary errors |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
schema: entity types, BIO tags, or sequence-label rules
prediction: token-level labels and extracted spans
metric: entity precision/recall/F1 and boundary cases
failure_check: span boundary, nested entity, unknown word, or inconsistent annotation
Expected_output: gold-vs-predicted span table with at least one miss
```

## Pass Check

You pass this chapter when you can inspect token/tag alignment and explain one boundary error or invalid tag transition.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer starts from the text unit and output type: token, span, sentence label, sequence, embedding, or generated text.
2. The evidence should include a small dataset example, model or pipeline choice, metric, and at least one inspected error case.
3. A good self-check distinguishes preprocessing issues from model issues, such as tokenization mistakes, label ambiguity, data imbalance, or hallucinated generation.

</details>
