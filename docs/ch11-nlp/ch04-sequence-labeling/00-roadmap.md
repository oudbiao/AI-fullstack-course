---
title: "11.4.1 Sequence Labeling Roadmap: One Label per Token"
sidebar_position: 0
description: "A concise hands-on roadmap for sequence labeling: understand BIO tags, NER, HMM/CRF history, BiLSTM-CRF, and span-level evaluation."
keywords: [sequence labeling guide, NER, BiLSTM-CRF]
---

# 11.4.1 Sequence Labeling Roadmap: One Label per Token

Sequence labeling predicts one label for each token. NER, word segmentation, part-of-speech tagging, and slot filling all use this idea.

## 11.4.1.1 See the Label Path First

![Sequence labeling chapter learning flowchart](/img/course/ch11-sequence-labeling-chapter-flow-en.png)

![HMM CRF sequence history map](/img/course/ch11-hmm-crf-sequence-history-map-en.png)

![BiLSTM CRF label path map](/img/course/ch11-bilstm-crf-label-path-map-en.png)

The key output is not one sentence label, but aligned token-level tags such as `B-PER`, `I-PER`, and `O`.

## 11.4.1.2 Run a BIO Tag Check

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

## 11.4.1.3 Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | HMM/CRF history | Understand sequence constraints and label transitions |
| 2 | NER and BIO | Create token-level labels and entity spans |
| 3 | BiLSTM-CRF | Connect contextual features with valid label paths |
| 4 | Project practice | Evaluate precision, recall, F1, boundary errors |

## 11.4.1.4 Pass Check

You pass this chapter when you can inspect token/tag alignment and explain one boundary error or invalid tag transition.
