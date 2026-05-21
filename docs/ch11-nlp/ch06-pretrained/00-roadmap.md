---
title: "11.6.1 Pretrained Models Roadmap: BERT, GPT, T5"
sidebar_position: 0
description: "A concise hands-on roadmap for pretrained NLP models: understand pretraining, BERT, GPT, T5, transformers pipelines, and transfer to real tasks."
keywords: [pretraining guide, BERT, GPT, T5, transformers]
---

# 11.6.1 Pretrained Models Roadmap: BERT, GPT, T5

Pretrained models move NLP from one-task training to a reusable foundation: pretrain on large text, then transfer to downstream tasks.

## See the Paradigm Map First

![BERT GPT T5 comparison chart](/img/course/bert-gpt-t5-comparison-en.webp)

![Learning order diagram for the pretrained language models chapter](/img/course/ch11-pretrained-chapter-flow-en.webp)

![Pretraining transfer finetune map](/img/course/ch11-pretraining-transfer-finetune-map-en.webp)

BERT emphasizes understanding, GPT emphasizes generation, and T5 rewrites many tasks into text-to-text form.

## Run a Model Family Choice Check

```python
task = {
    "needs_generation": True,
    "needs_sentence_label": False,
    "needs_text_to_text": True,
}

if task["needs_text_to_text"]:
    family = "T5-style text-to-text"
elif task["needs_generation"]:
    family = "GPT-style autoregressive"
else:
    family = "BERT-style understanding"

print("family:", family)
print("reason:", "match model objective to task output")
```

Expected output:

```text
family: T5-style text-to-text
reason: match model objective to task output
```

Do not choose by model name alone. Match tokenizer, objective, output format, cost, and deployment constraints.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Pretraining paradigm | Explain pretrain → transfer → fine-tune/infer |
| 2 | BERT | Understand mask prediction and bidirectional representations |
| 3 | GPT | Understand next-token generation and context window |
| 4 | T5 | Rewrite tasks into text-to-text form |
| 5 | Transformers practice | Connect tokenizer, model, pipeline, input, output |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
model_choice: BERT, GPT, T5, Transformers pipeline, or other pretrained baseline
tokenizer_output: ids, masks, decoded text, or batch shape
task_result: classification, generation, extraction, or text-to-text output
failure_check: wrong model family, token limit, domain mismatch, cost, or latency
Expected_output: model call result plus a short choice rationale
```

## Pass Check

You pass this chapter when you can explain why different objectives create different strengths, and run or design one small pretrained-model comparison experiment.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer starts from the text unit and output type: token, span, sentence label, sequence, embedding, or generated text.
2. The evidence should include a small dataset example, model or pipeline choice, metric, and at least one inspected error case.
3. A good self-check distinguishes preprocessing issues from model issues, such as tokenization mistakes, label ambiguity, data imbalance, or hallucinated generation.

</details>
