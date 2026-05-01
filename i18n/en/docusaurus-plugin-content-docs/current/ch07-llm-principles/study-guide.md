---
title: "Study Guide: How to Learn LLM Principles, Prompt, and Fine-Tuning Without Getting Confused"
sidebar_position: 1
description: "A learning guide for AI full-stack beginners: LLM history, Transformer, pretraining, Prompt, fine-tuning, alignment, project roadmap, and evaluation criteria."
keywords: [LLM study guide, how to learn Transformer, how to learn Prompt, how to learn fine-tuning, how to learn RLHF]
---

# Study Guide: How to Learn LLM Principles, Prompt, and Fine-Tuning Without Getting Confused

If you arrive at `06 LLM Principles, Prompt, and Fine-Tuning` and feel overwhelmed by the terminology, don’t rush to model leaderboards. First, you need to understand where LLM capabilities come from and how those capabilities are controlled and adapted.

## Core principle for this stage

On the first pass, focus on one evolution path: text is split into tokens, tokens become embeddings, Transformer models context, pretraining gives general capabilities, and Prompt, fine-tuning, and alignment make those capabilities more usable.

![LLM study guide evolution path](/img/course/ch07-study-guide-evolution-line.png)

## Recommended learning order

In the first round, quickly fill in the core NLP basics. At minimum, you should understand tokenizer, embedding, language models, and basic HuggingFace usage.

In the second round, study the LLM overview and development history. The key is not to memorize model names, but to understand how scale, data, architecture, and alignment work together to drive capability changes.

In the third round, study Transformer and pretraining. You need to know why Attention, context windows, training data, and compute matter.

In the fourth round, study Prompt. Prompt is the lightest-weight way to control behavior, and it is also the foundation for structured output, RAG, and Agent workflows.

In the fifth round, study fine-tuning and alignment. Focus on when fine-tuning is needed, what problems LoRA/QLoRA solve, and why RLHF is related to model behavior.

## Suggested learning pace

| Content type | Suggested time | Learning goal |
|---|---|---|
| NLP crash course | 4–8 hours | Understand token, embedding, and language models |
| LLM overview | 3–5 hours | Build an understanding of development history and capability boundaries |
| Transformer / pretraining | 8–16 hours | Understand where capabilities come from |
| Prompt | 4–8 hours | Be able to design structured task prompts |
| Fine-tuning / alignment | 8–16 hours | Be able to judge whether fine-tuning is needed and what risks exist |

## Stage project roadmap

For the first project, it is recommended to do a Prompt comparison experiment. Choose one task and compare a basic prompt, role prompt, step-by-step prompt, few-shot prompt, and structured output.

For the second project, it is recommended to do a structured output task, such as converting natural language into JSON, tables, or function arguments.

For the third project, you can design a domain fine-tuning plan. You do not have to train a large model right away, but you should explain the data format, annotation method, training approach, evaluation metrics, and risks.

## Common sticking points

The most common sticking point is mixing up Prompt, RAG, and fine-tuning. Prompt changes how inputs are organized, RAG adds external knowledge, and fine-tuning changes the model’s behavioral tendencies. They solve different problems.

The second sticking point is treating an LLM like a database. LLMs can hallucinate, knowledge can become outdated, and an answer is not the same as a fact.

The third sticking point is fine-tuning too early. Many problems should first be solved with Prompt, RAG, tool calling, or product logic.

## Passing criteria

After completing this stage, you should be able to explain the relationship between Token, Embedding, Attention, pretraining, Prompt, fine-tuning, and alignment.

If you can judge whether a requirement should use Prompt, RAG, or fine-tuning, and you can design a structured output Prompt, then you are ready to move on to the LLM application and RAG stage.
