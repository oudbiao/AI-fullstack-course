---
title: "6.1 Pre-reading Guide: What Exactly Is This Chapter About Pretrained Models?"
sidebar_position: 0
description: "First build a learning map for the pretrained models chapter: how the pretraining paradigm, BERT, GPT, T5, and transformers practice connect into the main line of modern NLP."
keywords: [pretraining guide, BERT, GPT, T5, transformers]
---

# Pre-reading Guide: What Exactly Is This Chapter About Pretrained Models?

![BERT GPT T5 comparison chart](/img/course/bert-gpt-t5-comparison.png)

## Chapter Positioning

This chapter is the bridge from traditional NLP to large-model applications. In the previous chapters, you learned text representation, text classification, sequence labeling, and Seq2Seq. These topics usually train models around a single task. Starting with this chapter, we enter the modern NLP paradigm: “pretrain a general-purpose foundation first, then transfer it to different tasks.”

The goal of studying this chapter is not to memorize the names BERT, GPT, and T5, but to understand three questions: how models gain language ability from large-scale text, why different pretraining objectives lead to different capability tendencies, and how the transformers library connects these models to real projects.

## Where This Chapter Fits in the Whole Course

![Learning order diagram for the pretrained language models chapter](/img/course/ch11-pretrained-chapter-flow.png)

This chapter is both the conclusion of natural language processing at Station 11 and the bridge leading into the large-model, RAG, and Agent sections at Stations 7–9. Later, when you call large-model APIs, do text retrieval, perform fine-tuning, or build RAG systems, you will repeatedly encounter concepts such as tokenizer, embedding, context length, generation methods, and model loading.

## What Do BERT, GPT, and T5 Each Represent?

| Model family | Core idea | What it helps you understand better |
|---|---|---|
| BERT | Learn bidirectional semantic representations through masked prediction | Basic intuition for classification, matching, extraction, and retrieval embeddings |
| GPT | Learn generation ability through autoregressive next-token prediction | Chatting, continuation, tool use, Prompting, and generative applications |
| T5 | Unify various NLP tasks as text-to-text | Task unification, instruction-style learning, and multitask transfer |

Do not think of these three model types as different versions replacing one another. They are more like three training and usage paradigms: BERT focuses on understanding, GPT focuses on generation, and T5 emphasizes rewriting every task into text-to-text form. Modern large models absorb these ideas, and learning them will help you understand why later techniques are designed the way they are.

## Learning Sequence for This Chapter

First, understand the pretraining paradigm: why a model that first learns from large-scale text and then transfers to specific tasks is more effective than training from scratch for each task. Second, look at BERT, with a focus on mask, bidirectional context, CLS representations, and fine-tuning for downstream tasks. Third, look at GPT, with a focus on autoregressive generation, context windows, and Prompting. Fourth, look at T5 and understand the idea of unifying translation, summarization, question answering, and classification as text-to-text. Finally, run a minimal example with transformers to connect tokenizer, model, pipeline, input, and output.

## Connection to the Large-Model Course

This chapter will help you understand many “engineering common sense” ideas in large-model applications ahead of time. For example, tokenizer determines how text is read by the model; context length determines how much material can be included at once; embedding can be used for retrieval and clustering; generation models need control over temperature, length, and stopping conditions; and model loading and inference costs can affect deployment choices.

When you move into RAG, your understanding of BERT and embeddings will help you understand vector retrieval; when you move into Prompt and Agent, GPT’s autoregressive generation will help you understand why the model generates plans and tool parameters step by step; when you move into fine-tuning, the relationship between pretraining and downstream task transfer becomes very important.

## Chapter Mini-Project Outcome

It is recommended that you complete a “pretrained model comparison mini-experiment.” The basic version can use transformers pipelines to run text classification, summarization, or generation examples separately and record the inputs and outputs. The standard version can compare the applicability boundaries of BERT-like models and GPT-like models on the same text task. The challenge version can turn a course paragraph into a small embedding retrieval example as preparation for later RAG work.

At minimum, the README should clearly explain which model was used, what the input was, what the output was, whether the model is better for understanding or generation, what dependency or download problems were encountered during runtime, and how this experiment connects to later RAG or LLM applications.

## Common Misunderstandings

Do not assume that “pretrained models” simply means “larger models.” What really matters is the training objective, the input-output format, and the transfer method. Also, do not rush to the newest model names at the beginning. First make sure you understand the basic paradigms of BERT, GPT, and T5, so that when you later see models like RoBERTa, DeBERTa, LLaMA, Qwen, and DeepSeek, you will know roughly what kinds of problems they are trying to solve.

## Passing Criteria

After finishing this chapter, you should be able to clearly explain the differences between the training ideas behind BERT, GPT, and T5, why the transformers library can unify access to many models, what tasks are suitable for understanding-oriented models, what tasks are suitable for generation-oriented models, and how these concepts connect to later RAG, Prompt, fine-tuning, and Agent work.
