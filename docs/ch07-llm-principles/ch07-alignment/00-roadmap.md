---
title: "7.1 Pre-class Guide: What Is This Alignment Chapter Really About?"
sidebar_position: 0
description: "First build a learning map for the LLM alignment chapter: why having capabilities does not mean the model is useful, reliable, or aligned with human intent."
keywords: [alignment guide, RLHF, DPO, safety alignment, human feedback]
---

# Pre-class Guide: What Is This Alignment Chapter Really About?

## Chapter Overview

Pretraining gives a model general language ability, and fine-tuning adapts it to tasks, but that still does not mean the model will answer the way humans expect. This chapter on alignment addresses how to make models more helpful, more honest, more safe, and more aligned with user intent and boundaries.

If pretraining answers “what the model knows,” and fine-tuning answers “what tasks the model is good at,” then alignment answers “how the model should behave.” That is also why, after ChatGPT, alignment shifted from a research topic to a core issue in LLM product experience.

## Where This Chapter Fits in the LLM Roadmap

![LLM alignment chapter relationship diagram](/img/course/ch07-alignment-chapter-flow-en.png)

Alignment is not an isolated technique, but a set of methods that connect model capability, user experience, and safety governance. Later, when you work on RAG, Agent, or tool calling, you will continue to encounter alignment questions: when should the model refuse, when should it ask for confirmation, and when should it avoid fabricating sources or executing actions on its own.

## Main Learning Path for This Chapter

| Section | Key Question | What You Should Be Able to Explain After Learning |
|---|---|---|
| Alignment Problems | Why capable models can still be hard to use | Hallucination, sycophancy, overreach, bias, and unstable outputs |
| RLHF | How to train model behavior using human preferences | The general process of SFT, reward models, and reinforcement learning |
| Alternative Methods | Why methods like DPO and RLAIF emerged | The engineering cost of alignment methods and alternative approaches |
| Safety Evaluation Lab | How to test whether alignment really improved | Fixed test cases, HHH scoring, refusal boundaries, and failure notes |

While learning, do not get stuck in formula details. Instead, focus on how human preferences are collected, how model behaviors are compared, how safety boundaries are injected, and how evaluation determines whether alignment has really improved.

## The Relationship Between Alignment and Application Development

Many application problems cannot be solved by Prompt alone. For example, a customer service bot cannot fabricate policies, a medical assistant cannot diagnose beyond its authority, and an Agent cannot directly delete files or make payments. You can constrain behavior through system prompts, tool permissions, RAG citations, and human confirmation, but whether the underlying model tends to follow instructions, admit when it does not know, and avoid dangerous outputs is still related to alignment.

![Alignment and application safety boundary map](/img/course/ch07-alignment-app-safety-map-en.png)

## What You Will Build in This Chapter

This chapter does not require you to train RLHF yourself. A good practice is to build a “model behavior comparison log”: design 10 questions that are likely to cause problems, such as ambiguous requests, conflicting instructions, missing sources, overreaching tool requests, and safety-boundary requests, then compare the differences in responses from different Prompts or different models. For the basic version, write it as a Markdown table; for the standard version, add scoring dimensions such as helpfulness, honesty, boundary awareness, and citation reliability; for the challenge version, connect it to the later RAG or Agent evaluation set. After that, run a small safety evaluation lab with fixed cases so you can see whether the model is too permissive, too restrictive, or just inconsistent.

## Common Misconceptions

The first misconception is equating alignment with “making the model more obedient.” Real alignment also includes refusing to do things it should not do, admitting when it does not know, avoiding misleading outputs, and protecting users. The second misconception is that alignment only happens during model training; system prompts, tool permissions, human confirmation, and log auditing at the application layer are also part of alignment in a broad sense. The third misconception is looking only at one response instead of doing systematic behavior evaluation.

## Passing Criteria

After finishing this chapter, you should be able to explain the difference between pretraining, fine-tuning, and alignment; describe the basic RLHF process; understand why alternative methods such as DPO emerged; and incorporate “helpfulness, honesty, and safety boundaries” into your own LLM application evaluation checklist.
