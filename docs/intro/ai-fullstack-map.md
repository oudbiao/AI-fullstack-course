---
sidebar_position: 2
title: "0.3 AI Full-Stack Capability Map"
description: "A compact visual map of the seven capability layers in AI full-stack learning."
keywords: [AI Full-Stack, capability map, AI learning path, LLM applications, RAG, AI Agent]
---

# 0.3 AI Full-Stack Capability Map

![AI Full-Stack Capability Map](/img/course/intro-ai-fullstack-capability-map-en.webp)

Read the picture first. The course is one path:

```text
tools -> data -> models -> LLM -> RAG -> Agent -> delivery
```

You do not need every detail now. Just remember:

| If you are blocked by... | Go back to... |
|---|---|
| running code | tools and Python |
| messy inputs | data |
| unreliable answers | evaluation and RAG |
| uncontrolled actions | Agent traces and permissions |

## The Seven Layers

| Layer | Course chapters | First visible evidence | Deeper question |
|---|---|---|---|
| Tools | 1 | A reproducible project folder and Git history | Can another person rerun it? |
| Python | 2 | Small scripts with clear inputs and outputs | Is the code readable, typed, and testable? |
| Data | 3 | Clean tables, charts, and notes | Do you know where the data is wrong or biased? |
| Models | 4-6 | Trained or inspected model experiments | What metric would change your decision? |
| LLM | 7 | Prompt, tokens, embeddings, Transformer intuition | Which behavior comes from data, decoding, or context? |
| RAG | 8 | Retrieval trace and answer evaluation | Did the answer use the right evidence? |
| Agent / delivery | 9-12 | Tool traces, permissions, multimodal demos, deployment notes | What can fail when users, files, and actions are real? |

The course is not a pile of topics. It is a debugging stack. When an AI application behaves badly, the cause may live several layers below the feature you are looking at.

## How To Use The Map

Before starting a project, mark the highest-risk layer. For example, a PDF question-answering app usually fails first in data cleaning and retrieval, not in the chat UI. An automation agent usually fails first in tool permissions, state, and evaluation, not in the prompt wording.

During each chapter, keep one artifact that proves the layer works. Screenshots are useful, but logs, README commands, small datasets, metric tables, and failure notes are stronger because they help you debug later.

Optional background: if you want the history behind these layers, skim the [15-stage AI development map](/appendix/ai-milestones).

Next, choose a learning path.
