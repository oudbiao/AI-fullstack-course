---
sidebar_position: 0
title: "AI Full-Stack Learning Course — A Comic History of Artificial Intelligence"
description: "The English home page for the AI Full-Stack Learning Course, using 15 science-comic pages to explain the history of AI from Turing and Dartmouth to Transformer, GPT, RAG, and AI Agents."
keywords: [AI full-stack course, AI history comic, artificial intelligence learning, machine learning for beginners, deep learning, Transformer, GPT, RAG, AI Agent, learn AI from scratch]
---

# AI Full-Stack Learning Course: A Comic History of Artificial Intelligence

Welcome to the AI Full-Stack Learning Course. Instead of starting with a long table of contents, this home page begins with a visual story: how artificial intelligence moved from the question “Can machines think?” to today’s RAG, tool calling, and AI Agents.

These 15 comic pages follow the same learning rhythm: **problem -> solution -> new problem**. You do not need to memorize every paper name first. Just follow the story: why each generation of AI appeared, what it solved, what it failed to solve, and why the next breakthrough became necessary.

## How to Read These Comics

| Reading clue | What to pay attention to |
|---|---|
| Historical period | Where this page sits in AI history |
| Key people | Who introduced or popularized the important idea |
| Breakthrough | What became possible for the first time |
| Limitation | Why the method was still not enough |
| Technical metaphor | How the blackboards, machines, arrows, and panels explain the core idea |

---

## Page 1: Turing and the Starting Point of the AI Dream (1936-1950)

![Turing and the starting point of the AI dream comic](/img/course/homepage-ai-history-comic-en-01-turing.png)

This page introduces the original AI question: can a machine behave intelligently? Turing turned “machine thinking” from a philosophical debate into something people could discuss through a testable conversation setup.

---

## Page 2: The Dartmouth Conference and the Birth of AI (1956)

![Dartmouth Conference and the birth of AI comic](/img/course/homepage-ai-history-comic-en-02-dartmouth.png)

This page explains the naming moment of AI as a research field. Researchers began treating “building intelligent machines” as a formal scientific goal, launching an optimistic era of symbolic reasoning and logic-based systems.

---

## Page 3: The Perceptron Boom and the First Neural Network Winter (1957-1969)

![Perceptron boom and first neural network winter comic](/img/course/homepage-ai-history-comic-en-03-perceptron.png)

This page shows the first rise and fall of neural networks. The perceptron could adjust weights from data, but a single-layer model could not solve nonlinear problems such as XOR.

---

## Page 4: Expert Systems: The Glory and Collapse of Hand-Written Rules (1970s-1980s)

![Expert systems era comic](/img/course/homepage-ai-history-comic-en-04-expert-systems.png)

This page covers the golden age of rule-based AI. Expert systems were useful in narrow domains, but rule bases became hard to write, hard to maintain, and hard to transfer to the messy real world.

---

## Page 5: Backpropagation Reignites Neural Networks (1986)

![Backpropagation reignites neural networks comic](/img/course/homepage-ai-history-comic-en-05-backprop.png)

This page explains how multilayer neural networks became trainable again. Backpropagation sends the error signal backward through the network, telling each parameter how it should be adjusted.

---

## Page 6: Handwritten Digit Recognition and the First Practical CNN Success (1989-1998)

![LeNet and the first practical CNN success comic](/img/course/homepage-ai-history-comic-en-06-lenet.png)

This page shows how CNNs proved their value in real industrial settings. Small filters slide across an image, gradually learning edges, strokes, and digit shapes.

---

## Page 7: Statistical Machine Learning: Data Replaces Hand-Written Rules (1990s-2000s)

![Statistical machine learning era comic](/img/course/homepage-ai-history-comic-en-07-statistical-ml.png)

This page explains the shift from “write rules” to “learn patterns from data.” SVMs, decision trees, random forests, and boosting made classification, search, advertising, and tabular prediction more reliable.

---

## Page 8: ImageNet and AlexNet: The Deep Learning Explosion (2009-2012)

![ImageNet and AlexNet deep learning explosion comic](/img/course/homepage-ai-history-comic-en-08-imagenet-alexnet.png)

This page covers the turning point of modern deep learning. Large-scale data, GPUs, and CNNs came together, allowing models to learn useful visual features directly from raw images.

---

## Page 9: ResNet: Why Add X Back? (2015)

![ResNet residual connection comic](/img/course/homepage-ai-history-comic-en-09-resnet.png)

This page explains why very deep networks need shortcuts. ResNet lets information bypass complex layers and adds the original input X back, making much deeper networks trainable.

---

## Page 10: RNN and LSTM: Early Workhorses for Language Sequences (1997-2014)

![RNN and LSTM sequence model comic](/img/course/homepage-ai-history-comic-en-10-rnn-lstm.png)

This page shows how machines used to read sentences one token at a time. RNNs and LSTMs were important for language, speech, and time-series tasks, but sequential computation was slow and long-distance memory remained difficult.

---

## Page 11: Attention: Stop Memorizing Everything, Look at What Matters (2014)

![Attention for machine translation comic](/img/course/homepage-ai-history-comic-en-11-attention.png)

This page introduces the intuition behind attention: when generating the current word, the model can directly look at the most relevant parts of the input sentence instead of compressing everything into one vector.

---

## Page 12: Transformer: Attention Is All You Need (2017)

![Transformer self-attention comic](/img/course/homepage-ai-history-comic-en-12-transformer.png)

This page explains the architecture behind modern large language models. Transformer removes recurrent processing and lets tokens communicate through self-attention, enabling more parallel and scalable training.

---

## Page 13: BERT and GPT: The Beginning of the Pretraining Era (2018-2020)

![BERT and GPT pretraining comic](/img/course/homepage-ai-history-comic-en-13-bert-gpt.png)

This page explains the start of large-scale pretraining. BERT behaves like a reading-comprehension student, while GPT behaves like a writing robot. Together, they helped AI move from task-specific models toward foundation models.

---

## Page 14: SFT, RLHF, and ChatGPT: From Text Completer to Assistant (2022)

![SFT RLHF and ChatGPT comic](/img/course/homepage-ai-history-comic-en-14-rlhf-chatgpt.png)

This page explains how large language models became more useful assistants. SFT teaches models from human-written examples, while RLHF helps them learn which answers humans prefer.

---

## Page 15: RAG, Tool Calling, and Agents: AI Moves Toward Real Tasks (2023-Present)

![RAG tool calling and AI Agent comic](/img/course/homepage-ai-history-comic-en-15-rag-agent.png)

This page explains where AI applications are going now: models do not only answer questions; they retrieve information, call tools, make plans, observe results, and complete tasks within safety boundaries.

---

## Where to Continue After the Comics

If this is your first time learning AI, use the following path:

| Goal | Next step |
|---|---|
| Build AI applications from scratch | Start with [Developer Tools Foundations](ch01-tools), then make sure your environment, terminal, and Git workflow are stable |
| Understand why models can learn | Study [Minimal Math Foundations for AI](ch04-ai-math) and [Machine Learning from Basics to Practice](ch05-machine-learning) |
| Build LLM applications | Focus on [LLM Principles, Prompting, and Fine-Tuning](ch07-llm-principles), [LLM Application Development and RAG](ch08-rag), and [AI Agents and Agentic Systems](ch09-agent) |
| Explore the full AI timeline | Continue with the [AI Milestones and Algorithms Timeline](appendix/ai-milestones) |

Learning AI is not about memorizing every historical milestone. The deeper pattern is always the same: an old limitation becomes visible, a new solution appears, and the new boundary pushes the next wave of innovation.
