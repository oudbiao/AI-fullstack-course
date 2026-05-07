---
title: "A.5 Hardware and Cloud Resource Guide"
sidebar_position: 2
---

# A.5 Hardware and Cloud Resource Guide

![Hardware and Cloud Resource Decision Tree](/img/course/appendix-hardware-cloud-decision-tree-en.png)

![Cost comparison chart for local, cloud, and API approaches](/img/course/appendix-hardware-local-cloud-api-cost-map-en.png)

:::tip Reading the charts
When choosing hardware, start with the task: learning fundamentals, running demos, training models, and building applications all have very different resource needs. When reading the chart, compare your local computer, cloud GPU, and API options side by side, and don’t get trapped from the start by the idea that you “must buy a graphics card.”
:::

This page is not about helping you “pile on specs.” It is here to help you decide: at different stages, should you buy hardware, when is renting cloud resources more cost-effective, and which specs actually improve the learning experience.

## Start with the conclusion

- Learning stations 1–5 basically do not need a GPU.
- 6 Deep Learning and Transformer fundamentals, 10 Computer Vision, and 11 Natural Language Processing can start with cloud GPUs; no need to rush into buying a graphics card.
- 8 LLM Application Development and RAG, and 9 AI Agent and intelligent agent systems, involve many projects that actually rely more on engineering skills than on a powerful local GPU.
- For beginners, the usual spending priority is: memory > storage > stable environment > GPU.

## Hardware needs by stage

| Learning station | Common tasks | Minimum local recommendation | More comfortable setup | Cloud alternative |
|---|---|---|---|---|
| 1 Developer Tools Basics, 2 Python Programming Basics, 3 Data Analysis and Visualization | Python, data cleaning, charting, Notebook | 8GB RAM / 256GB SSD | 16GB / 512GB SSD | Not needed |
| 4 Minimum Essential AI Math Basics, 5 Introduction to Machine Learning and Practice | Math experiments, traditional machine learning | 8GB RAM | 16GB RAM | Not needed |
| 6 Deep Learning and Transformer Fundamentals | PyTorch, CNN, RNN, Transformer basics | 16GB RAM | 16GB+ / 512GB SSD | Colab / Cloud GPU |
| 7 LLM Principles, Prompting, and Fine-Tuning | LLM principles, Prompt experiments, fine-tuning concept experiments | 16GB RAM | 32GB RAM is better | Cloud GPU is more realistic |
| 8 LLM Application Development and RAG, 9 AI Agent and Intelligent Agent Systems | RAG, Agent, application development | 16GB RAM | 16GB+ / stable network | CPU + API is enough for most projects |
| 10 Computer Vision, 11 Natural Language Processing | CV, NLP experiments | 16GB RAM | 16GB+ / external monitor is more comfortable | Colab / AutoDL / cloud server |
| 12 AIGC and Multimodal | Multimodal trials, project assembly | 16GB RAM | 32GB RAM is smoother | Image and video generation is more suitable for the cloud |

## Three hardware tiers

### Enough-to-get-by tier

Suitable for:

- People just starting systematic learning
- Limited budgets
- Mainly running teaching examples and small-to-medium projects

Recommendations:

- 16GB RAM
- 512GB SSD
- A regular CPU laptop or desktop
- Use cloud resources temporarily when a GPU is needed

### Comfortable tier

Suitable for:

- Frequent local experiments
- Running browser, IDE, Notebook, and terminal at the same time
- Wanting a stable long-term learning experience

Recommendations:

- 32GB RAM
- 1TB SSD
- Multi-core CPU
- Add a mid-range discrete GPU if needed, or keep cloud GPU as the main option

### Heavy experimentation tier

Suitable for:

- People who are clearly going to do training experiments
- Frequent large vision or deep learning tasks
- Needing fast local iteration

Recommendations:

- 32GB–64GB RAM
- 1TB SSD or larger
- A newer graphics card
- But first ask yourself: is this really for learning, or just “device anxiety”?

## For beginners, the most valuable specs are not necessarily the GPU

Many people initially overestimate the value of the GPU and underestimate the importance of memory and storage.

What usually affects the experience more is:

1. Enough memory
   When the browser, VS Code, Jupyter, and terminal are open together, the difference between 16GB and 8GB is very noticeable.

2. Enough SSD storage
   Datasets, model caches, and virtual environments all take up space. A too-small disk will force you to clean up constantly and can disrupt your learning rhythm.

3. A stable environment
   Even on the same machine, if the environment is messy, you can waste a lot of time debugging.

4. Comfortable display and input devices
   When you spend long hours reading code and docs, an external monitor, keyboard, and mouse make a real difference.

## How to choose cloud resources more reasonably

### Free or low-cost trial use

Suitable for:

- Running course examples
- Doing small experiments
- Getting familiar with the GPU training workflow

Features:

- Quick to start
- Low cost
- Not suitable for long-term stable training

### Hourly cloud GPU rental

Suitable for:

- Long training times
- Needing more stable VRAM and compute
- Wanting to control costs without buying hardware upfront

Suggestions:

- First get the workflow working on a small dataset, then move to the cloud
- Prepare your code, data, and experiment plan before starting the machine
- After training, export results promptly and shut down to stop the cost

### API-first application route

Suitable for:

- Building RAG, Agent, and intelligent assistant applications
- Focusing more on product and engineering than on training a foundation model yourself

This route usually does not require a local GPU. The focus is on:

- API wrapping
- Logging and monitoring
- Retrieval and tool calling
- User experience and cost control

## When is it worth buying a local GPU?

Only when at least two of the following are true does buying a local GPU make more sense:

- You are sure you will do training experiments for a long time
- You are often interrupted by cloud queues and time limits
- You have a clear need for faster local iteration
- You already know which tasks you mainly run and how much VRAM you need

If you are still at the stage of “maybe I’ll need it later,” don’t rush to buy one.

## Common misconceptions

### Misconception 1: You can’t learn AI without a graphics card

This is the most common misunderstanding. Many abilities in the early stages have nothing to do with a graphics card, and many application engineering projects in later stages can also be completed with APIs and cloud resources.

### Misconception 2: Buy all the gear first, then start learning

What hurts learning most is not ordinary hardware; it is delaying the start. It is usually more reasonable to move forward with the hardware you already have and upgrade only when you hit a real bottleneck.

### Misconception 3: Sacrificing overall stability for a graphics card

If you buy a high-end GPU but the RAM, storage, cooling, and system stability can’t keep up, the actual experience may still be poor.

## A realistic buying recommendation

If you are getting ready to start now, the most stable plan is usually:

- Use your current computer to complete learning stations 1–5
- Rent cloud GPUs as needed for 6 Deep Learning and Transformer Fundamentals, 10 Computer Vision, and 11 Natural Language Processing
- Prioritize application projects for 8 LLM Application Development and RAG, and 9 AI Agent and Intelligent Agent Systems
- Once you are sure you will be doing training for the long term, then decide whether to upgrade your local hardware

This is more rational than spending your whole budget on hardware from the start, and it is also better suited for long-term learning.
