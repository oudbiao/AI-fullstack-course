---
title: "Frequently Asked Questions"
sidebar_position: 3
---

# Frequently Asked Questions

![FAQ Newcomer Question Decision Tree](/img/course/appendix-faq-decision-tree-en.png)

![FAQ Anxiety Reset and Action Mapping Diagram](/img/course/appendix-faq-confidence-reset-map-en.png)

:::tip Reading Guide
The goal of the FAQ is not to give you one “single correct answer,” but to help you turn vague anxiety into actionable questions. When reading the diagrams, first locate whether you are stuck on math, hardware, time, projects, papers, or job hunting, then go back to the corresponding section and fill in only the minimum necessary content.
:::

This page does not explain concepts. Instead, it focuses on the practical questions that beginners ask most often. You can think of it as an appendix to consult first when you are feeling stuck.

## 1. Can I learn AI if I’m not good at math?

Yes. In fact, most people learn math as they go rather than finishing math first and then starting.

A better order for beginners is:

1. First get Python, data processing, and plotting working smoothly.
2. Then look back and see where linear algebra and probability/statistics are actually used in code.
3. Finally fill in formulas, derivations, and more abstract understanding.

If formulas overwhelm you right away, it usually does not mean you are unsuitable. It usually means the sequence is wrong.

A steadier approach is:

- When learning matrices, also look at array operations in `NumPy`.
- When learning gradient descent, also look at a minimal training loop.
- When learning probability, also look at the loss function and predicted probabilities in classification tasks.

Your goal is not to “prove things” first. It is to first understand what role this math tool plays in the model.

## 2. Do I need to buy a GPU?

Most beginners do not need one at the start.

| Learning stage | Main tasks | GPU required? | More realistic advice |
|---|---|---|---|
| 1 Developer Tools Fundamentals, 2 Python Programming Basics, 3 Data Analysis and Visualization | Development environment, Python, data analysis | No | A personal computer is enough |
| 4 Minimal Required Foundation in AI Math, 5 Introduction to Machine Learning and Practice | Math experiments, traditional machine learning | No | A personal computer is enough |
| 6 Deep Learning and Transformer Fundamentals | PyTorch, CNN, RNN, Transformer basics | Optional | Start with Colab or a cloud GPU |
| 7 LLM Principles, Prompt, and Fine-Tuning | LLM principles, Prompt, understanding fine-tuning | Depends on the experiment | First understand the workflow, then decide whether to rent a GPU |
| 8 LLM Application Development and RAG, 9 AI Agent and Agent Systems | RAG, Agent, application engineering | Usually not required | You can build many projects with just CPU + API |
| 10 Computer Vision, 11 Natural Language Processing, 12 AIGC and Multimodal | Vision, text, multimodal, and generative projects | Depends on the project | Training and generation tasks are better suited to cloud GPUs |

If this is your first systematic study, the usual priority is:

1. A stable computer setup
2. Enough memory and disk space
3. A learning pace you can keep up for the long term
4. Only after that, buying a graphics card

## 3. Do I have to study every stage?

Not necessarily, but it is recommended that you keep the main path going as much as possible.

A more practical way to understand it is:

- Learning stages 1–6: core foundation, recommended for most people
- Learning stages 7–9: the main LLM application path, helping you move from principles and RAG to Agent systems
- Learning stages 10–12: direction expansion and capstone projects, suitable for choosing CV, NLP, or AIGC multimodal directions based on interest

If your goal is to “build AI application projects as soon as possible,” you can start with:

`1 Developer Tools Fundamentals -> 2 Python Programming Basics -> 3 Data Analysis and Visualization -> 6 Deep Learning and Transformer Fundamentals -> 8 LLM Application Development and RAG -> 9 AI Agent and Agent Systems`

If your goal is to build a stronger foundation in model principles, you can follow:

`1–6 Core Foundations -> 7 LLM Principles, Prompt, and Fine-Tuning -> 8 LLM Application Development and RAG -> 9 AI Agent and Agent Systems`

## 4. How many hours per week is reasonable?

Compared with “how many hours a day,” what matters more is whether you can keep making steady progress for more than 12 weeks.

| Weekly time | Suggested pace | Suitable for |
|---|---|---|
| 4–6 hours | Slow, but sustainable | Working adults, busy students |
| 7–10 hours | Fairly ideal | Most self-learners |
| 12–18 hours | Fast progress, but can be tiring | People with clear project goals |

A common beginner mistake is to go hard in the first week and then stop in the third. A more stable approach is:

- 1 hour per day on weekdays
- 1 review session and 1 coding practice session on weekends
- One small project or experiment summary every 2–3 weeks

## 5. When is the best time to start doing projects?

The earlier, the better, but the project difficulty should match your stage.

A suitable pace is:

- After learning Python basics: small scripts and small tools
- After learning data analysis: data cleaning and visualization projects
- After learning machine learning: a complete classification or regression project
- After learning deep learning: a training loop project
- After learning RAG / Agent: a real application project that you can showcase

Do not think of “projects” as necessarily large systems. For beginners, what matters most is:

- The project goal is clear
- The input and output are clear
- You can run a minimal closed loop
- You can explain why you did it that way

## 6. What should I do if my code keeps failing to run?

First, do not rush to doubt yourself. Most problems come from the environment, paths, dependency versions, and input data.

You can troubleshoot in this order:

1. Look at the last 10 lines of the error message, not just the first line.
2. Confirm that you are in the correct Python environment.
3. Confirm that the dependencies are installed in the current environment.
4. Confirm that the file paths, data paths, and model paths are correct.
5. Reduce the problem to a minimal reproducible example.

It is recommended to learn these troubleshooting commands first:

```bash
python --version
which python
pip --version
pip list
pwd
ls
```

If you have not yet built troubleshooting habits, you can also check the appendix [Learning Roadblock Rescue](./troubleshooting.md).

## 7. If I don’t understand papers, does that mean I can’t learn AI well?

No.

For beginners, papers should be used to supplement understanding and broaden your perspective. They should not be the starting point for learning.

A better order is:

1. First read tutorials and understand the task, inputs and outputs, and evaluation method.
2. Then read the implementation or minimal code.
3. Finally, read the paper with your questions in mind.

If you start by forcing yourself to read papers, two problems often happen:

- You only remember terminology and do not develop task intuition
- You feel like you know nothing, and then become even less willing to try

## 8. How far do I need to learn before I can start looking for a job or taking projects?

A realistic standard is not “Have I learned everything?” but whether you have these four abilities:

1. You can set up the environment independently and run a medium-complexity project
2. You can explain the project’s data flow, model choice, evaluation method, and failure cases
3. You can understand common Python / ML / LLM engineering code
4. You can break requirements into executable small tasks and complete them step by step

If you already have 2–3 projects that you can explain clearly, you can actually start preparing your resume and interview practice instead of waiting until “everything is learned.”

## 9. I always feel like I’m learning too slowly. What should I do?

This is almost a stage every self-learner goes through.

The real comparison should not be “How much did others learn in a day?” but:

- Am I more able to understand code this month than last month?
- Am I starting to modify examples on my own instead of only copying them?
- Am I getting better at explaining a concept to someone else?

If the answers to these questions are increasingly “yes,” then you are making progress.

## 10. When I get stuck, what are the three most useful things to strengthen first?

If things already feel a bit messy, the most valuable areas to strengthen first are:

1. Python fundamentals and debugging ability
2. Data processing and visualization ability
3. The mindset of building a minimal closed loop for a complete project

Many people think what they lack is “more advanced models,” but what they really lack is:

- Understanding inputs and outputs
- Knowing how to find and fix errors
- Knowing how to break problems apart
- Knowing how to verify results

Once these four things are in place, the rest of the learning process becomes much smoother.
