---
sidebar_position: 5
title: "Course Boss Battle Challenge Map"
description: "Design the core capabilities of each stage as lightweight Boss battles, and use Basic, Standard, and Challenge versions to help learners complete fun and verifiable stage checkpoints."
keywords: [Boss Battle, AI learning challenge, stage checkpoint, fun learning, project challenge]
---

# Course Boss Battle Challenge Map

![Course Boss Battle Challenge Map](/img/course/boss-challenge-map-en.png)

Boss battles are not meant to add pressure. They are meant to help learners understand “what exactly do I need to master in this stage?” Each Boss is a small, integrated challenge that requires you to connect the most important abilities from that stage and leave behind evidence you can show.

Boss battles come in three levels: Basic guarantees beginners can complete it, Standard is suitable for a portfolio, and Challenge is for people who have extra capacity. For the first pass through the course, you only need to clear the Basic version.

## The 4-step process for defeating a Boss

```mermaid
flowchart LR
  A["Choose the Basic version"] --> B["Complete the minimum task"]
  B --> C["Save success and failure evidence"]
  C --> D["Write a reflection"]
  D --> E["Upgrade only when you need a portfolio piece"]
```

On your first pass, do not chase the Challenge version. The value of a Boss battle is not showing off; it is making sure every stage has evidence that proves you truly learned it.

## Boss Battle Overview

| Stage | Boss Name | Core Skill | Basic Pass Evidence |
|---|---|---|---|
| 1 Developer Tools Basics | Workshop Gatekeeper | terminal, environment, Git, README | From an empty folder to one Git commit |
| 2 Python Programming Basics | JSON Dungeon Keeper | functions, files, exceptions, data structures | A CLI that can save tasks |
| 3 Data Analysis and Visualization | Dirty Data Detective | cleaning, statistics, charts, explanation | A data quality report |
| 4 AI Math Basics | Metrics Maze | vectors, probability, loss, metrics | A runnable mini experiment |
| 5 Machine Learning | Baseline Guardian | splitting, baseline, evaluation, error samples | A trustworthy baseline |
| 6 Deep Learning | Shape Beast | tensor, training loop, loss, curves | One training log and curve |
| 7 Prompt and Large Models | JSON Drift Monster | Prompt, schema, structured output | 10 fixed input-output tests |
| 8 RAG | Citation Hallucination Dragon | chunk, retrieval, citation, evaluation | 10 Q&A examples with citations |
| 9 Agent | Infinite Loop Demon King | tools, trace, stop condition, permissions | 3 replayable task traces |
| 10–12 Direction Extensions | Multimodal Chaos Entity | vision, text, multimodal, review | One complete input-to-output case |
| Graduation Project | Final Product Boss | integrated design, deployment, evaluation, demo | A runnable Demo and evaluation report |

For every Boss battle, keep both success and failure evidence. Success proves you can do it; failure proves you can reflect on it.

## Boss 1: Workshop Gatekeeper

This Boss checks whether you truly have a reproducible development environment. Many later problems come from not having this first stop solid: not knowing the current directory, not reading errors, not committing versions, and not writing run commands.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Create a project folder, write `hello_ai.py`, and run it | The terminal prints a sentence |
| Standard | Add a README, virtual environment instructions, and a Git commit | Others can reproduce it by following the README |
| Challenge | Deliberately create a path error and write a debugging record | There are failure samples and a fix record |

After passing, you should have “terminal survival” and “Git archiving” skills.

## Boss 2: JSON Dungeon Keeper

This Boss checks whether you can write a small program that really saves data. It does not need a complex interface, but it must handle normal input, empty input, and broken files.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Add, view, and complete learning tasks | Data can be saved to JSON |
| Standard | Support categories, search, and exception handling | Empty files and corrupted JSON do not crash immediately |
| Challenge | Write 3 command-line test cases | Normal, exception, and empty input are all recorded |

After passing, you should be able to explain how lists, dictionaries, functions, file read/write, and exception handling combine into a small tool.

## Boss 3: Dirty Data Detective

This Boss checks whether you can turn imperfect data into trustworthy conclusions. Data analysis is not a chart-making contest; it starts with data quality.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Check for missing values, duplicates, and outliers | Output a data quality checklist |
| Standard | Clean the data and draw 2 charts | Each chart has one sentence of conclusion and one limitation |
| Challenge | Intentionally keep one wrong conclusion and explain why it is wrong | There is a before-and-after cleaning comparison |

After passing, you should be able to clearly explain where the data came from, what is not trustworthy, and what the conclusion boundaries are.

## Boss 4: Metrics Maze

This Boss checks whether you can turn abstract math concepts into runnable mini experiments. You do not need to become a mathematician, but you should be able to use code to explain common quantities in models.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Use code to compute the similarity of two vectors | You can explain what the result size means |
| Standard | Compare probability, loss, or distance metrics | There are hand-calculated examples and code examples |
| Challenge | Explain what differences appear when the same problem uses different metrics | There is a reason for the metric choice |

After passing, you should no longer fear the words similarity, probability, loss, and evaluation metric.

## Boss 5: Baseline Guardian

This Boss checks whether you can judge if model results are trustworthy. A model project without a baseline makes it hard to show that it really works.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Split train/test data and train a Dummy baseline | Baseline metrics are output |
| Standard | Train a real model and compare it with the baseline | There is a metrics table and error samples |
| Challenge | Check for data leakage or class imbalance | There is a leakage check record |

After passing, you should be able to explain whether the model is better than the “dumbest method.”

## Boss 6: Shape Beast

This Boss checks whether you can complete a deep learning training run and locate shape, loss, or data issues when errors happen.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Run a minimal training loop | There is loss output |
| Standard | Save training curves and validation metrics | You can explain whether overfitting is happening |
| Challenge | Deliberately create a shape mismatch and fix it | There are error logs and a fix record |

After passing, you should know that deep learning projects are not just about the final score; you also need to look at the training process.

## Boss 7: JSON Drift Monster

This Boss checks whether you can make an LLM output structured results reliably. The biggest fear in Prompt projects is succeeding once and drifting ten times.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Design a Prompt that outputs JSON | At least 5 outputs have complete fields |
| Standard | Use schema validation on 10 fixed inputs | There are pass rates and failure samples |
| Challenge | Compare Prompt versions | There is a version table and improvement record |

After passing, you should treat Prompt as a testable component, not a mysterious spell.

## Boss 8: Citation Hallucination Dragon

This Boss checks whether you can make RAG answer based on documents and prove that citations support the answer.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Import 3 Markdown documents and answer 5 questions | Each answer gives a source |
| Standard | Build 10 evaluation questions and check citation_ok | There are retrieval logs and a citation check table |
| Challenge | Compare different chunk or top-k strategies | There are statistics on failure types |

After passing, you should be able to tell the difference between “the answer looks right” and “the answer is supported by sources.”

## Boss 9: Infinite Loop Demon King

This Boss checks whether you can design a controllable Agent. The hard part of an Agent is not being able to call tools; it is being able to stop, reflect, and limit permissions.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | Make the Agent complete 3 fixed tasks | Each task has step records |
| Standard | Save `tool_calls` and `agent_traces` | You can replay one failure |
| Challenge | Add human confirmation for high-risk actions | There are privilege-escalation tests and safety notes |

After passing, you should be able to explain when an Agent should act automatically and when it must stop and ask a person.

## Final Boss: Showable AI Product

The final Boss is not about stacking all technologies together. It is about turning a clear problem into a product that can run, be evaluated, and be demonstrated.

| Difficulty | Task | Pass Condition |
|---|---|---|
| Basic | A Demo that runs locally | README and sample inputs/outputs are complete |
| Standard | Has an evaluation set, logs, failure samples, and a demo script | You can explain the effect and limitations |
| Challenge | Deploy it and explain cost, safety, and monitoring | There is a complete portfolio page |

For the final presentation, it is recommended to tell the story with a “Boss battle clear record”: I first got the environment working, then did data, then models, then RAG and Agent, and finally integrated them into a product.

## Boss Battle Record Template

```md
## Boss Battle: Citation Hallucination Dragon

### Challenge Goal
Make RAG answer 10 course questions and check whether citations support the answers.

### Difficulty
Standard.

### Pass Evidence
Save `eval_questions.csv`, `retrieval_logs.jsonl`, and `citation_check.csv`.

### Failure Sample
Question: What is the difference between Agent and RAG?
Failure: Retrieval only matched the RAG page and did not match the Agent page.

### Fix Action
Expand the imported document scope and save stage information in metadata.

### Next Challenge
Compare how different chunk sizes affect the hit rate.
```

The meaning of Boss battles is to turn learning into clear levels. Every time you clear a level, you gain a capability that is explainable, demonstrable, and reviewable.
