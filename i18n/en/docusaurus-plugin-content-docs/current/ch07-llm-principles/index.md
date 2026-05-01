---
title: "7 LLM Principles, Prompt, and Fine-tuning"
sidebar_position: 0
description: "Understand the development history of large language models (LLMs), Token, Embedding, Transformer, pretraining, Prompt, fine-tuning, and alignment, laying the foundation for LLM applications and Agents."
keywords: [large language models, LLM, Transformer, Prompt Engineering, LoRA, fine-tuning, RLHF]
---

# 7 LLM Principles, Prompt, and Fine-tuning

![Main visual for LLM principles](/img/course/ch07-llm-principles.png)

This stage is about answering: “Where do LLM capabilities come from, and how are they controlled and adapted?” You are not just learning a few model names; you are understanding the relationships among Token, Embedding, Transformer, pretraining, Prompt, fine-tuning, and alignment.

## Story-driven introduction: Opening the magic box of chatbots

An LLM looks like magic: you type a sentence, and it can write code, summarize documents, play roles, and plan tasks. What this stage does is open that magic box: text is first split into Tokens, Tokens become vectors, the Transformer computes relationships in context, pretraining gives the model language ability, and Prompt and fine-tuning then guide that ability toward specific tasks.

## Learning quest map

![LLM learning quest map](/img/course/ch07-learning-quest-map.png)

## Interactive exercise: Write Prompts like doing experiments

Don’t just ask, “Is this Prompt good?” Instead, change only one variable each time: whether you add a role, provide examples, require step-by-step reasoning, constrain the output format, or include grading criteria. Save different versions of inputs and outputs, and you will gradually build your own Prompt experiment handbook.

## Project bonus content

The bonus project for this stage is a “Prompt Experiment Atlas”: choose the same task, design at least five prompting methods, and compare output quality, stability, format controllability, and cost. When you later work on RAG, Agents, structured outputs, and tool calling, this atlas will become your starting point for prompt engineering.

## Stage overview

| Information | Description |
|---|---|
| Suitable for | Learners who have completed the basics of deep learning and Transformer and want to move into LLM applications, RAG, or Agent learning |
| Estimated study time | 90–120 hours |
| Prerequisites | Complete Stage 6: Deep Learning and Transformer Basics; if your NLP foundation is weak, you can pair this with Chapter 11 Natural Language Processing (elective track) or the NLP crash course in this stage |
| Stage deliverables | Prompt experiments, structured output tasks, domain fine-tuning, or fine-tuning plan design |

## Minimal path for beginners

Beginners should first understand the relationships among Token, Embedding, Attention, Transformer, pretraining, Prompt, fine-tuning, and alignment. You do not need to train your own large model at the beginning. As long as you can design stable Prompts, make the model output structured results, and judge whether a task is better suited to Prompt, RAG, or fine-tuning, you have completed the minimal path.

## Advanced path

Experienced learners can go deeper into Transformer variants, pretraining data, LoRA/QLoRA, instruction fine-tuning, RLHF, and model evaluation. You can further try designing a fine-tuning plan for a domain task, clearly defining the data format, training cost, evaluation metrics, and risk boundaries.

## Where LLMs fit in AI history

LLMs did not appear out of nowhere; they build on deep learning, NLP, Transformer, and the pretraining paradigm. The real change is this: model scale, data scale, and instruction alignment turned language models from “solving a single NLP task” into “using a language interface to complete many tasks.”

![Main backbone of LLM capability sources](/img/course/ch07-llm-capability-backbone.png)

## What beginners should do first, and what advanced learners should do later

When beginners study this stage for the first time, they should first think of LLMs as “interfaces that complete tasks based on context.” Practice stable Prompting, structured output, and simple evaluation first, then understand why pretraining, fine-tuning, and alignment work the way they do.

Experienced learners can focus on solution selection: when is Prompt enough, when is RAG needed, when should you consider fine-tuning, and how should you design an evaluation set to verify the results? Your goal is not to chase model names, but to choose the right LLM solution for a real problem.

## Learning path for this stage

Chapter 1 starts with a fast NLP crash course, including tokenizer, embedding, pretrained models, and a quick hands-on with HuggingFace.

Chapter 2 covers the LLM overview to understand the development history, core concepts, and industry landscape of LLMs.

Chapter 3 goes deep into Transformer, focusing on architecture, variants, efficient attention, and scaled computation.

Chapter 4 covers pretraining techniques, including data, training methods, and engineering challenges.

Chapter 5 covers Prompt Engineering, helping you understand how to shape model behavior through input design.

Chapter 6 covers fine-tuning, focusing on LoRA, QLoRA, PEFT, and data labeling.

Chapter 7 covers RLHF and alignment, explaining why a model being powerful does not mean it is reliable, controllable, or safe.

## What you should be able to do after finishing

- Explain the basic meaning of Token, Embedding, Attention, and context window
- Clearly describe the differences among pretraining, instruction fine-tuning, Prompt, and fine-tuning
- Design basic Prompts and require the model to output structured results
- Judge whether a task is better suited to Prompt, RAG, or fine-tuning
- Understand the purpose of LoRA/QLoRA and their applicable boundaries
- Build intuition about model behavior for later LLM applications, RAG, and Agent systems

## Common misconceptions

Do not think of an LLM as “a bigger search engine” or “a database with knowledge.” An LLM is still fundamentally a model that generates tokens based on context. It may produce incorrect content, and it may behave unstably because the Prompt, context, or task definition is unclear.

Also, do not rush into fine-tuning. Many application problems should first be solved with Prompting, structured output, RAG, or system design. Fine-tuning is usually not the first step.

## LLM error theater: what to check first when answers are unstable

If the model output goes off track, first check whether the task description is clear, whether the output format includes examples, and whether the context contains conflicting information. If structured output often fails, first reduce the number of fields, add examples, and add parsing validation. If the result is still unstable, compare Prompt, RAG, or fine-tuning approaches using a fixed question set instead of judging based on a single experience.

## How to read it the first time: must-read, project reference, and elective deep dives

| Reading label | Suggested sections | Learning goal |
|---|---|---|
| Must-read | NLP crash course, LLM core concepts, Prompt basics, structured output | First understand model input/output, context, and task expression |
| Project reference | Prompt practice, fine-tuning overview, data labeling, stage project | Focus on this when doing Prompt comparison or domain adaptation projects |
| Elective deep dive | Transformer deep dive, pretraining engineering, LoRA/QLoRA, RLHF and alignment | Go deeper only if you want to focus on model understanding, fine-tuning, or evaluation |

On your first pass, don’t treat all Transformer and pretraining details as memorization material. More importantly, you should be able to judge whether a task should use Prompt, RAG, fine-tuning, or Agent.

## Stage review card: from model principles to task adaptation

After finishing this stage, you can use the table below to check whether you truly understand “where LLMs come from, how to use them, and how to adapt them.”

| Review question | What you should be able to answer |
|---|---|
| Token and Embedding | Why must text be split into tokens first, then converted into vectors? |
| Transformer | Why can the attention mechanism handle contextual relationships? |
| Pretraining | What does the model mainly learn from data and training objectives? |
| Prompt | Which problems can be solved first by making the task description clearer? |
| Fine-tuning | Which problems belong to long-term behavior adaptation rather than knowledge updates? |
| Alignment | Why does “the model can answer” not mean “the model is reliable, safe, and aligned with intent”? |

The real exit point of this stage is not being able to recite many model names, but being able to judge: should a task be solved with Prompt, structured output, RAG, fine-tuning, or a later Agent system?

## Stage project

The basic version is to complete a Prompt comparison experiment and record the differences in performance between plain prompts, role prompts, step-by-step prompts, and structured output. The standard version requires building a Prompt Experiment Atlas around the same task and comparing stability, format control, cost, and error types. The challenge version can design a domain fine-tuning plan or a minimal fine-tuning experiment, explaining the data source, labeling rules, training method, evaluation approach, and safety risks.

If you want a more detailed learning rhythm, you can read [Learning Guide: How to Learn LLM Principles Without Getting Confused](./study-guide.md).




## Fun task card for this stage

| Play style | Task for this stage |
|---|---|
| Story task | Make the assistant express itself stably: design Prompts, constrain structured output, and check drift with fixed inputs. |
| Boss fight | **JSON Drift Monster** |
| Unlockable badges | Prompt Tuning Master, Schema Guardian |
| Beginner easy mode | Complete just one minimal input-to-output loop and keep a screenshot or command output |
| Portfolio evidence | Prompt version table and schema validation results |

If you feel there is a lot of content in this stage, first treat this task card as your minimum goal. Once you can complete the beginner easy mode, you can continue learning; later, when preparing your portfolio, come back and upgrade to the standard version and challenge version.

## Stage deliverables

| Deliverable | Minimum version | Portfolio version |
|---|---|---|
| Prompt comparison experiment | Compare plain prompts, role prompts, and step-by-step prompts | Include fixed inputs, output comparisons, version records, and failure samples |
| Structured output sample | Make the model output JSON or a table | Include schema, parsing validation, error retries, and regression samples |
| Task adaptation decision | Explain when to use Prompt, RAG, or fine-tuning | Include a decision table, cost estimate, and applicable boundaries |
| Fine-tuning plan | Write down the data source and labeling ideas | Include training method, evaluation set, safety risks, and alternatives |
| README/report | Show the experimental inputs and outputs | Explain Prompt versions, metrics, failure types, and next steps |

## Relationship to the AI learning assistant capstone project

This stage can map to AI Learning Assistant v0.7: connect to the LLM API to generate study plans, review cards, Prompt templates, and structured summaries. If you are following the capstone project path, it is recommended that by the end of this stage you submit at least one version record: what new capability was added in this stage, how to run it, what the sample input/output looks like, what problems you encountered, and what you plan to change next.


## Stage completion criteria

| Completion level | What you need to achieve |
|---|---|
| Minimum completion | Be able to explain the boundaries of Transformer, pretraining, Prompt, fine-tuning, and alignment |
| Recommended completion | Complete at least one runnable mini project for this stage and record the run method, sample input/output, and problems encountered in the README |
| Portfolio completion | Integrate the output of this stage into the “AI Learning Assistant” capstone project, leaving screenshots, logs, evaluation samples, and next-step plans |

After finishing this stage, you do not need to memorize every detail. More importantly, you should be able to clearly explain: what problem this stage solves, how it relates to the previous stage, and how it will support later learning. The next stage will connect LLMs to RAG and application systems.
