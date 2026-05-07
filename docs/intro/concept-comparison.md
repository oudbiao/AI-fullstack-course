---
title: "Concept Sorting Board"
sidebar_position: 6
description: "A compact visual reference for AI full-stack terms that beginners often confuse: API, SDK, model, inference, Prompt, RAG, Agent, Token, Embedding, and deployment."
keywords: [AI Concepts, API, SDK, RAG, Agent, Fine-tuning, Token, Embedding]
---

# Concept Sorting Board

![Concept sorting board](/img/course/intro-concept-sorting-board-en.png)

Use this page when a term appears and you are not sure where it belongs. Do not memorize the whole glossary first. Sort the term into a box, then read the short meaning.

## 1. Sort Before You Memorize

| Box | Terms that usually belong here | Quick meaning |
|---|---|---|
| Calling and integration | API, SDK, library, framework, CLI, Web API | How your code reaches another capability |
| Model and serving | model, inference, training, fine-tuning, deployment | Whether you are using, changing, or serving a model |
| LLM workflow | Prompt, System Prompt, few-shot, structured output, function calling, tool use | How the model is instructed and connected to tools |
| Data and vectors | Token, Embedding, chunk, metadata, vector database, context window | How content is split, represented, retrieved, and shown to the model |
| Engineering delivery | logs, evaluation, latency, cost, security, monitoring | Whether the demo can become a reliable project |

## 2. Common Terms in One Line

| Term | Plain meaning | Do not confuse it with |
|---|---|---|
| API | A contract for calling a service | SDK, which is a helper package around calls |
| SDK | Developer tools that make an API easier to use | The service itself |
| Model | Learned parameters that generate or predict | The API wrapper around it |
| Inference | Using a model to get an output | Training, which changes parameters |
| Fine-tuning | Continuing training on an existing model | Prompt editing |
| Prompt | Instructions and input for this call | Private knowledge storage |
| RAG | Retrieve outside documents, then answer with context | Fine-tuning the model |
| Agent | A multi-step system that can plan and use tools | Any simple chatbot with one API call |
| Embedding | A vector representation of text, images, or other content | The vector database that stores vectors |
| Chunk | A document piece used for retrieval | The whole original document |

## 3. Three Fast Decisions

| If the problem is... | Start with... | Why |
|---|---|---|
| The answer format is unstable | Prompt and structured output | The task boundary is unclear |
| The model lacks your documents | RAG | The missing part is external knowledge |
| The task needs repeated actions | Workflow or Agent | The system must decide or execute steps |

For most beginners, the order is: improve the Prompt, add retrieval if knowledge is missing, add tool calling only when the task truly needs external actions.

## 4. Mini Practice

Read these three sentences and name the box:

| Sentence | Correct box |
|---|---|
| “The answer cites the wrong paragraph.” | Data and vectors, then RAG evaluation |
| “The JSON field is sometimes missing.” | LLM workflow |
| “It works locally but fails after deployment.” | Engineering delivery |

When writing a README, be precise: name the model service, the API or SDK, whether RAG or tools are used, how results are evaluated, and what the known limits are.
