---
title: "Stage Learning Task Sheet"
description: "Break down LLM principles, Prompt, and fine-tuning into executable learning tasks, practice deliverables, and pass criteria."
keywords: [LLM, Prompt, structured output, fine-tuning, learning task sheet]
---

# Stage Learning Task Sheet: LLM Principles, Prompt, and Fine-tuning

The goal of this stage is to help you understand why LLMs can generate text, how Prompt affects outputs, and when you need structured output, RAG, or fine-tuning. Don’t treat Prompt as magic; treat it as an engineering input that can be versioned, tested, and reviewed.

## Required tasks for this stage

| Task | Deliverable | Pass criteria |
| --- | --- | --- |
| Run the full Chapter 7 hands-on workshop | Terminal output from `llm_stage_workshop.py` | Can explain tokens, prompt versions, validation failures, and solution routes from one run |
| Understand the basic principles of LLMs | An LLM workflow diagram | Can explain the relationship between token, context, probability-based generation, and Transformer |
| Complete Prompt fundamentals practice | A set of Prompt comparison samples | Can compare how role, context, examples, and constraints affect outputs |
| Complete structured output | A JSON output demo | Can define fields, validate types, and handle parsing failures |
| Understand the boundary between RAG and fine-tuning | A technical decision table | Can explain when to use Prompt, RAG, fine-tuning, or rules |
| Complete the stage project | A Prompt assistant or review-card generator | Has Prompt versions, fixed inputs, output comparisons, and failure samples |

## Recommended learning order

First run the full Chapter 7 hands-on workshop once, then understand token, context, and generation mechanisms. After that, learn Prompt fundamentals, advanced Prompting, structured output, and the boundary of fine-tuning. On your first pass, don’t rush to train a model. First make sure you can “call and constrain model outputs reliably.”

Every time you change a Prompt, you should record the version and test samples. Prompt engineering is not about writing a longer sentence based on intuition; it is about controlling variables, comparing outputs, recording failures, and locking in effective changes.

## Relationship to the AI Learning Assistant project

This stage corresponds to version v0.7 of the AI Learning Assistant Prompt assistant. You can let the system generate study plans, knowledge cards, review summaries, or mistake explanations based on learning records, and record Prompt versions and failure samples.

A recommended minimum feature set includes: input study topic and current level, output structured study suggestions; save Prompt versions; use fixed samples to test whether outputs are stable.

## Common sticking points

Common problems include Prompts getting longer and longer but producing unstable results, JSON outputs with extra explanations, drifting field types, the model hallucinating nonexistent information, and old samples getting worse after a Prompt change. When troubleshooting, first look at the input, Prompt version, raw model output, and parsing logic; don’t look only at the final displayed text.


## Easy / Standard / Challenge tasks

| Difficulty | What you need to do | Suitable for |
|---|---|---|
| Easy | Make the model output in a fixed format 5 times | First-time learners, learners with limited time, or beginners |
| Standard | Validate 10 fixed inputs with a schema | Learners who want to include this stage in their portfolio |
| Challenge | Compare two Prompt versions and write an improvement log | Learners with a foundation who want stronger project evidence |

## Badges and Boss fight for this stage

| Type | Content |
|---|---|
| Boss fight | JSON Drift Monster |
| Unlockable badges | Prompt Tuner, Schema Guardian |
| Minimum pass mantra | Make it run first, then explain it, then record failures |
| Evidence-saving suggestion | Save screenshots, logs, failure samples, or evaluation tables to `reports/`, `evals/`, or `logs/` |

Once you complete the Easy version, you can move on; only the Standard version is recommended for your portfolio; do the Challenge version only if you have extra capacity.

## Stage portfolio deliverables

If you want to preserve the results of this stage in your portfolio, it is recommended to keep at least the following files or equivalent materials.

| Deliverable | Description |
| --- | --- |
| `prompts/` | Save Prompt templates, version numbers, applicable tasks, and change notes |
| `prompt_eval_cases.csv` | Fixed inputs, expected output points, actual outputs, and scores |
| `structured_output_schema.json` | Structured output fields, types, required items, and enum values |
| `failure_cases.md` | Samples such as JSON parsing failures, missing fields, hallucinations, and style drift |
| `llm_stage_workshop_output.txt` | Output from the full Chapter 7 hands-on workshop, including pass rates and failure reasons |
| `README.md` | Project goals, how to run it, Prompt versions, evaluation results, and limitations |

These materials will upgrade a Prompt project from “I can write prompts” to “I can turn model outputs into a stable, testable, and maintainable application interface.”

## Stage pass questions

After finishing this stage, you should be able to answer these questions: why token and context matter, why Prompt examples affect outputs, why structured output needs validation, when you should use RAG instead of adding more Prompting, and when fine-tuning is worth considering.

## Completion checklist

- [ ] I can explain the basic process of LLM text generation and context limits.
- [ ] I can run the full Chapter 7 hands-on workshop and explain each printed section.
- [ ] I can design a Prompt with a role, task, constraints, and examples.
- [ ] I can make the model output structured JSON and handle parsing failures.
- [ ] I can record Prompt versions and compare results with fixed samples.
- [ ] I can explain the applicable boundaries of Prompt, RAG, fine-tuning, and rule-based systems.
