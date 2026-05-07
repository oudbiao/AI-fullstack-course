---
title: "7.8.1 Pre-Class Guide: How Should You Actually Study This Capstone Project Chapter?"
sidebar_position: 0
description: "First, build the learning map for the Stage 7 project chapter: how domain tasks choose between Prompt, RAG, and fine-tuning, and how data, training, evaluation, and presentation form a complete LLM project loop."
keywords: [LLM project guide, domain fine-tuning, Prompt, RAG, LLM evaluation]
---

# 7.8.1 Pre-Class Guide: How Should You Actually Study This Capstone Project Chapter?

This chapter is not about piling on more terminology. Instead, it takes the overview of LLMs, pre-training, Prompt, fine-tuning, alignment, and evaluation that you learned earlier and places them into a concrete project.

The most important skill in an LLM project is not blindly fine-tuning when you see a problem, nor forcing every task into a Prompt. It is being able to judge whether the problem is unclear task expression, insufficient domain knowledge, unstable style, or missing evaluation criteria. Different problems require different solutions.

## Where This Chapter Fits in the Course

The theme of Stage 7 is understanding where LLM capabilities come from, and how Prompt, fine-tuning, and alignment affect model behavior. The capstone project is the exit point of this stage, and it should help you turn these concepts into verifiable engineering decisions.

You need to prove that you can center a project around a clear domain task, design a baseline, compare Prompt, RAG, or fine-tuning approaches, prepare samples, record results, and explain why you chose a certain path.

![LLM capstone project roadmap](/img/course/ch07-projects-route-map-en.png)

## The Real Problems This Chapter Solves

This chapter answers five questions: how to narrow an LLM project down to a clear domain and task; how to start with a Prompt baseline instead of jumping straight to a complex solution; how to decide whether to use Prompt, RAG, or fine-tuning; how to prepare data and an evaluation set; and how to present model performance, failure cases, and trade-offs.

A common mistake for beginners is to interpret “the model performs poorly” as “we need fine-tuning.” In real projects, many issues come from missing materials, unclear Prompt wording, unconstrained output format, insufficient evaluation examples, or a poorly designed application workflow.

## Recommended Learning Order for Beginners

It is recommended that you first choose a small and clear domain task, such as course Q&A classification, chapter summary structuring, customer intent recognition, contract clause classification, or learning advice generation. Then write a Prompt baseline, collect a set of test examples, and record both successes and failures. Next, identify the failure type: if supporting context is missing, consider RAG; if fixed-format or style output is unstable, consider fine-tuning; if the task itself is vague, first rewrite the task definition and evaluation criteria.

![LLM project method-selection loop](/img/course/ch07-project-method-choice-loop-en.png)

## The Main Thread to Focus on While Studying This Chapter

The core idea of this chapter can be summarized as: an LLM project is not about “choosing the strongest model,” but about making trade-offs around task, data, method, and evaluation.

Once you understand this line, you will know why project reports need to explain “why not use another solution.” This kind of trade-off explanation shows your LLM engineering ability more clearly than simply demonstrating results.

## What This Project Is Really Training

This project is really training four things: narrowing a task into a clear domain, first building a Prompt baseline, deciding whether to keep optimizing the Prompt, connect RAG, or do fine-tuning, and finally using an evaluation set and failure cases to prove the solution works.

If you choose a domain fine-tuning project, pay special attention to data quality, the training/validation split, format stability, and comparison with the baseline. If you choose a RAG solution, focus on the source of knowledge, chunking, retrieval, citations, and how to handle “no answer” cases. If you choose a Prompt-based solution, focus on structured output, example design, and version iteration.

## How This Chapter Connects to Later Stages

This chapter directly leads into Stage 8 on LLM application development and RAG. The solution-selection ability you build here will later expand into knowledge-base Q&A, intelligent assistants, tool calling, and Agent systems.

If you do not learn this chapter solidly, common problems later will be: adding frameworks whenever you see a problem; mixing fine-tuning, RAG, and Prompt into one category; having no evaluation set; being unable to explain why a solution works; and presenting only the final answer without failure analysis or trade-off reasoning.

## How Beginners and Advanced Learners Should Read This

When beginners study this chapter for the first time, they should first focus on the main thread and the smallest runnable example. You do not need to understand every detail at once. As long as you can explain what problem this chapter solves, what the inputs and outputs are, and how the smallest project runs, you can move on.

More experienced learners can treat this chapter as a chance to fill gaps and practice engineering: focus on boundary conditions, failure cases, evaluation methods, code reproducibility, and how it connects to the earlier and later stages. After reading, it is best to turn the chapter content into your own project README or experiment notes.

## Suggested Study Time and Difficulty

| Study mode | Suggested time | Goal |
|---|---|---|
| Quick skim | 20–30 minutes | Understand what this chapter solves and where it will be used later |
| Minimum pass | 1–2 hours | Run a minimal example and complete the chapter’s project exit |
| Deep practice | Half a day to 1 day | Add error analysis, comparison experiments, or project README notes |

## Self-Check Questions for This Chapter

| Self-check question | Passing standard |
|---|---|
| What problem does this chapter solve? | Can explain its role in the whole course in one sentence |
| What are the minimum inputs and outputs? | Can clearly state what the example needs as input and what it will produce |
| Where are the common failure points? | Can list at least one cause of an error, poor result, or misunderstanding |
| What can you retain after finishing? | Can write the output of this chapter into a project README, experiment log, or portfolio |

## Chapter Project Exit

After finishing this chapter, it is recommended to complete a “domain task LLM solution comparison project.” The project should include at least task definition, sample data, Prompt baseline, failure analysis, improvement plan, evaluation results, and a conclusion.

If you need a guided starter before designing your own domain project, run [7.8.4 Hands-on: Full Chapter 7 Workshop](./03-stage-hands-on-workshop.md) first. It gives you a complete offline practice loop with fixed cases, prompt-version pass rates, structured-output failures, and solution-route notes.

The minimum version can compare just two Prompt versions; the advanced version can compare Prompt, RAG, and a small-scale fine-tuning run; the portfolio version should clearly explain method trade-offs, evaluation metrics, failure cases, and next steps.

Before you close the project, it is also a good idea to use the Deliverables Kit page as a checklist for README structure, evaluation records, failure notes, screenshots, and next steps. A project becomes more convincing when the handoff is easy to understand and easy to reproduce.


## Debug Detective Case

| Case | Content |
|---|---|
| Case name | JSON drift incident |
| Scene | The LLM output sometimes misses fields and sometimes adds extra explanations, making structured results impossible to parse reliably. |
| Investigation steps | Fix 10 inputs, save the raw outputs, validate with a schema, and compare Prompt versions. |
| Closing evidence | `prompt_eval_cases.csv`, Prompt version table, schema pass rate. |

When practicing the project, do not keep only success screenshots. At minimum, pick one real failure sample and write it into `reports/failure_cases.md` in the format “phenomenon, clues, suspected cause, investigation steps, fix action, regression check.” This will make the project feel much more like a real engineering work.

## Project Deliverable Standards

For each LLM capstone project, it is recommended to deliver it according to the same portfolio standard rather than only showing one model response. The minimum deliverables should include: a README, one reproducible run command, a set of example inputs and outputs, Prompt version records, one failure sample analysis, and a next-step improvement plan.

| Deliverable | Minimum requirement | Advanced requirement |
|---|---|---|
| README | Clearly state the project goal, how to run it, the model, and examples | Add solution trade-offs, cost estimate, evaluation, and retrospective |
| Example inputs and outputs | Keep at least 1 fixed test sample | Keep comparison samples for Prompt, RAG, fine-tuning, or rule-based solutions |
| Evaluation record | Clearly state the standard used to judge output quality | Add a fixed evaluation set, human scoring, and failure-type statistics |
| Prompt/data record | Save Prompt versions or training sample format | Add schema, validation, data quality, and safety boundary notes |
| Presentation material | Screenshots or short GIFs to prove it runs | Turn it into a teachable case of LLM solution selection |

The most important thing in LLM projects is not “does the model answer convincingly,” but being able to clearly explain whether the problem comes from task expression, knowledge gaps, format stability, or missing evaluation, and why you chose the current technical path.

## Passing Criteria

By the end of this chapter, you should be able to build a baseline around a domain task, judge what problems Prompt, RAG, and fine-tuning are each best at solving, prepare a small evaluation set, explain model limitations with failure cases, and write the technical trade-offs into a project report.

If you can clearly explain “why not fine-tune here,” “why RAG is needed here,” and “why this Prompt change works,” then you have reached the portfolio exit standard for the LLM principles and fine-tuning phase.

## Version Roadmap Suggestions

| Version | Goal | Deliverable focus |
|---|---|---|
| Basic | Run the minimum loop | Can take input, process it, output it, and keep one set of examples |
| Standard | Form a presentable project | Add configuration, logs, error handling, README, and screenshots |
| Challenge | Approach portfolio quality | Add evaluation, comparison experiments, failure sample analysis, and next-step roadmap |

It is recommended to finish the basic version first. Do not aim for something huge and complete right away. Each time you level up, make sure to write “what new capability was added, how it was validated, and what problems remain” into the README.
