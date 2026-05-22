---
title: "7.8.1 Project Roadmap: Choose Prompt, RAG, or Finetuning"
description: "A hands-on roadmap for the Chapter 7 capstone: define a domain task, build a prompt baseline, choose the right method, and present evidence."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "LLM project guide, domain fine-tuning, Prompt, RAG, LLM evaluation"
---
This capstone turns Chapter 7 into one engineering decision: is the problem task expression, missing knowledge, unstable format, unsafe behavior, or weak evaluation?

## See the Project Route First

![LLM capstone project roadmap](/img/course/ch07-projects-route-map-en.webp)

![LLM project method-selection loop](/img/course/ch07-project-method-choice-loop-en.webp)

![Portfolio evidence pack diagram](/img/course/ch07-hands-on-portfolio-evidence-pack-en.webp)

Do not start with the strongest model or the most complex framework. Start with a small domain task, a prompt baseline, fixed examples, and a failure log.

## Run an Evidence Pack Check

Use this tiny project log before writing a report. It forces you to show the baseline, improvement, next route, and whether finetuning is actually justified.

```python
project = {
    "task": "classify course questions",
    "baseline_pass_rate": 0.62,
    "prompt_v2_pass_rate": 0.78,
    "rag_needed": True,
    "finetune_needed": False,
}

improvement = project["prompt_v2_pass_rate"] - project["baseline_pass_rate"]

print("task:", project["task"])
print("improvement:", round(improvement, 2))
print("next_route:", "RAG" if project["rag_needed"] else "Prompt")
print("fine_tune_now:", project["finetune_needed"])
```

Expected output:

```text
task: classify course questions
improvement: 0.16
next_route: RAG
fine_tune_now: False
```

If your project cannot fill these fields, keep the project smaller. A clear comparison beats a large but untestable demo.

## Learn in This Order

| Step | Do | Evidence |
|---|---|---|
| 1 | Pick one domain task | One-sentence task definition and 10 fixed examples |
| 2 | Build a prompt baseline | Prompt version, outputs, pass/fail notes |
| 3 | Classify failure types | Task wording, missing knowledge, format drift, safety boundary |
| 4 | Choose the next method | Prompt iteration, RAG, or finetuning decision note |
| 5 | Package the result | README, run command, screenshots, failure case, next steps |

If you want a guided starter, run [7.8.4 Hands-on: Full Chapter 7 Workshop](./03-stage-hands-on-workshop.md) before designing your own domain project.

## Decision Rule: Name the Failure Before Choosing the Method

The capstone should not say “we used RAG” or “we fine-tuned” just because the method sounds advanced. First name the dominant failure.

| Dominant failure | Better first route | Evidence you need |
|---|---|---|
| The model does not know the answer source | RAG | Retrieved document supports the answer |
| The output format drifts | Structured output + validation | Parser pass rate improves |
| The instruction is vague | Prompt iteration | Same cases improve after one prompt change |
| The behavior repeats incorrectly across many cases | Fine-tuning / LoRA candidate | Enough labeled examples and held-out eval cases |
| The task needs external action | Tool / Agent route | Tool call trace and recovery behavior |

This one table keeps the project from becoming a method showcase. It turns it into an engineering decision.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
project_choice: Prompt, RAG, fine-tuning, or hybrid route
baseline: simplest working method first
evaluation: fixed cases and scoring rule
deliverable: README, prompts, outputs, failures, decision log
bridge: Chapter 8 turns this into retrieval-backed applications
```

## Project Deliverable Standards

| Deliverable | Minimum Standard | Stronger Portfolio Version |
|---|---|---|
| README | Goal, run command, model or API choice, sample input/output | Add method trade-offs, cost notes, evaluation, and retrospective |
| Examples | At least 10 fixed cases | Compare prompt, RAG, finetuning, or rule-based versions |
| Evaluation | Clear pass/fail rule | Add scores, failure-type statistics, and regression notes |
| Prompt/data record | Save prompt versions or sample format | Add schema validation, data quality checks, and safety notes |
| Presentation | Screenshot or short GIF proving it runs | Explain why the chosen route beats alternatives |

## Pass Check

You pass this chapter when you can clearly explain “why not finetune here,” “why RAG is needed here,” or “why this prompt change works,” with a fixed evaluation set rather than a single good answer.

The final project can be basic: compare two prompt versions on one domain task. The stronger version adds RAG or a small finetuning experiment, but only after the baseline and failure log prove the need.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer explains how tokens, context, attention, prompts, and generation behavior connect in one request-response path.
2. The evidence should include at least one reproducible prompt or structured-output test, plus notes on why the output passed or failed.
3. A good self-check separates prompt design, RAG, fine-tuning, and alignment: use the lightest method that fixes the observed problem.

</details>
