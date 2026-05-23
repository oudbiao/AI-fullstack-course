---
title: "7.0 Learning Checklist: LLM Principles, Prompt, and Fine-tuning"
description: "A compact checklist for Chapter 7: LLM principles, Prompt experiments, structured output, RAG/fine-tuning decisions, and portfolio evidence."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "LLM study checklist, Prompt evaluation, Transformer, fine-tuning, RLHF"
---
Use this page as a printable checklist. If you need the full explanation, return to the [Chapter 7 entry page](/ch07-llm-principles/).

![LLM study guide evolution path](/img/course/ch07-study-guide-evolution-line-en.webp)

## Two-Hour First Pass

| Time box | Do this | Stop when you can say |
|---|---|---|
| 20 min | Read the token-to-answer picture on the entry page | "Text becomes tokens, vectors, context, then next-token prediction." |
| 25 min | Skim 7.1 and run one tokenizer example | "Token count affects cost and context limits." |
| 25 min | Skim 7.2 and the LLM history page | "Scale, data, Transformer, and alignment changed what models can do." |
| 30 min | Run the prompt testing script from the entry page | "I can compare prompt versions with fixed cases." |
| 20 min | Read the solution-choice table | "I should not fine-tune before checking Prompt, RAG, tools, and validation." |

## Required Evidence

| Evidence | Minimum version |
|---|---|
| `prompts/` | Three prompt versions for one task |
| `prompt_eval_cases.csv` | At least five fixed inputs and a simple score column |
| `structured_output_schema.json` | Required fields and allowed value types |
| `failure_cases.md` | At least three failed outputs and the likely cause |
| `mini_gpt2_train_log.txt` | Training log from [7.4.5 Rent a GPU and Run a Hand-Built GPT-2](/ch07-llm-principles/ch04-pretraining/04-rent-gpu-gpt2-lab/) |
| `llm_stage_workshop_output.txt` | Output from [7.8.4 Hands-on: Full Chapter 7 Workshop](/ch07-llm-principles/ch08-projects/03-stage-hands-on-workshop/) |
| `README.md` | How to run, what passed, what failed, what to try next |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
prompt_versions: at least three versions for one task
eval_cases: fixed inputs with scores and failure notes
schema_check: structured output is parsed and validated
method_choice: Prompt/RAG/fine-tuning/tools decision is written down
gpt2_record: mini GPT-2 training log, environment info, and sample output
exit_proof: workshop output plus README notes
```

## Quality Gates

| Gate | Pass condition |
|---|---|
| Prompt comparison | Same cases, one changed variable, saved outputs and scores. |
| Structured output | Parser rejects missing fields or wrong types. |
| Failure analysis | Each failure has a likely cause: instruction, input, schema, missing knowledge, or safety. |
| Method choice | Decision table explains why Prompt, RAG, fine-tuning, tools, or Agent comes first. |
| Hand-built GPT-2 | Run mini GPT-2 and identify embedding, attention, loss, and generate in the code. |

Expected result: your Chapter 7 folder contains prompt versions, fixed eval cases, parser/schema checks, failure notes, a mini GPT-2 training log, workshop output, and a README that explains the method choice.

## Exit Questions

- Can you explain token, embedding, attention, context window, pretraining, Prompt, fine-tuning, and alignment without copying definitions?
- Can you change one prompt variable at a time and compare results with the same input cases?
- Can you validate JSON output instead of trusting text that only looks like JSON?
- Can you explain when missing information calls for RAG instead of a longer Prompt?
- Can you explain when repeated behavior adaptation might justify fine-tuning?
- Can you open or rent a GPU notebook, run mini GPT-2, and save loss plus generated text?

<details>
<summary>Check reasoning and explanation</summary>

1. Treat each term as part of one flow: token and embedding are the representation layer, attention routes context, the context window limits what can be seen at once, pretraining builds the base model, Prompt steers the run, fine-tuning changes behavior with data, and alignment keeps outputs useful and safe.
2. Keep the same cases, change only one prompt variable, and save both the outputs and the score so the comparison is reproducible instead of anecdotal.
3. Use a schema or parser to validate structure, required fields, and types. If parsing fails, reject the output instead of reading it as if it were correct.
4. Use RAG when the answer depends on fresh, private, or citable facts from documents rather than what the model may remember.
5. Fine-tuning becomes worth considering when the same behavior keeps showing up across many high-quality examples and Prompt plus validation still is not enough.

</details>

If the answer is yes, move to Chapter 8. Chapter 8 will connect these ideas to real LLM applications and RAG systems.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
prompt_versions: at least three versions for one task
eval_cases: fixed inputs with scores and failure notes
schema_check: structured output is parsed and validated
method_choice: Prompt/RAG/fine-tuning/tools decision is written down
gpt2_record: mini GPT-2 training log, environment info, and sample output
exit_proof: workshop output plus README notes
```
