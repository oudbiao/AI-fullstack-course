---
title: "7.5.1 Prompt Engineering Roadmap: Brief, Output, Evaluation"
description: "A short hands-on roadmap for Prompt Engineering: turn vague requests into reusable task briefs, structured outputs, and repeatable evaluations."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Prompt Guide, Prompt Engineering, Structured Output, Prompt Evaluation"
---
Prompt engineering is the interface between your application and the model. The goal is not to write a clever sentence; the goal is to make one model call predictable, parseable, testable, and easy to improve.

## See the Prompt Loop First

![Prompt engineering chapter relationship diagram](/img/course/ch07-prompt-chapter-flow-en.webp)

![Prompt three-layer task specification diagram](/img/course/ch07-prompt-spec-three-layer-map-en.webp)

![Prompt iteration test closed loop diagram](/img/course/ch07-prompt-iteration-loop-en.webp)

Use this chapter when the model already has the general ability, but the result is vague, unstable, in the wrong format, or hard to evaluate.

## Run a Prompt Contract Check

Before calling any LLM, describe the prompt as a contract: task, context, output format, and constraints. This tiny script checks whether the contract is complete enough to test.

```python
prompt_contract = {
    "task": "Extract chapter metadata",
    "context": "One course markdown file",
    "output_format": ["chapter", "goals", "prerequisites", "risks"],
    "constraints": ["return JSON only", "mark missing facts as null"],
}

required = ["task", "context", "output_format", "constraints"]
missing = [field for field in required if not prompt_contract.get(field)]

print("ready:", not missing)
print("fields:", ", ".join(required))
print("test_case_count:", 3)
```

Expected output:

```text
ready: True
fields: task, context, output_format, constraints
test_case_count: 3
```

![Prompt contract check result map](/img/course/ch07-prompt-contract-check-result-map-en.webp)

If `ready` is `False`, fix the prompt brief before you try more examples. A vague prompt produces vague debugging.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Prompt basics | Rewrite one vague request into task, context, format, constraints |
| 2 | Advanced prompting | Add examples, steps, role, and boundary notes only when they help |
| 3 | Structured output | Make JSON, table, or Markdown output that another program can parse |
| 4 | Prompt practice | Compare prompt versions on the same fixed inputs |
| 5 | Evaluation lab | Record pass rate, failure type, and the next prompt change |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
prompt_contract: task, context, constraints, output format
fixed_cases: same inputs used across prompt versions
schema_check: structured output validated by parser
failure_note: prompt failure grouped by cause
bridge: Chapter 8 adds retrieved context to this loop
```

## Pass Check

You pass this chapter when you can keep the input set fixed, change one prompt layer at a time, and explain why the new version is better with evidence instead of a feeling.

The exit mini project is a course-content extraction prompt: input one course document, output chapter topic, learning goals, prerequisites, key terms, practice suggestions, and risk notes as JSON or a Markdown table.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer explains how tokens, context, attention, prompts, and generation behavior connect in one request-response path.
2. The evidence should include at least one reproducible prompt or structured-output test, plus notes on why the output passed or failed.
3. A good self-check separates prompt design, RAG, fine-tuning, and alignment: use the lightest method that fixes the observed problem.

</details>
