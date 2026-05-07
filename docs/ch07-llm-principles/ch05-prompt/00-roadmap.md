---
title: "7.5.1 Prompt Engineering Roadmap: Brief, Output, Evaluation"
sidebar_position: 0
description: "A short hands-on roadmap for Prompt Engineering: turn vague requests into reusable task briefs, structured outputs, and repeatable evaluations."
keywords: [Prompt Guide, Prompt Engineering, Structured Output, Prompt Evaluation]
---

# 7.5.1 Prompt Engineering Roadmap: Brief, Output, Evaluation

Prompt engineering is the interface between your application and the model. The goal is not to write a clever sentence; the goal is to make one model call predictable, parseable, testable, and easy to improve.

## 7.5.1.1 See the Prompt Loop First

![Prompt engineering chapter relationship diagram](/img/course/ch07-prompt-chapter-flow-en.png)

![Prompt three-layer task specification diagram](/img/course/ch07-prompt-spec-three-layer-map-en.png)

![Prompt iteration test closed loop diagram](/img/course/ch07-prompt-iteration-loop-en.png)

Use this chapter when the model already has the general ability, but the result is vague, unstable, in the wrong format, or hard to evaluate.

## 7.5.1.2 Run a Prompt Contract Check

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

If `ready` is `False`, fix the prompt brief before you try more examples. A vague prompt produces vague debugging.

## 7.5.1.3 Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | Prompt basics | Rewrite one vague request into task, context, format, constraints |
| 2 | Advanced prompting | Add examples, steps, role, and boundary notes only when they help |
| 3 | Structured output | Make JSON, table, or Markdown output that another program can parse |
| 4 | Prompt practice | Compare prompt versions on the same fixed inputs |
| 5 | Evaluation lab | Record pass rate, failure type, and the next prompt change |

## 7.5.1.4 Pass Check

You pass this chapter when you can keep the input set fixed, change one prompt layer at a time, and explain why the new version is better with evidence instead of a feeling.

The exit mini project is a course-content extraction prompt: input one course document, output chapter topic, learning goals, prerequisites, key terms, practice suggestions, and risk notes as JSON or a Markdown table.
