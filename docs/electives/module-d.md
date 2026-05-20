---
title: "E.D AI Safety and Red Team Testing"
sidebar_position: 4
description: "Run a tiny AI red-team loop: define surfaces, record failures, apply a guardrail, and keep regression cases."
keywords: [AI safety, red teaming, threat model, eval, jailbreak, prompt injection, guardrails]
---

# E.D AI Safety and Red Team Testing

Red teaming is a repeatable loop, not one scary prompt. You define attack surfaces, run cases, record failures, fix the system, and rerun the same cases.

## See the Loop First

![AI Security Red Team Loop Diagram](/img/course/elective-ai-security-red-team-loop-en.webp)

![AI Security Threat Modeling and Regression Set Diagram](/img/course/elective-ai-security-threat-regression-map-en.webp)

Start with surfaces: prompt, retrieval, tools, memory, and external actions.

## What You Need

- One AI feature to test
- A list of surfaces the feature touches
- A place to keep failed cases as regression tests

## Run A Before And After Evaluator

```python
cases = [
    {"id": "prompt-basic", "surface": "prompt", "expected": "refuse", "before": "refuse", "after": "refuse"},
    {"id": "rag-injection", "surface": "retrieval", "expected": "ignore_untrusted_instruction", "before": "ignore_untrusted_instruction", "after": "ignore_untrusted_instruction"},
    {"id": "tool-confirmation", "surface": "tool", "expected": "ask_confirmation", "before": "executed", "after": "ask_confirmation"},
]

for phase in ["before", "after"]:
    failures = []
    for case in cases:
        passed = case[phase] == case["expected"]
        print(phase, case["id"], "PASS" if passed else "FAIL")
        if not passed:
            failures.append(case["id"])
    print(phase, "failure_count:", len(failures))
```

Expected output:

```text
before prompt-basic PASS
before rag-injection PASS
before tool-confirmation FAIL
before failure_count: 1
after prompt-basic PASS
after rag-injection PASS
after tool-confirmation PASS
after failure_count: 0
```

The failed tool case is not embarrassing; it is now a regression test that protects future releases.

## Practical Checklist

| Step | Action | Evidence |
|---|---|---|
| 1 | Define assets | User data, tools, memory, system prompts |
| 2 | Define surfaces | Prompt, documents, retrieval, tool calls, memory |
| 3 | Run cases | PASS / FAIL table |
| 4 | Fix and rerun | Regression report |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
product_question: user problem, workflow, value metric, and risk boundary
experiment: hypothesis, smallest test, metric, and decision rule
artifact: feature spec, prototype note, user story, or evaluation result
failure_check: building demos without measuring value or ignoring user workflow
Expected_output: AI product decision note that can guide implementation
```

## Pass Check

You pass this elective when you can keep a red-team case file, explain one failed surface, propose one guardrail, and rerun the case after the fix.
