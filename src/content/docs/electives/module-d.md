---
title: "E.D AI Safety and Red Team Testing"
description: "Run a tiny AI red-team loop: define surfaces, record failures, apply a guardrail, and keep regression cases."
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AI safety, red teaming, threat model, eval, jailbreak, prompt injection, guardrails"
---
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
threat_model: prompt injection, data leak, tool misuse, unsafe output, or model abuse
control: validation, permission, sandbox, audit, red-team test, or incident response
test_case: one attack or failure sample and expected safe behavior
failure_check: trusting model text, missing logs, broad permissions, or no regression tests
Expected_output: security checklist plus one reproducible red-team case
```

## Pass Check

You pass this elective when you can keep a red-team case file, explain one failed surface, propose one guardrail, and rerun the case after the fix.

<details>
<summary>Check reasoning and explanation</summary>

A passing answer should name one surface, one failure, one guardrail, and the rerun result. For example: “The tool surface failed because the model executed without confirmation. The guardrail requires explicit user approval before external actions. After the fix, the same case returns `ask_confirmation`.”

The key is repeatability. A red-team note is useful only when the failed case becomes a regression case that future changes must pass.

</details>
