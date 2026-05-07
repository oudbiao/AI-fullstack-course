---
title: "E.D AI Safety and Red Team Testing"
sidebar_position: 4
description: "A concise hands-on guide to AI red teaming: model assets, attack surfaces, failure categories, fixes, and regression checks."
keywords: [AI safety, red teaming, threat model, eval, jailbreak, prompt injection, guardrails]
---

# E.D AI Safety and Red Team Testing

Red teaming is not “try a scary prompt once.” It is a loop: model the threat, run cases, record failures, fix the system, and keep the failure as a regression test.

## See the Loop First

![AI Security Red Team Loop Diagram](/img/course/elective-ai-security-red-team-loop-en.png)

![AI Security Threat Modeling and Regression Set Diagram](/img/course/elective-ai-security-threat-regression-map-en.png)

Start with surfaces, not attack names: prompt, retrieval, tools, memory, and external actions.

## Run a Minimal Red-Team Evaluator

```python
cases = [
    {"surface": "prompt", "expected": "refuse", "observed": "refuse"},
    {"surface": "retrieval", "expected": "ignore_untrusted_instruction", "observed": "ignore_untrusted_instruction"},
    {"surface": "tool", "expected": "ask_confirmation", "observed": "executed"},
]

failures = []
for case in cases:
    passed = case["expected"] == case["observed"]
    print(case["surface"], "PASS" if passed else "FAIL")
    if not passed:
        failures.append(case["surface"])

print("failure_count:", len(failures))
print("regression_cases:", failures)
```

Expected output:

```text
prompt PASS
retrieval PASS
tool FAIL
failure_count: 1
regression_cases: ['tool']
```

The point is not to hide the failure. The point is to keep it, fix it, and rerun it.

## Practical Checklist

| Step | Action | Evidence |
|---|---|---|
| 1 | Define assets | User data, tools, memory, system instructions |
| 2 | Define attack surfaces | Prompt, documents, retrieval, tool calls, memory |
| 3 | Run cases | PASS / FAIL table |
| 4 | Fix and rerun | Regression report |

## Pass Check

You pass this elective when you can keep a red-team case file, explain one failed surface, propose a guardrail, and rerun the case after the fix.
