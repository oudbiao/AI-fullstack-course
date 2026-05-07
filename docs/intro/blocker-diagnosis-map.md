---
sidebar_position: 17
title: "Learning Blocker Diagnosis Map"
description: "A compact blocker map that helps learners locate the stuck layer, return to the right chapter, and record repair evidence."
keywords: [learning blockers, Debug map, AI learning diagnosis, RAG troubleshooting, Agent troubleshooting]
---

# Learning Blocker Diagnosis Map

![Learning blocker diagnosis flowchart](/img/course/intro-blocker-diagnosis-flow-en.png)

When you get stuck, do not add more materials first. Locate the blocker layer, run the smallest test, then record the fix.

## Fast Diagnosis Table

| Symptom | Likely layer | Return first | Smallest test | Evidence |
| --- | --- | --- | --- | --- |
| Commands do not run | Tools/environment | Chapter 1, environment setup | Open a new terminal and rerun from the README | command log |
| Python cannot read files | Code/project structure | Chapter 2 | Print `Path.cwd()` and a tiny file read/write | input/output file |
| Data conclusions feel wrong | Data quality | Chapter 3 | Print columns, missing values, duplicates | data dictionary |
| Metrics or loss make no sense | Math/model evaluation | Chapters 4-6 | Recreate one metric with 5 samples | metric note |
| LLM JSON is unstable | Prompt/schema | Chapter 7 | Test 10 fixed inputs with a parser | prompt version table |
| RAG answer has weak sources | Retrieval/citation | Chapter 8 | Print retrieval results before generation | retrieval logs |
| Agent loops or overreaches | Tools/permissions | Chapter 9 | Limit to 3 steps and save trace | `agent_traces.jsonl` |
| Project works only locally | Delivery/deployment | Delivery standards, deployment chapters | Clone clean and follow README | clean-run log |
| You cannot explain the project | Portfolio story | Portfolio checklist | Add problem, input/output, evaluation, failure | README update |

## Four Questions Before You Search

1. What command or input produced the problem?
2. What did you expect?
3. What actually happened?
4. Can you reproduce it with a smaller example?

## Blocker Note Template

```md
## Blocker

Layer:
Command or input:
Expected:
Actual:
Smallest reproduction:
Fix:
Regression check:
```

Smooth learning does not mean never failing. It means each failure becomes easier to locate the next time.
