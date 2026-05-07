---
sidebar_position: 16
title: "Failure and Evaluation Index"
description: "A compact index for deciding what failure sample, evaluation set, log, or review file to keep for each AI project type."
keywords: [failure case index, evaluation template, test cases, AI project evaluation, portfolio]
---

# Failure and Evaluation Index

![AI project debug index map](/img/course/appendix-quick-ref-debug-index-map-en.png)

Successful screenshots are not enough. A good AI project keeps failures that can be reproduced and evaluation cases that can be rerun.

## 1. Map the Failure First

| Symptom | Likely layer | Evidence to keep |
|---|---|---|
| Command, import, or path error | Environment or Python | Command, full error, version info |
| Wrong chart or conclusion | Data | Data sample, cleaning note, chart before/after |
| Suspiciously high model score | ML evaluation | Split rule, baseline, leakage check |
| Loss does not improve | Deep learning | Config, curve, tensor shape notes |
| JSON fields drift | Prompt | Prompt version, fixed test input, output diff |
| RAG cites wrong source | Retrieval or citation | chunks, top-k logs, citation comparison |
| Agent chooses wrong tool | Tool schema or planning | trace, tool input/output, stop condition |
| Works locally but not online | Deployment | env vars, logs, startup command |

## 2. Minimum Files

```text
reports/
├── failure_cases.md
├── improvement_record.md
└── demo_notes.md

evals/
├── eval_questions.csv
├── prompt_cases.csv
└── agent_tasks.jsonl

logs/
├── llm_calls.jsonl
├── retrieval_logs.jsonl
└── agent_traces.jsonl
```

## 3. Failure Sample Format

```md
## Failure title

- Input:
- Expected:
- Actual:
- Layer:
- Evidence:
- Likely cause:
- Fix:
- Regression test:
```

Keep failure notes short, but make them reproducible. A failure you can replay is engineering evidence, not embarrassment.
