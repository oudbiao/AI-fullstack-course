---
sidebar_position: 15
title: "Common Errors and Troubleshooting Index"
description: "A compact troubleshooting index for environment, Python, data, model, RAG, Agent, and deployment issues."
keywords: [troubleshooting, common errors, Debug, AI learning blockers, RAG errors, Agent errors]
---

# Common Errors and Troubleshooting Index

![Troubleshooting rescue map](/img/course/appendix-troubleshooting-rescue-map-en.png)

Most learning blockers are not caused by “not understanding AI.” They are caused by directory, environment, dependency, data format, API, retrieval, tool, or deployment mismatches. Debug in order.

## 1. First Move by Symptom

| Symptom | Check first | Minimal proof |
|---|---|---|
| Command fails | Current directory and whether the command exists | `pwd`, `which <command>`, project scripts |
| Python import fails | Interpreter and package installation location | `python -m pip show <package>` |
| File cannot be read | Relative path, filename case, encoding | Print `pwd`, then open one tiny file |
| Output JSON is unstable | Prompt schema and validation | Ask for one minimal JSON object |
| RAG answer is wrong | Retrieved chunks before the model call | Print top-k retrieval results |
| Agent loops | Step limit, stop condition, trace | Run with max 3 steps |
| Deployment fails | Environment variables, startup command, file paths | Run the README in a clean environment |

## 2. The Debug Loop

1. Save the full error and the exact command.
2. Confirm directory, runtime, dependency versions, and configuration.
3. Reproduce with the smallest input.
4. Locate the failing layer: environment, code, data, model, retrieval, tool, or deployment.
5. Fix it, then write the prevention note back into the README or script.

Do not read only the last line of a stack trace. The useful clue is often the actual path, interpreter, model name, field name, or tool parameter printed in the middle.

## 3. Layer Checklist

| Layer | Fast question |
|---|---|
| Environment | Am I in the right folder and using the right runtime? |
| Dependencies | Did I install into the same Python or Node environment that runs the project? |
| Data | Are paths, columns, encodings, and sample rows correct? |
| Model | Are input shape, labels, model name, quota, and timeout correct? |
| RAG | Are chunks, metadata, top-k, rerank, and citations visible in logs? |
| Agent | Are tool schema, step limit, permissions, and trace recorded? |
| Deployment | Are env vars, ports, startup command, and logs visible online? |

## 4. Troubleshooting Record Template

```md
## Issue title

### Symptom
Command I ran and error I saw.

### Key error
Paste the important lines, not only the last line.

### Reproduction
1. Directory:
2. Command:
3. Input or config:

### Checks
Environment, path, dependency, data, parameter, or log I checked.

### Fix
What changed and why it worked.

### Prevention
What to add to README, validation script, log field, or test case.
```

Every solved error should leave better documentation or a safer command for the next run.
