---
title: "1.4.2 AI Coding Agent Workflow"
description: "Learn how to turn a vague coding request into a traceable AI coding-agent run with scope, permissions, tests, evidence, and human review."
sidebar:
  order: 2
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AI coding agent, Codex, agentic coding workflow, software engineering with AI"
---
![AI coding agent workflow whiteboard](/img/course/ch01-ai-coding-agent-workflow-whiteboard-en.webp)

AI coding agents are no longer only autocomplete tools. Modern tools such as [Codex](https://openai.com/index/codex-now-generally-available/) and [Google Antigravity](https://antigravity.google/blog/introducing-google-antigravity) can inspect a repository, edit files, run tests, and leave a task trail. That power is useful only when you give the agent a narrow job, a permission boundary, and a way to prove what changed.

This lesson teaches the workflow before the magic. You should finish with a small run card that another developer can review.

## Why This Appeared

Older AI coding help mostly answered "how do I write this function?" Agentic coding answers a different question:

> Can a model move through a real repository, change code, verify it, and explain the result without hiding the risk?

That shift happened because model reasoning, tool calling, long context, terminal access, and code-review interfaces became good enough to form a loop:

1. Read the repo and constraints.
2. Plan a small edit.
3. Patch files.
4. Run tests or checks.
5. Summarize evidence.
6. Ask for human review when risk is high.

The problem it solves is not "write code faster." The deeper problem is reducing handoff cost: the agent can collect context, make a scoped change, and package proof for the human.

## What Problem It Solves

| Problem in coding work | Agent role | Human role | Evidence to keep |
|---|---|---|---|
| Large repo is hard to navigate | Search files, map entry points, identify likely owners | Confirm the requested scope | Search notes and touched files |
| Small bug requires many mechanical edits | Apply consistent edits and formatters | Check intent and product behavior | Diff, tests, screenshots |
| Tests fail for unclear reasons | Read logs, isolate failing layer, propose repair | Decide whether repair is in scope | Failing command and fixed command |
| Refactor risk is hidden | Produce a risk card before editing | Approve, narrow, or reject | Risk level and rollback note |
| Review takes too long | Summarize why each change exists | Review behavior and edge cases | Commit message and QA notes |

## Decision Table

Use this table before giving the agent a task.

| Situation | Good agent task | Not a good first task | Required gate |
|---|---|---|---|
| One failing unit test | "Fix this test failure and explain the root cause" | "Rewrite the whole module" | Run the failing test and one nearby test |
| UI copy or layout issue | "Adjust this section and verify screenshot" | "Redesign the whole app" | Browser screenshot |
| New lesson page | "Add a page matching existing template" | "Invent a new curriculum structure" | Link check and course QA |
| Security or data deletion | "Inspect and propose a patch plan" | "Run destructive cleanup" | Human approval |
| Dependency upgrade | "Assess breaking changes and update one package" | "Upgrade everything" | Lockfile diff and build |

## Runnable Lab: Build an Agent Run Card

Create `agent_run_card.py` and run it with Python 3.10 or later.

```python
import json
from pathlib import Path


task = {
    "request": "fix a broken course sidebar link",
    "files_likely_touched": ["src/content/docs", "astro.config.mjs"],
    "can_run_tests": True,
    "touches_user_data": False,
    "changes_public_behavior": True,
}


def classify_risk(info):
    if info["touches_user_data"]:
        return "high"
    if info["changes_public_behavior"]:
        return "medium"
    return "low"


def choose_gates(info):
    gates = ["read surrounding files", "make minimal patch", "record diff"]
    if info["can_run_tests"]:
        gates.append("run relevant QA command")
    if info["changes_public_behavior"]:
        gates.append("capture before/after behavior")
    return gates


run_card = {
    "task": task["request"],
    "agent_scope": "one narrow bug or content fix",
    "risk": classify_risk(task),
    "permissions": {
        "read": True,
        "edit": True,
        "network": False,
        "destructive_commands": False,
    },
    "gates": choose_gates(task),
    "evidence_file": "agent_evidence.md",
}

Path("agent_run_card.json").write_text(json.dumps(run_card, indent=2), encoding="utf-8")
print(json.dumps(run_card, indent=2))
```

Expected output:

```text
{
  "task": "fix a broken course sidebar link",
  "agent_scope": "one narrow bug or content fix",
  "risk": "medium",
  "permissions": {
    "read": true,
    "edit": true,
    "network": false,
    "destructive_commands": false
  },
  "gates": [
    "read surrounding files",
    "make minimal patch",
    "record diff",
    "run relevant QA command",
    "capture before/after behavior"
  ],
  "evidence_file": "agent_evidence.md"
}
```

## Read the Code Line by Line

`task` is the input contract. It says what the human wants, what files are likely involved, and which risks exist.

`classify_risk()` is the permission gate. The agent should not treat a content typo and a user-data migration as the same type of work.

`choose_gates()` turns the task into proof steps. It is the difference between "I changed it" and "I changed it, checked it, and can show what happened."

`run_card` is the handoff artifact. In a real repository, attach it to the PR, commit message, or task note.

## Mini Exercise

Change the request to one of these tasks and rerun the script:

1. "update the homepage meta description"
2. "delete old user accounts"
3. "add a new API endpoint"

For each run, answer:

| Question | What to decide |
|---|---|
| Is the risk low, medium, or high? | Explain the reason, not only the label |
| What extra gate is needed? | Test, screenshot, security review, migration backup, or human approval |
| What evidence would convince a reviewer? | Diff, log, screenshot, request/response, or rollback note |

## Evidence to Keep

Before you finish any AI coding task, leave this minimum packet:

```text
request: what the human asked for
scope: files or behavior intentionally touched
risk: low, medium, or high
commands: checks that were run
result: pass, fail, or not run with reason
diff_summary: what changed and why
rollback: how to undo or where the commit is
```

## Small Summary

AI coding agents are most useful when they are treated as junior engineers with fast hands and a strict notebook. Give them a narrow task, make permissions explicit, run checks, and require evidence.

<details>
<summary>Check reasoning and explanation</summary>

You pass this lesson when you can turn a vague request into a scoped run card, name the risk, choose the verification gates, and explain what evidence a reviewer should inspect.

</details>
