---
title: "9.9.4 Persistence and Recovery"
sidebar_position: 51
description: "Starting from state snapshots, event logs, idempotent execution, and crash recovery, understand why Agent tasks must be recoverable."
keywords: [persistence, recovery, checkpoint, event log, idempotency, resume, deployment]
---

# 9.9.4 Persistence and Recovery

:::tip Where this section fits
Once an Agent starts handling long tasks, multi-step workflows, or background jobs, one problem quickly becomes critical:

- What if it crashes halfway through?

If the system has no persistence and recovery capability, then:

- Work done earlier is wasted
- Repeated execution causes repeated side effects
- Users only see “the task disappeared”

So the core idea of this lesson is:

> **Let tasks continue statefully after failures and restarts, instead of starting from zero every time.**
:::

## Learning Objectives

- Understand the meaning of “persistence” and “recovery” in Agent tasks
- Learn to distinguish between state snapshots and event logs
- Implement a minimal checkpoint + recovery flow with a runnable example
- Understand why idempotency matters in the recovery path

---

## Why do Agents especially need recoverability?

### Because many tasks are not completed instantly

For example:

- Research report generation
- Multi-tool approval workflows
- Multi-round background crawling and organization

These tasks often span:

- Multiple calls
- Multiple steps
- Longer time windows

### What happens without recovery capability?

- If the task is interrupted, everything starts over
- Already executed actions may be repeated
- Users cannot tell the current progress

### An analogy

An Agent without persistence is like a workstation that “forgets everything when the power goes out.”
A production-ready system is more like an IDE with auto-save and restore points.

---

## What exactly is being persisted?

### The most important thing is task state

For example:

- Which step execution is currently on
- Which steps have been completed
- What the intermediate results are

### Next comes the event log

The event log answers:

- What exactly happened before?

For example:

- Which tool was called
- What response was received
- Which step failed

### The difference between snapshots and logs

You can remember it like this:

- `checkpoint / snapshot`: a compressed slice of the current state
- `event log`: the stream of events that happened along the way

In real engineering, these two are often used together.

---

## First, run a minimal recovery workflow

The example below simulates a three-step task:

1. Read materials
2. Generate a summary
3. Write the report

The system writes a checkpoint after each step.
If a failure occurs in step 2, it continues from the last checkpoint.

```python
import copy


TASK_PLAN = ["load_data", "summarize", "write_report"]


def execute_step(step, state):
    if step == "load_data":
        state["data"] = ["refund policy", "invoice policy", "address change policy"]
    elif step == "summarize":
        state["summary"] = ";".join(state["data"])
    elif step == "write_report":
        state["report"] = f"Final report: {state['summary']}"
    return state


class WorkflowRunner:
    def __init__(self):
        self.event_log = []
        self.last_checkpoint = None

    def checkpoint(self, state):
        self.last_checkpoint = copy.deepcopy(state)

    def log_event(self, event_type, payload):
        self.event_log.append({"type": event_type, "payload": copy.deepcopy(payload)})

    def run(self, fail_on_step=None):
        state = self.last_checkpoint or {"current_index": 0, "completed_steps": []}

        while state["current_index"] < len(TASK_PLAN):
            step = TASK_PLAN[state["current_index"]]
            self.log_event("step_started", {"step": step, "state": state})

            if step == fail_on_step:
                self.log_event("step_failed", {"step": step})
                raise RuntimeError(f"crash_on_{step}")

            state = execute_step(step, state)
            state["completed_steps"].append(step)
            state["current_index"] += 1

            self.checkpoint(state)
            self.log_event("step_completed", {"step": step, "state": state})

        return state


runner = WorkflowRunner()

try:
    runner.run(fail_on_step="summarize")
except RuntimeError as e:
    print("first run crashed:", e)

print("checkpoint after crash:", {
    "current_index": runner.last_checkpoint["current_index"],
    "completed_steps": runner.last_checkpoint["completed_steps"],
})

final_state = runner.run()
print("\nrestored final state:")
print({
    "completed_steps": final_state["completed_steps"],
    "report": final_state["report"],
})

print("\nevent types:")
print([event["type"] for event in runner.event_log])
```

Expected output:

```text
first run crashed: crash_on_summarize
checkpoint after crash: {'current_index': 1, 'completed_steps': ['load_data']}

restored final state:
{'completed_steps': ['load_data', 'summarize', 'write_report'], 'report': 'Final report: refund policy;invoice policy;address change policy'}

event types:
['step_started', 'step_completed', 'step_started', 'step_failed', 'step_started', 'step_completed', 'step_started', 'step_completed']
```

![Agent Checkpoint Recovery Result Map](/img/course/ch09-agent-checkpoint-recovery-result-map-en.webp)

### What is the most important thing to learn from this example?

It connects the three most important pieces in the recovery path:

1. Write a checkpoint after each step
2. Keep an event log when errors happen
3. After restart, continue from the last checkpoint

### Why not write the checkpoint only at the end of the task?

Because if the task crashes midway,
you still cannot recover anything.

So for long tasks, a more practical choice is:

- Step-level checkpoints

### Why is the event log important?

A checkpoint can only tell you “what the current state is,”
but it cannot fully explain:

- Why it became that state
- Where the failure happened

Logs help with postmortems and debugging.

![Agent Checkpoint, Event Log, and Recovery Diagram](/img/course/ch09-agent-persistence-checkpoint-eventlog-map-en.webp)

:::tip Reading the diagram
This diagram splits recovery into two paths: checkpoint handles “where to recover to now,” while event log handles “what happened before.” When long tasks go live, it is best to use both together rather than storing only the final result.
:::

---

## Why is idempotency the core of the recovery path?

### What does idempotent mean?

Idempotency can be roughly understood as:

- Repeating the same action multiple times still produces the same result

### Why is it especially needed during recovery?

If the system crashes before “writing the report,” after restarting you may not know:

- Whether this step was actually completed

If the action is not idempotent, it can lead to:

- Duplicate writes
- Duplicate charges
- Duplicate messages

### A simplified example

```python
processed = set()


def send_email_once(task_id, address):
    if task_id in processed:
        return {"ok": True, "status": "skipped_duplicate"}
    processed.add(task_id)
    return {"ok": True, "status": f"sent_to:{address}"}


print(send_email_once("task-1", "a@example.com"))
print(send_email_once("task-1", "a@example.com"))
```

Expected output:

```text
{'ok': True, 'status': 'sent_to:a@example.com'}
{'ok': True, 'status': 'skipped_duplicate'}
```

This is the simplest idea behind idempotency protection.

---

## What do people most often forget when designing recovery?

### Storing only the “result,” not the “progress”

If you only store the summary and do not store:

- Which step you are currently on

then recovery still remains difficult.

### Storing only checkpoints, not logs

This allows recovery, but makes it hard to investigate why the failure happened.

### External side effects have no idempotency key

This makes recovery risky,
because the system cannot tell whether replaying will create duplicate side effects.

---

## How is this usually done in real systems?

### State table

Store:

- Task id
- Current step
- Current state snapshot
- Update time

### Event table

Store:

- Event type
- Time
- Input/output summary
- Error information

### Recovery service

Responsible for:

- Scanning unfinished tasks on restart
- Loading the latest checkpoint
- Continuing from a safe point

---

## Most common misconceptions

### Misconception 1: If there is a database, then it is “recoverable”

Not true.
The key is whether you have stored:

- Enough information to recover

### Misconception 2: Recovery just means “run it again”

Running it again often causes duplicate side effects.
Recovery is not redoing; it is continuing statefully.

### Misconception 3: Only very long tasks need recovery

As long as a task includes external side effects or multi-step execution,
recovery capability is important.

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
runtime: queues, workers, state store, tool services, and model endpoint
persistence: checkpoints, event log, memory store, and recovery path
ops_signal: latency, cost, error rate, trace coverage, and saturation
failure_check: stuck run, duplicate action, partial failure, or runaway cost
recovery_action: resume, rollback, cancel, human handoff, or degrade gracefully
```

## Summary

The most important thing in this lesson is to build a production-grade judgment:

> **Persistence and recovery for an Agent is not simply writing the result to disk. It is about using checkpoints, event logs, and idempotency mechanisms to let a task continue safely after failures.**

Once this chain is designed clearly,
the system moves from a demo that “sometimes works” to a production system that can still continue after failures.

---

## Exercises

1. Add a `retry_count` field to the example to record the number of retries for each step.
2. Change `write_report` into an action with external side effects, then think about how idempotency should be implemented.
3. Why do we say checkpoint and event log are both indispensable in recovery?
4. If a task is especially long, would you choose a checkpoint after every step, or every few steps? Why?

<details>
<summary>Reference answers and explanation</summary>

1. `retry_count` should be stored per step, not only per whole run. That lets you see which step is unstable and prevents a retry storm from being hidden inside one final status.
2. If `write_report` has external side effects, make it idempotent with a stable operation id, existence checks, deduplication, and a record of whether the external write already succeeded.
3. Checkpoints give you the latest resumable state; event logs explain how the system reached that state. Recovery needs both the snapshot and the history of decisions and side effects.
4. For long tasks, checkpoint after important irreversible or expensive steps, and every few low-risk steps. Checkpointing every step is safest but may add storage and latency overhead.

</details>
