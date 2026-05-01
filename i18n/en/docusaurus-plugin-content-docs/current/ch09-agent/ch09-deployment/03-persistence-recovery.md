---
title: "9.4 Persistence and Recovery"
sidebar_position: 51
description: "Starting from state snapshots, event logs, idempotent execution, and crash recovery, understand why Agent tasks must be recoverable."
keywords: [persistence, recovery, checkpoint, event log, idempotency, resume, deployment]
---

# Persistence and Recovery

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

## 1. Why do Agents especially need recoverability?

### 1.1 Because many tasks are not completed instantly

For example:

- Research report generation
- Multi-tool approval workflows
- Multi-round background crawling and organization

These tasks often span:

- Multiple calls
- Multiple steps
- Longer time windows

### 1.2 What happens without recovery capability?

- If the task is interrupted, everything starts over
- Already executed actions may be repeated
- Users cannot tell the current progress

### 1.3 An analogy

An Agent without persistence is like a workstation that “forgets everything when the power goes out.”  
A production-ready system is more like an IDE with auto-save and restore points.

---

## 2. What exactly is being persisted?

### 2.1 The most important thing is task state

For example:

- Which step execution is currently on
- Which steps have been completed
- What the intermediate results are

### 2.2 Next comes the event log

The event log answers:

- What exactly happened before?

For example:

- Which tool was called
- What response was received
- Which step failed

### 2.3 The difference between snapshots and logs

You can remember it like this:

- `checkpoint / snapshot`: a compressed slice of the current state
- `event log`: the stream of events that happened along the way

In real engineering, these two are often used together.

---

## 3. First, run a minimal recovery workflow

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

print("checkpoint after crash:", runner.last_checkpoint)

final_state = runner.run()
print("\nrestored final state:")
print(final_state)

print("\nevent log:")
for event in runner.event_log:
    print(event["type"], event["payload"])
```

### 3.1 What is the most important thing to learn from this example?

It connects the three most important pieces in the recovery path:

1. Write a checkpoint after each step
2. Keep an event log when errors happen
3. After restart, continue from the last checkpoint

### 3.2 Why not write the checkpoint only at the end of the task?

Because if the task crashes midway,  
you still cannot recover anything.

So for long tasks, a more practical choice is:

- Step-level checkpoints

### 3.3 Why is the event log important?

A checkpoint can only tell you “what the current state is,”  
but it cannot fully explain:

- Why it became that state
- Where the failure happened

Logs help with postmortems and debugging.

![Agent Checkpoint, Event Log, and Recovery Diagram](/img/course/ch09-agent-persistence-checkpoint-eventlog-map.png)

:::tip Reading the diagram
This diagram splits recovery into two paths: checkpoint handles “where to recover to now,” while event log handles “what happened before.” When long tasks go live, it is best to use both together rather than storing only the final result.
:::

---

## 4. Why is idempotency the core of the recovery path?

### 4.1 What does idempotent mean?

Idempotency can be roughly understood as:

- Repeating the same action multiple times still produces the same result

### 4.2 Why is it especially needed during recovery?

If the system crashes before “writing the report,” after restarting you may not know:

- Whether this step was actually completed

If the action is not idempotent, it can lead to:

- Duplicate writes
- Duplicate charges
- Duplicate messages

### 4.3 A simplified example

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

This is the simplest idea behind idempotency protection.

---

## 5. What do people most often forget when designing recovery?

### 5.1 Storing only the “result,” not the “progress”

If you only store the summary and do not store:

- Which step you are currently on

then recovery still remains difficult.

### 5.2 Storing only checkpoints, not logs

This allows recovery, but makes it hard to investigate why the failure happened.

### 5.3 External side effects have no idempotency key

This makes recovery risky,  
because the system cannot tell whether replaying will create duplicate side effects.

---

## 6. How is this usually done in real systems?

### 6.1 State table

Store:

- Task id
- Current step
- Current state snapshot
- Update time

### 6.2 Event table

Store:

- Event type
- Time
- Input/output summary
- Error information

### 6.3 Recovery service

Responsible for:

- Scanning unfinished tasks on restart
- Loading the latest checkpoint
- Continuing from a safe point

---

## 7. Most common misconceptions

### 7.1 Misconception 1: If there is a database, then it is “recoverable”

Not true.  
The key is whether you have stored:

- Enough information to recover

### 7.2 Misconception 2: Recovery just means “run it again”

Running it again often causes duplicate side effects.  
Recovery is not redoing; it is continuing statefully.

### 7.3 Misconception 3: Only very long tasks need recovery

As long as a task includes external side effects or multi-step execution,  
recovery capability is important.

---

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
