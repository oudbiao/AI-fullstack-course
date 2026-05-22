---
title: "9.2.3 Chain-of-Thought Reasoning Strategy"
description: "Starting from the core role of CoT, understand why 'split the steps first, then answer' can improve multi-step task performance, and when it helps or slows a system down."
sidebar:
  order: 6
head:
  - tag: meta
    attrs:
      name: keywords
      content: "chain of thought, CoT, reasoning trace, scratchpad, decomposition"
---
:::tip[Section focus]
If we summarize the previous section in one sentence, it is:

- Reasoning problems depend on intermediate states

Then this section asks:

> **How do we make the model more willing, and more consistent, to write intermediate states explicitly?**

The Chain-of-Thought strategy, or CoT, is built around one core idea:

- Don’t ask for the answer directly
- First let the model break the problem into steps, then give the conclusion
:::
## Learning Objectives

- Understand why Chain-of-Thought reasoning improves multi-step task performance
- Understand what kinds of tasks CoT is suitable for, and what kinds it is not
- Use runnable examples to understand the difference between “answer directly” and “answer step by step”
- Understand how to use structured reasoning in production instead of uncontrolled long-form text

---

## Why does “thinking through the steps first” help?

### Because many problems do not reach the end in one jump

For example, consider this problem:

- A support queue has 18 open tickets, 4 are duplicates that should be merged, and then 7 urgent tickets arrive. How many tickets need triage?

If the model generates the answer directly,
it may make very typical mistakes:

- Treat duplicate tickets as new tickets instead of removing them
- Forget the urgent tickets that arrive later
- Get the step order wrong

But if it first breaks the process into steps:

1. `18 - 4 = 14`
2. `14 + 7 = 21`

the final answer is usually more stable.

### The core of CoT is not “writing more,” but “exposing intermediate structure”

This point is especially important.
The real value of Chain-of-Thought is not:

- Making the output longer

But rather:

- Making local facts
- Intermediate variables
- Step dependencies

explicit.

### An analogy: scratch notes are not for looking serious

When operations teams review a complex case, they use scratch notes not to make the answer longer,
but to:

- prevent the mental state from being lost
- break a complex problem into smaller parts
- make checking easier

CoT plays a similar role for models.

---

## First, let’s compare “direct answer” and “chain reasoning”

The example below does not call an LLM,
but it clearly demonstrates:

- Why a “rough direct mapping” is easy to get wrong
- Why “break the steps first, then calculate” is more stable

```python
import re

problem = "A support queue has 18 open tickets, 4 are duplicates to merge, and then 7 urgent tickets arrive. How many tickets need triage?"


def bad_direct_answer(text):
    numbers = list(map(int, re.findall(r"\d+", text)))
    open_tickets, duplicates, urgent = numbers
    # Common mistake: treating duplicates as new tickets instead of removing them
    return open_tickets + duplicates + urgent


def chain_reason_answer(text):
    open_tickets, duplicates, urgent = map(int, re.findall(r"\d+", text))

    steps = []
    unique_tickets = open_tickets - duplicates
    steps.append(f"First remove duplicate tickets: {open_tickets} - {duplicates} = {unique_tickets}")

    final_count = unique_tickets + urgent
    steps.append(f"Then add urgent tickets: {unique_tickets} + {urgent} = {final_count}")

    return final_count, steps


print("problem:", problem)
print("bad direct answer:", bad_direct_answer(problem))

answer, steps = chain_reason_answer(problem)
print("\nchain reasoning steps:")
for step in steps:
    print("-", step)
print("final answer:", answer)
```

Expected output:

```text
problem: A support queue has 18 open tickets, 4 are duplicates to merge, and then 7 urgent tickets arrive. How many tickets need triage?
bad direct answer: 29

chain reasoning steps:
- First remove duplicate tickets: 18 - 4 = 14
- Then add urgent tickets: 14 + 7 = 21
final answer: 21
```

### What does this code show most clearly?

It shows that:

- Direct mapping can easily misinterpret the problem
- Explicitly breaking the steps makes errors easier to expose

For example:

- Does “4 duplicates” mean `+4` or `-4`?

As soon as you write that step out,
the mistake is much harder to hide.

### Why is CoT especially useful for math, logic, and planning tasks?

Because these tasks usually have:

- Clear intermediate variables
- Clear step order
- Clear local dependencies

That naturally matches chain reasoning.

### Why shouldn’t CoT be used for every task?

Because not every problem needs step-by-step reasoning.
For example:

- “What is the capital of France?”

This kind of question is more like retrieval, and does not need a long reasoning chain.

So CoT is not about “the more, the better” by default,
but rather:

- More valuable for multi-step problems

---

## How is CoT usually used in Agents?

### First decompose, then call tools

In many Agent tasks, CoT is not directly used for arithmetic,
but to decide:

- What to do first
- What to do next
- Which step needs a tool

For example:

1. Identify the problem type
2. Decide whether to check policy first or inventory first
3. After getting observations, organize the conclusion

### It can also become more structured “reasoning slots”

In production, we do not always need the model to output a long natural-language thought process.
Many systems instead switch to a shorter, more structured format, such as:

- `facts`
- `subtasks`
- `decision`
- `next_action`

This kind of structure is often easier to:

- validate
- log
- debug

### CoT is also often combined with self-checking

A very common enhancement is:

1. Reason first
2. Check key steps
3. Output the final answer

This can reduce some careless mistakes.

![Chain-of-Thought and self-check structure diagram](/img/course/ch09-cot-self-check-structure-map-en.webp)

:::tip[Reading note]
This diagram emphasizes “structured intermediate states,” not letting the model output endless long-form thinking. In production, reasoning is more often compressed into verifiable slots such as facts, subtasks, decision, and next_action.
:::
---

## When is CoT most helpful?

### Problems that require step-by-step decomposition

For example:

- Multi-step calculation
- Conditional filtering
- Combined decision-making
- Complex rule checking

### Problems that require explaining the process

For example:

- Why recommend this solution
- Why can’t this request be executed
- Why does this answer follow the rules

When the system must provide not only a conclusion but also a reason,
explicit intermediate steps are very valuable.

### Problems with high error cost

If a mistake in calculation or judgment can lead to serious consequences,
then explicit steps are usually more worthwhile.

---

## When might CoT actually hurt performance?

### Simple retrieval questions

If the problem itself does not require step-by-step reasoning,
forcing the model to output a long process usually just means:

- slower
- longer
- more expensive

### A reasoning chain that is too long can confuse itself

When the chain gets too long, the model may:

- Have correct early steps but drift later
- Repeat explanations
- Become inconsistent in intermediate states

In other words, longer CoT is not automatically better.

### It may not be suitable to expose the full chain to users

In many products, a more reasonable approach is:

- Keep the reasoning structure internal
- Show users only a concise explanation

Because what users usually need is:

- A clear conclusion
- Necessary reasons

Not a long scratchpad.

---

## A more practical structured CoT pattern

The following example shows a style that is more suitable for Agents:

- Do not output a large block of free-form text
- Instead, split it into fixed slots

```python
ticket = {
    "question": "The refund queue has 18 open tickets, 4 are duplicates to merge, and then 7 urgent tickets arrive. How many tickets need triage?",
    "policy": "Duplicate support tickets should be merged before triage.",
}


def structured_reasoning(ticket):
    facts = [
        "Duplicate support tickets should be merged before triage",
        "The queue starts with 18 open tickets, removes 4 duplicates, then receives 7 urgent tickets",
    ]
    calculation = ["18 - 4 = 14", "14 + 7 = 21"]
    decision = "The team should triage 21 tickets."

    return {
        "facts": facts,
        "calculation": calculation,
        "decision": decision,
    }


result = structured_reasoning(ticket)
print(result)
```

Expected output:

```text
{'facts': ['Duplicate support tickets should be merged before triage', 'The queue starts with 18 open tickets, removes 4 duplicates, then receives 7 urgent tickets'], 'calculation': ['18 - 4 = 14', '14 + 7 = 21'], 'decision': 'The team should triage 21 tickets.'}
```

The advantages of this format are:

- Easier to read
- Easier to test
- Easier to post-process

---

## Common misconceptions

### Misconception 1: CoT just means making the model more verbose

No.
The core is:

- Making intermediate structure explicit

### Misconception 2: All tasks should use CoT by default

Not true.
Whether to enable it depends on whether the problem is truly multi-step.

### Misconception 3: With CoT, the answer is always reliable

Also not true.
Chain reasoning can improve stability,
but it does not automatically eliminate all errors.

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
task_goal: what the agent is trying to solve
plan_or_trace: reasoning steps, plan, ReAct trace, or execution graph
observation: what changed after each action
failure_check: hallucinated step, stale observation, loop, or unverified conclusion
eval_action: compare against expected result and revise the plan
```

## Summary

The most important thing in this section is not remembering the English name `Chain-of-Thought`,
but building a practical judgment:

> **When a problem depends on multi-step intermediate states, letting the model explicitly break down the steps often improves stability; but the value of CoT lies in structured intermediate process, not in endlessly lengthening the output.**

Once you understand that clearly,
the next time you see:

- ReAct
- Plan-and-Execute
- Self-checking and evaluation

it will be much easier to understand that they are continuing to organize reasoning on top of CoT.

---

## Exercises

1. Replace the ticket-queue problem in the example with your own multi-step operations problem, and compare `bad_direct_answer` and `chain_reason_answer`.
2. Why do we say the core value of CoT lies in “explicit intermediate structure” rather than “longer output”?
3. Think of a simple question that is not suitable for CoT, and explain why.
4. If you were using CoT in a product, would you prefer free-form text or structured slots? Why?

<details>
<summary>Reference implementation and walkthrough</summary>

1. The chain answer should expose sub-results, while the bad direct answer often hides assumptions or arithmetic mistakes.
2. CoT is valuable when the task has dependencies that need to be tracked; long text without structure is just verbosity.
3. Simple lookup questions such as "What is the capital of Japan?" do not need CoT because intermediate reasoning adds noise and cost.
4. For products, structured slots are usually better because they are easier to validate, log, evaluate, and hide from users when needed.

</details>
