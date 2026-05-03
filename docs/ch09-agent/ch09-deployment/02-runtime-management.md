---
title: "9.3 Runtime Management"
sidebar_position: 50
description: "Understand how an Agent stays stable after deployment through concurrency control, timeouts, retries, circuit breaking, and metrics observability."
keywords: [runtime management, concurrency, timeout, retry, circuit breaker, metrics]
---

# Runtime Management

:::tip Section focus
For a local demo, “it runs once” is usually good enough.
Production systems have very different requirements:

- It still runs under peak traffic
- It stays stable when dependencies are flaky
- It keeps latency and cost under control

That is what runtime management is here to solve.
:::

## Learning Objectives

- Understand what concurrency, timeouts, retries, and circuit breakers are protecting against
- Learn how to build a minimal runtime manager
- Understand why runtime metrics are just as important as model metrics
- Build an engineering mindset that prioritizes system stability over one-off success

---

## 1. Why are Agents especially prone to runtime problems?

### 1.1 One request is often not one call

A typical Agent workflow includes:

- Model inference
- Tool calling
- Retrieval
- Another round of inference

This means a single user request may contain several sub-calls.
The longer the chain, the more runtime fluctuations are amplified.

### 1.2 What shows up first after deployment is often not “wrong answers,” but “unstable execution”

Typical symptoms:

- More timeouts under high concurrency
- Retry storms after temporary upstream failures
- Requests queuing too long
- A few slow requests dragging down overall throughput

So runtime management is essentially about protecting system availability.

---

## 2. The four most important runtime mechanisms

### 2.1 Concurrency control

Limit the number of tasks running at the same time so resources are not exhausted all at once.

### 2.2 Timeouts

Set a boundary for each step to prevent requests from hanging forever.

### 2.3 Retries

Retry only a limited number of times for temporary errors, instead of starting over for every error.

### 2.4 Circuit breaking

When a dependency keeps failing, stop calling it for a while to avoid amplifying the failure.

---

## 3. First, run a minimal runtime manager

```python
import asyncio
import time


class AgentRuntime:
    def __init__(self, max_concurrency=2, timeout_sec=0.8, max_retries=1, breaker_threshold=2):
        self.semaphore = asyncio.Semaphore(max_concurrency)
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries
        self.breaker_threshold = breaker_threshold

        self.breaker_open = False
        self.failure_streak = 0
        self.metrics = {
            "total": 0,
            "success": 0,
            "timeout": 0,
            "error": 0,
            "retry": 0,
            "rejected_by_breaker": 0,
            "latency_ms_total": 0.0,
        }

    async def _upstream_call(self, task):
        await asyncio.sleep(task["latency"])
        if task.get("should_fail"):
            raise RuntimeError("upstream_error")
        return {"task_id": task["id"], "result": f"ok:{task['payload']}"}

    async def handle(self, task):
        self.metrics["total"] += 1

        if self.breaker_open:
            self.metrics["rejected_by_breaker"] += 1
            return {"ok": False, "error": "circuit_open"}

        started = time.perf_counter()
        last_error = None

        for attempt in range(self.max_retries + 1):
            if attempt > 0:
                self.metrics["retry"] += 1

            try:
                async with self.semaphore:
                    result = await asyncio.wait_for(
                        self._upstream_call(task),
                        timeout=self.timeout_sec,
                    )

                latency_ms = (time.perf_counter() - started) * 1000
                self.metrics["success"] += 1
                self.metrics["latency_ms_total"] += latency_ms
                self.failure_streak = 0
                return {"ok": True, "result": result, "attempts": attempt + 1}

            except asyncio.TimeoutError:
                last_error = "timeout"
                if attempt == self.max_retries:
                    self.metrics["timeout"] += 1
                    self.failure_streak += 1
                    break
            except Exception as e:
                last_error = str(e)
                if attempt == self.max_retries:
                    self.metrics["error"] += 1
                    self.failure_streak += 1
                    break

        if self.failure_streak >= self.breaker_threshold:
            self.breaker_open = True

        return {"ok": False, "error": last_error}

    def summary(self):
        avg_latency = (
            self.metrics["latency_ms_total"] / self.metrics["success"]
            if self.metrics["success"] else 0.0
        )
        return {**self.metrics, "avg_latency_ms": round(avg_latency, 2)}


async def main():
    runtime = AgentRuntime(max_concurrency=2, timeout_sec=0.7, max_retries=1, breaker_threshold=2)

    tasks = [
        {"id": "r1", "payload": "refund", "latency": 0.2},
        {"id": "r2", "payload": "slow", "latency": 1.0},
        {"id": "r3", "payload": "fail", "latency": 0.1, "should_fail": True},
        {"id": "r4", "payload": "normal", "latency": 0.3},
        {"id": "r5", "payload": "after_breaker", "latency": 0.1},
    ]

    results = []
    for task in tasks:
        results.append(await runtime.handle(task))

    print("results:")
    for item in results:
        print(item)

    print("\nmetrics:")
    print(runtime.summary())
    print("breaker_open:", runtime.breaker_open)


asyncio.run(main())
```

### 3.1 Which parts of this code should you pay the most attention to?

- `Semaphore`: concurrency limiting
- `wait_for`: timeout
- `attempt > 0`: retry counting
- `breaker_open`: circuit breaking

### 3.2 Why is this already very close to real runtime problems?

Because it covers three real production scenarios:

- Normal success
- Slow requests timing out
- Continuous failures triggering protection

---

## 4. How should runtime metrics be interpreted?

Start with these:

- `success / total`: success rate
- `timeout / total`: timeout rate
- `retry / total`: retry ratio
- `rejected_by_breaker`: number of requests rejected by the circuit breaker
- `avg_latency_ms`: average successful latency

If the timeout rate is high, check first:

- Is the upstream service slow?
- Is the timeout threshold too small?
- Is concurrency too high, causing queueing?

If the retry ratio is high, check first:

- Are you retrying errors that cannot be recovered?
- Is the upstream service unstable?

---

## 5. The most common runtime optimization directions

### 5.1 Rate limiting and backpressure

When the system is close to saturation, you should actively:

- Reject low-priority requests
- Or cap the queue length

### 5.2 Fallback and degradation

For example:

- Disable expensive tool chains
- Switch to cached results
- Return a lighter-weight safe response

### 5.3 Use different policies for different dependencies

Different tools should not share exactly the same:

- Timeout
- Retry
- Circuit-breaker thresholds

Because their stability and cost are different.

---

## 6. Most common misconceptions

### 6.1 Misconception 1: Higher concurrency is always better

Too much concurrency can directly overwhelm both your system and the upstream service.

### 6.2 Misconception 2: Retries always improve success rate

If error classification is wrong, retries will only amplify the failure.

### 6.3 Misconception 3: Only look at average latency

High-percentile latency and timeout rate often reflect the real user experience much better.

---

## Summary

The most important thing in this section is to build a deployment mindset:

> **The core of Agent runtime management is not to keep trying until every request succeeds, but to protect overall system stability with concurrency control, timeouts, retries, and circuit breaking.**

Once this layer is in place, the system truly has a foundation for production deployment.

---

## Exercises

1. Change the example `max_concurrency` to `1` and `3`, and compare the results.
2. Increase `timeout_sec` and observe how the timeout rate changes.
3. Why can’t “number of retries” be designed independently of “error type”?
4. Think about it: if a certain tool is especially expensive, what protection would you add at the runtime layer?
