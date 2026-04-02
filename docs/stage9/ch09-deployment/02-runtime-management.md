---
title: "9.2 运行时管理"
sidebar_position: 50
description: "从并发控制、超时、重试、熔断与指标观测出发，理解 Agent 上线后如何稳定运行。"
keywords: [runtime management, concurrency, timeout, retry, circuit breaker, metrics]
---

# 运行时管理

:::tip 本节定位
本地 demo 只要“能跑一次”通常就算成功。  
线上系统的要求完全不同：

- 高峰期还能跑
- 依赖抖动时还能稳
- 延迟和成本还能控

这就是运行时管理要解决的问题。
:::

## 学习目标

- 理解并发、超时、重试、熔断分别在防什么
- 学会搭一个最小运行时管理器
- 理解为什么运行指标和模型指标一样重要
- 建立“系统稳定性优先于单次成功”的工程意识

---

## 一、为什么 Agent 特别容易遇到运行时问题？

### 1.1 一次请求往往不是一次调用

Agent 常见链路包括：

- 模型推理
- 工具调用
- 检索
- 再推理

这意味着一条用户请求可能包含多段子调用。  
链路越长，运行时波动越容易被放大。

### 1.2 上线后最先暴露的往往不是“答错”，而是“跑不稳”

典型症状：

- 高并发时超时变多
- 上游暂时失败后重试风暴
- 请求排队过长
- 个别慢请求拖垮整体吞吐

所以运行时管理本质上是在保护系统可用性。

---

## 二、四个最关键的运行时机制

### 2.1 并发控制

限制同时执行的任务数，避免资源被瞬间打满。

### 2.2 超时

为每个步骤设边界，防止请求无限挂起。

### 2.3 重试

只对临时错误做有限重试，而不是所有错误都重来。

### 2.4 熔断

当某个依赖连续失败时，短期停止继续打它，避免把故障放大。

---

## 三、先跑一个最小运行时管理器

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

### 3.1 这段代码最该看哪几处？

- `Semaphore`：并发限制
- `wait_for`：超时
- `attempt > 0`：重试计数
- `breaker_open`：熔断

### 3.2 为什么这已经很接近真实运行时问题？

因为它覆盖了三类真实线上情况：

- 正常成功
- 慢请求超时
- 连续失败触发保护

---

## 四、运行时指标该怎么读？

优先看这几项：

- `success / total`：成功率
- `timeout / total`：超时率
- `retry / total`：重试比
- `rejected_by_breaker`：熔断拒绝量
- `avg_latency_ms`：平均成功延迟

如果超时率高，先查：

- 上游慢不慢
- 超时阈值是否太小
- 并发是否太高导致排队

如果重试比高，先查：

- 是不是在重试不可恢复错误
- 上游是否不稳定

---

## 五、运行时优化最常见的方向

### 5.1 限流和背压

当系统接近满载时，要主动：

- 拒绝低优先级请求
- 或排队上限控制

### 5.2 降级

例如：

- 关闭高成本工具链
- 切换到缓存结果
- 返回更轻量的安全答复

### 5.3 分依赖设置策略

不同工具不应共用完全相同的：

- 超时
- 重试
- 熔断阈值

因为它们稳定性和成本不同。

---

## 六、最常见误区

### 6.1 误区一：并发越高越好

并发太高可能直接把系统和上游一起压垮。

### 6.2 误区二：重试一定提升成功率

错误分类不对时，重试只会放大故障。

### 6.3 误区三：只看平均延迟

高分位延迟和超时率往往更能反映真实体验。

---

## 小结

这节最重要的是建立一个部署视角：

> **Agent 运行时管理的核心，不是让每次请求“尽量试到成功”，而是用并发、超时、重试和熔断把系统整体稳定性保护住。**

当这层补齐后，系统才算真正具备了上线基础。

---

## 练习

1. 把示例的 `max_concurrency` 改成 `1` 和 `3`，比较结果变化。
2. 把 `timeout_sec` 调大，观察超时率会怎么变。
3. 为什么说“重试次数”不能脱离“错误类型”单独设计？
4. 想一想：如果某个工具特别贵，你会在运行时层加什么保护？
