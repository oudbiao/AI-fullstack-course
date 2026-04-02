---
title: "1.3 并发编程（含 asyncio）"
sidebar_position: 10
description: "从 I/O 密集任务讲起，理解线程、协程和 asyncio 在服务代码中的适用边界，并学会一个最小并发控制器。"
keywords: [asyncio, concurrency, async, semaphore, gather, Python]
---

# 并发编程（含 asyncio）

:::tip 本节定位
并发编程在 Python 里最容易被学成“API 记忆题”。  
但对工程来说，更重要的问题其实是：

> **什么时候需要并发，什么时候会把事情搞得更复杂？**

尤其在 AI 应用和服务侧，很多任务本质上是 I/O 密集型，这正是 `asyncio` 最擅长的场景。
:::

## 学习目标

- 理解 I/O 密集和 CPU 密集任务的区别
- 理解 `asyncio` 为什么适合很多服务场景
- 学会用 `gather`、`Semaphore` 和超时控制组织并发
- 建立“并发是工具，不是默认答案”的意识

---

## 一、为什么很多 Python 工程会走到 asyncio？

### 1.1 因为很多任务都在“等”

例如：

- 等 HTTP 返回
- 等数据库返回
- 等文件读取

这类任务真正占时间的不是 CPU 计算，  
而是等待外部 I/O。

### 1.2 asyncio 的核心价值

它允许你在等待一个任务时，  
切去推进别的任务。

这特别适合：

- 爬取
- API 编排
- 多工具服务
- 批量请求

### 1.3 一个类比

同步代码像一个窗口一次只服务一个人。  
异步代码更像取号排队，窗口在等某个人资料时还能先办别人的单。

---

## 二、先看一个最小异步并发示例

```python
import asyncio


async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name} done"


async def main():
    results = await asyncio.gather(
        fetch("task_a", 0.2),
        fetch("task_b", 0.1),
    )
    print(results)


asyncio.run(main())
```

### 2.1 这段代码真正想说明什么？

它说明：

- 两个等待任务可以并发推进

如果换成同步串行，  
总耗时会更接近：

- `0.2 + 0.1`

而不是：

- `max(0.2, 0.1)`

### 2.2 为什么这在 AI 应用里很常见？

因为很多应用会同时做：

- 检索
- 调多个 API
- 读写多个服务

这些都不是重 CPU，而是重等待。

---

## 三、为什么并发不是越多越好？

### 3.1 并发过大可能把上游打崩

如果你一次发 1000 个请求，  
也许不是更快，而是：

- 被限流
- 超时增加
- 上游雪崩

### 3.2 所以常常需要并发上限

最简单的做法之一就是：

- `Semaphore`

它能限制同时正在跑的任务数。

```python
import asyncio


semaphore = asyncio.Semaphore(2)


async def bounded_fetch(name, delay):
    async with semaphore:
        print("start", name)
        await asyncio.sleep(delay)
        print("end", name)
        return name


async def main():
    tasks = [bounded_fetch(f"task_{i}", 0.2) for i in range(5)]
    results = await asyncio.gather(*tasks)
    print(results)


asyncio.run(main())
```

### 3.3 这段代码最值得学什么？

并发不只是“能不能一起跑”，  
还包括：

- 一次放多少一起跑

这正是很多线上服务的核心控制点。

---

## 四、超时和取消为什么也很重要？

### 4.1 没有超时，慢任务会一直挂着

这在外部依赖很多时非常危险。  
最常见做法是：

- `asyncio.wait_for(...)`

```python
import asyncio


async def slow_task():
    await asyncio.sleep(2)
    return "done"


async def main():
    try:
        result = await asyncio.wait_for(slow_task(), timeout=0.5)
        print(result)
    except asyncio.TimeoutError:
        print("timeout")


asyncio.run(main())
```

### 4.2 为什么这对 Agent 特别重要？

因为 Agent 很多时候依赖：

- 外部工具
- 上游模型
- 检索系统

如果没有超时，系统很容易卡住整条链路。

---

## 五、什么时候不该优先用 asyncio？

### 5.1 纯 CPU 密集任务

例如：

- 大量数值计算
- 图像批量变换

这类任务更适合：

- 多进程
- 原生高性能库

### 5.2 团队还没准备好接受异步复杂度

异步代码会引入：

- 调试复杂度
- 状态管理难度

如果场景不需要，不必强上。

### 5.3 同步已经够简单够稳

小脚本、小任务里，  
同步有时反而更清晰。

---

## 六、最常见误区

### 6.1 误区一：并发就是更快

不一定。  
关键看任务是不是 I/O 密集。

### 6.2 误区二：`async` 到处都该加

异步是手段，不是风格标签。

### 6.3 误区三：只会 `gather` 就算会 asyncio

真实工程里更重要的常常是：

- 限流
- 超时
- 错误处理

---

## 小结

这节最重要的，不是把 `asyncio` 学成 API 清单，  
而是建立一个实用判断：

> **如果任务主要在等待 I/O，那么异步并发通常能显著提升吞吐；但真正上线时，还必须配合并发上限、超时和错误控制。**

只要这个判断稳住了，你后面再看服务端并发代码就会顺很多。

---

## 练习

1. 把 `Semaphore(2)` 改成 `Semaphore(1)` 和 `Semaphore(5)`，比较日志顺序变化。
2. 想一想：为什么很多 Agent / API 编排任务天然适合 asyncio？
3. 为什么说超时控制在异步系统里和 `gather` 一样重要？
4. 举一个你觉得“不适合用 asyncio 优先解决”的任务例子。
