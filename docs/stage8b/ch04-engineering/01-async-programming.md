---
title: "4.2 异步编程与并发调用"
sidebar_position: 17
description: "从为什么 LLM 工程常常慢在等待，到 asyncio、gather、Semaphore 和超时控制，理解并发调用的工程主线。"
keywords: [asyncio, concurrency, gather, semaphore, timeout, async programming, LLM engineering]
---

# 异步编程与并发调用

:::tip 本节定位
做 LLM 应用时，很多人第一次的性能瓶颈不是模型不够强，而是：

> **系统大部分时间都在等。**

等接口、等检索、等工具、等数据库。  
异步编程就是在解决这种“CPU 没在忙，但任务还卡着”的问题。
:::

## 学习目标

- 理解为什么 LLM 应用天然适合异步并发
- 分清同步调用和异步调用的区别
- 学会 `async` / `await` / `gather` 的基本用法
- 理解并发限制和超时控制为什么重要
- 看懂一个更贴近真实场景的异步调用示例

---

## 一、为什么 LLM 工程特别容易遇到“等待”？

### 1.1 一个真实得不能再真实的场景

你做一个问答助手，一次请求可能要：

1. 查知识库
2. 调模型
3. 再调一个工具

如果每一步都顺序等完再做下一步，整体延迟很容易拉长。

### 1.2 关键点：很多步骤不是“计算慢”，而是“等待慢”

例如：

- 网络请求
- 数据库查询
- 第三方 API

这些阶段，CPU 很多时候并没有真正忙满。  
这就意味着：

> 可以在等待一个任务的时候，先去做别的任务。 

这正是异步编程最有价值的地方。

---

## 二、同步和异步到底差在哪？

### 2.1 同步：一个任务做完再做下一个

```python
import time

def task(name, delay):
    time.sleep(delay)
    return f"{name} done"

start = time.time()
print(task("A", 1))
print(task("B", 1))
print("elapsed =", round(time.time() - start, 2))
```

这段代码会大约花 2 秒。

### 2.2 异步：发出去后先别傻等

```python
import asyncio
import time

async def task(name, delay):
    await asyncio.sleep(delay)
    return f"{name} done"

async def main():
    start = time.time()
    results = await asyncio.gather(
        task("A", 1),
        task("B", 1)
    )
    print(results)
    print("elapsed =", round(time.time() - start, 2))

asyncio.run(main())
```

这一版通常只要大约 1 秒。

### 2.3 真正的差别是什么？

不是“异步更神秘”，而是：

> 等待期间，调度器不会傻站着，而会去推进别的协程。 

---

## 三、`async` 和 `await` 到底在表达什么？

### 3.1 `async def`

表示：

> 这是一个协程函数。 

它不会立刻像普通函数那样直接完成，而是可以被调度执行。

### 3.2 `await`

表示：

> 这里需要等一个异步结果回来。 

但等的这段时间，调度器可以去处理别的协程。

### 3.3 一个最容易理解的类比

同步像：

- 做饭时站在锅前傻等水烧开

异步像：

- 水在烧时，你先去切菜

---

## 四、`gather` 为什么这么常见？

### 4.1 因为很多 LLM 场景天然就是“并发查几路”

例如：

- 同时调 3 个检索器
- 同时请求多个模型候选
- 同时查几个数据源

这时 `asyncio.gather()` 很自然。

### 4.2 一个更贴近 LLM 场景的示例

```python
import asyncio

async def retrieve_docs():
    await asyncio.sleep(0.3)
    return ["退款政策", "证书说明"]

async def call_model():
    await asyncio.sleep(0.5)
    return "模型初步回复"

async def fetch_user_profile():
    await asyncio.sleep(0.2)
    return {"user_level": "beginner"}

async def main():
    docs, model_reply, profile = await asyncio.gather(
        retrieve_docs(),
        call_model(),
        fetch_user_profile()
    )
    print(docs)
    print(model_reply)
    print(profile)

asyncio.run(main())
```

这就已经非常像真实应用里“并行查几层信息”的写法了。

---

## 五、为什么不能无限并发？

### 5.1 因为外部系统不是无限扛得住

如果你一口气并发 1000 个请求，可能会遇到：

- API 限流
- 数据库被打爆
- 文件句柄耗尽
- 上游服务超时

所以异步编程不是“并发越多越好”，而是：

> **要在吞吐和稳定性之间找平衡。**

### 5.2 用 `Semaphore` 做并发限制

```python
import asyncio

semaphore = asyncio.Semaphore(3)

async def limited_task(i):
    async with semaphore:
        await asyncio.sleep(0.2)
        return f"task_{i}"

async def main():
    results = await asyncio.gather(*(limited_task(i) for i in range(10)))
    print(results)

asyncio.run(main())
```

这个例子表示：

- 虽然一共发起了 10 个任务
- 但同一时刻最多只允许 3 个一起跑

---

## 六、超时控制为什么特别重要？

### 6.1 因为有些请求会“卡死”

真实系统里，如果一个上游服务慢到离谱，而你又没有超时控制，整个请求就可能一直挂住。

### 6.2 一个最小超时示例

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
        print("task timeout")

asyncio.run(main())
```

这在工程里非常关键，因为“无限等待”通常比“明确失败”更糟。

---

## 七、异步编程在 LLM 工程里的典型使用点

### 7.1 检索并发

同时查：

- FAQ
- 向量库
- 数据库

### 7.2 多模型并发

例如：

- 主模型 + 备用模型
- 多候选答案并发生成

### 7.3 工具并发

比如一个 Agent 要同时：

- 查天气
- 查用户状态
- 查订单记录

### 7.4 日志与监控链路

有些日志和上报也适合异步做，避免堵住主请求。

---

## 八、一个更像真实系统的小例子

```python
import asyncio

async def search_kb(query):
    await asyncio.sleep(0.3)
    return f"知识库结果: {query}"

async def get_user_status(user_id):
    await asyncio.sleep(0.2)
    return {"user_id": user_id, "progress": 0.15}

async def call_llm(prompt):
    await asyncio.sleep(0.4)
    return f"LLM 回复: {prompt}"

async def handle_request(query, user_id):
    kb_result, user_status = await asyncio.gather(
        search_kb(query),
        get_user_status(user_id)
    )

    prompt = f"请根据以下信息回答：{kb_result}，用户状态：{user_status}"
    answer = await call_llm(prompt)
    return answer

print(asyncio.run(handle_request("退款政策是什么", 1)))
```

这个例子已经很像真实后端：

- 前半段并发取上下文
- 后半段再统一送给模型

---

## 九、初学者最常踩的坑

### 9.1 把异步理解成“更快的同步”

异步不是加速魔法，它更像是更聪明的等待方式。

### 9.2 一上来就无限并发

这很容易把系统压坏。

### 9.3 没有超时和异常处理

一旦某个任务卡死，整个请求链路就可能拖垮。

---

## 十、小结

这一节最重要的不是背 `async` / `await` 语法，而是理解：

> **异步编程的核心，是把“等待时间”利用起来，让系统在 I/O 密集型场景下更高效、更稳定。**

这在 LLM 工程里几乎是绕不开的基本功。

---

## 练习

1. 把本节的并发示例里任务数从 10 增加到 30，并调整 `Semaphore` 的大小。
2. 在 `handle_request()` 里再加一个并发工具调用。
3. 想一想：为什么异步编程特别适合“多外部依赖”的 LLM 应用？
4. 用自己的话解释：异步编程为什么不是“让单个任务更快”，而是“让整体等待更聪明”？
