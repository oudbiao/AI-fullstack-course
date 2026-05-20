---
title: "E.B Python 进阶路线图"
sidebar_position: 0
description: "Python 进阶选修模块的简明实操路线图：装饰器、生成器、asyncio 和元编程，用来构建可追踪工程流水线。"
---

# E.B Python 进阶路线图

当你的原型开始重复逻辑、等待慢请求、处理流式数据或动态注册工具时，再回来学这个选修模块。

## 先看工程路线

![Python 进阶模块学习地图](/img/course/elective-python-advanced-module-map.webp)

![生成器流式管道图](/img/course/elective-generator-stream-pipeline.webp)

高级 Python 的价值不在炫技，而在让代码更可观察、更可复用、更容易控制。

## 跑最小异步 trace

```python
import asyncio

async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name}:done"

async def main():
    results = await asyncio.gather(
        fetch("retrieval", 0.1),
        fetch("rerank", 0.05),
    )
    print(results)

asyncio.run(main())
```

预期输出：

```text
['retrieval:done', 'rerank:done']
```

这是最小 async 习惯：启动相互独立的工作，等待全部结果，再留下 trace。

## 按这个顺序学

| 步骤 | 课程 | 练习产物 |
|---|---|---|
| 1 | [E.B.1 装饰器](./01-decorators-advanced.md) | 不改业务代码就加 timing 或 logging |
| 2 | [E.B.2 迭代器与生成器](./02-iterators-advanced.md) | 不一次性加载全部数据也能流式处理 |
| 3 | [E.B.3 并发](./03-concurrency.md) | 用 timeout、cancel 和限流思维运行 async 任务 |
| 4 | [E.B.4 元编程](./04-metaprogramming.md) | 显式注册工具或 handler |

## 通过标准

你能构建一个可追踪 pipeline，至少用到装饰器、生成器、异步调用或注册表中的一种，并解释为什么更容易调试，就算通过本模块。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
python_pattern: decorator, iterator, generator, concurrency primitive, or metaprogramming hook
code_artifact: minimal runnable example plus printed output
use_case: where this pattern improves an AI app, pipeline, tool, or server
failure_check: hidden side effects, unreadable abstraction, race condition, or overengineering
Expected_output: small advanced-Python example with a practical AI-system use note
```
