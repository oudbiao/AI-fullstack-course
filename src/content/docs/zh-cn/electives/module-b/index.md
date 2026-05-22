---
title: "E.B Python 进阶路线图"
description: "Python 进阶选修模块的简明实操路线图：装饰器、生成器、asyncio 和元编程，用来构建可追踪工程流水线。"
sidebar:
  order: 0
---
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
| 1 | [E.B.1 装饰器](/zh-cn/electives/module-b/01-decorators-advanced/) | 不改业务代码就加 timing 或 logging |
| 2 | [E.B.2 迭代器与生成器](/zh-cn/electives/module-b/02-iterators-advanced/) | 不一次性加载全部数据也能流式处理 |
| 3 | [E.B.3 并发](/zh-cn/electives/module-b/03-concurrency/) | 用 timeout、cancel 和限流思维运行 async 任务 |
| 4 | [E.B.4 元编程](/zh-cn/electives/module-b/04-metaprogramming/) | 显式注册工具或 handler |

## 通过标准

你能构建一个可追踪 pipeline，至少用到装饰器、生成器、异步调用或注册表中的一种，并解释为什么更容易调试，就算通过本模块。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
Python 模式：装饰器、迭代器、生成器、并发原语，或元编程钩子
代码产物：最小可运行示例加上打印输出
使用场景：这种模式在哪种 AI 应用、流水线、工具或服务器中更有用
失败检查：隐藏副作用、难读的抽象、竞态条件或过度设计
期望产出：带实际 AI 系统用途说明的小型高级 Python 示例
```

<details>
<summary>检查思路与讲解</summary>

一个合格答案会展示一个可追踪流水线：比如装饰器加日志、生成器做流式处理、async 并发跑多个 I/O，或者用注册表显式管理工具。证据应包含最小可运行输出，以及为什么调试会更容易。

如果只是“写得更巧”，还不够。要说明可观察性、可复用性或控制力为什么真的变好了。

</details>
