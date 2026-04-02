---
title: "1.1 装饰器高级用法"
sidebar_position: 8
description: "从日志、计时、重试和权限控制这些真实工程需求出发，理解装饰器为什么是 Python 服务代码里的高频模式。"
keywords: [decorators, Python, wraps, retry, logging, timing, authorization]
---

# 装饰器高级用法

:::tip 本节定位
很多人第一次学装饰器时，印象停留在：

- 语法看起来有点绕
- 好像很高级

但在真实工程里，装饰器最重要的价值其实非常朴素：

> **把“横切逻辑”统一包起来，而不是在每个函数里重复写。**

这些横切逻辑常见包括：

- 日志
- 计时
- 重试
- 权限校验

所以这节课不会把装饰器讲成“花式语法”，而会把它放回工程问题里看。
:::

## 学习目标

- 理解装饰器在工程中最常见的用途
- 学会写带参数和保留元信息的装饰器
- 理解 `functools.wraps` 为什么重要
- 通过可运行示例掌握日志、计时、重试三类常见模式

---

## 一、为什么装饰器在工程里这么常见？

### 1.1 因为很多逻辑会反复附着在不同函数外层

例如你有很多函数都需要：

- 打日志
- 统计耗时
- 捕获异常
- 做权限检查

如果每个函数都手写一遍，代码会很快变成重复模板。

### 1.2 装饰器的核心价值

它不是“神奇修改函数”，  
而是：

- 接收一个函数
- 在外面包一层通用逻辑
- 再返回新的函数

也就是说：

> **装饰器是在复用“函数外层行为”。**

### 1.3 一个类比

函数像真正做事的人。  
装饰器像给所有人统一加上的：

- 工牌
- 打卡
- 安检
- 计时器

人没变，但工作流程更统一了。

---

## 二、先从最常见的日志装饰器开始

下面这段代码会演示：

- 调用前后打印日志
- 保留原函数元信息

```python
from functools import wraps


def log_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[LOG] calling {fn.__name__} args={args} kwargs={kwargs}")
        result = fn(*args, **kwargs)
        print(f"[LOG] {fn.__name__} returned {result}")
        return result

    return wrapper


@log_call
def add(a, b):
    return a + b


print(add(3, 5))
print(add.__name__)
```

### 2.1 为什么 `wraps` 很重要？

如果没有 `@wraps(fn)`，  
被装饰后的函数元信息可能会丢失，例如：

- `__name__`
- `__doc__`

这会影响：

- 调试
- 日志
- 文档生成
- 某些框架行为

### 2.2 为什么日志装饰器很常见？

因为它是一种非常典型的“横切逻辑”：

- 跟业务本身无关
- 却需要很多函数都做

---

## 三、计时装饰器：性能问题先可见

很多工程问题不是“功能错”，而是：

- 太慢

计时装饰器可以帮助你快速定位热点。

```python
import time
from functools import wraps


def measure_time(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - started
        print(f"[TIME] {fn.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


@measure_time
def fake_inference():
    time.sleep(0.2)
    return "done"


print(fake_inference())
```

### 3.1 这类装饰器在 AI 工程里很实用

例如你想快速看：

- tokenizer 花了多久
- 检索花了多久
- 模型推理花了多久

不用每个函数都手写计时逻辑。

---

## 四、带参数的装饰器：为什么它更接近真实工程？

很多时候你不只是想“是否启用某逻辑”，  
还想指定：

- 重试几次
- 权限级别是什么
- 超时阈值是多少

这就需要带参数装饰器。

```python
from functools import wraps


def require_role(role):
    def decorator(fn):
        @wraps(fn)
        def wrapper(user, *args, **kwargs):
            if user.get("role") != role:
                raise PermissionError(f"required role: {role}")
            return fn(user, *args, **kwargs)

        return wrapper

    return decorator


@require_role("admin")
def delete_model(user, model_name):
    return f"deleted:{model_name}"


admin = {"name": "alice", "role": "admin"}
guest = {"name": "bob", "role": "guest"}

print(delete_model(admin, "v1"))
try:
    print(delete_model(guest, "v1"))
except PermissionError as e:
    print("error:", e)
```

### 4.1 为什么这里是两层函数？

因为：

- 第一层接收装饰器参数
- 第二层接收被装饰函数
- 第三层才是真正执行时的包装逻辑

一开始会觉得绕，  
但只要记住三层分工就不容易乱。

---

## 五、重试装饰器：非常典型的生产代码模式

下面这个例子会模拟一个不稳定函数，  
并用装饰器统一重试逻辑。

```python
from functools import wraps


def retry(max_retries=2):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"[RETRY] attempt={attempt + 1} error={e}")
            raise last_error

        return wrapper

    return decorator


state = {"count": 0}


@retry(max_retries=2)
def flaky_call():
    state["count"] += 1
    if state["count"] < 3:
        raise RuntimeError("temporary error")
    return "ok"


print(flaky_call())
```

### 5.1 这段代码的工程意义是什么？

它说明装饰器非常适合收纳：

- 重试
- 限流
- 熔断

这类外围控制逻辑。

### 5.2 为什么重试装饰器要慎用？

因为不是所有错误都适合重试。  
例如：

- 参数错误
- 权限错误

重试只会浪费资源。

---

## 六、装饰器最容易踩的坑

### 6.1 误区一：一看到装饰器就觉得“高级”

装饰器不是为了炫技，  
而是为了减少重复逻辑。

### 6.2 误区二：装饰器套太多层

如果一个函数上挂了太多装饰器，  
调试和理解都会变难。

### 6.3 误区三：不用 `wraps`

这会让元信息丢失，后期排查问题很痛苦。

---

## 小结

这节最重要的，不是把装饰器背成语法谜题，  
而是建立一个工程判断：

> **装饰器最适合封装日志、计时、重试、权限这种横切逻辑，让业务函数本身保持更干净。**

只要这个判断立住了，  
你后面读框架代码、服务中间件和库源码时就会顺很多。

---

## 练习

1. 给 `measure_time` 再加一个参数 `label`，练习带参数装饰器。
2. 想一想：什么错误适合放进 `retry` 装饰器，什么错误不适合？
3. 把日志、计时和重试三个装饰器叠在一个函数上，观察执行顺序。
4. 用自己的话解释：为什么装饰器特别适合“横切逻辑”？
