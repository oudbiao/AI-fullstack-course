---
title: "E.B.1 装饰器进阶用法"
sidebar_position: 8
description: "用装饰器封装日志、重试、计时和权限逻辑，避免在每个函数里重复写。"
keywords: [decorators, Python, wraps, retry, logging, timing, authorization]
---

# E.B.1 装饰器进阶用法

![Python 装饰器执行流程图](/img/course/elective-python-decorator-flow.webp)

![装饰器横切逻辑分层图](/img/course/elective-decorator-crosscutting-layers.webp)

装饰器会在函数外面包一层可复用行为。当很多函数都需要同样的日志、计时、重试、权限检查或 trace 时，就适合用装饰器。

## 准备内容

- Python 3.10+
- 不需要第三方包
- 理解函数基础

## 关键术语

- **Wrapper（包装函数）**：真正运行在原函数外层的内部函数。
- **Cross-cutting logic（横切逻辑）**：很多地方都需要，但不属于业务核心的逻辑。
- **`functools.wraps`**：装饰后仍保留原函数名称和元信息。
- **装饰器顺序**：函数被调用时，最上面的装饰器先执行。

## 运行日志和重试装饰器

创建 `decorator_demo.py`：

```python
from functools import wraps


def log_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[LOG] start {fn.__name__}")
        result = fn(*args, **kwargs)
        print(f"[LOG] end {fn.__name__}")
        return result

    return wrapper


def retry(max_retries=2):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_retries + 2):
                try:
                    return fn(*args, **kwargs)
                except RuntimeError as error:
                    last_error = error
                    print(f"[RETRY] attempt={attempt} error={error}")
            raise last_error

        return wrapper

    return decorator


state = {"attempt": 0}


@log_call
@retry(max_retries=2)
def fetch_model_info(model_id):
    state["attempt"] += 1
    if state["attempt"] < 2:
        raise RuntimeError("temporary network error")
    return {"model_id": model_id, "status": "ready"}


print(fetch_model_info("demo-v1"))
print(fetch_model_info.__name__)
```

运行：

```bash
python decorator_demo.py
```

预期输出：

```text
[LOG] start fetch_model_info
[RETRY] attempt=1 error=temporary network error
[LOG] end fetch_model_info
{'model_id': 'demo-v1', 'status': 'ready'}
fetch_model_info
```

这个例子说明三件事：业务函数保持简短，重试逻辑集中管理，`wraps` 保留了函数名。

## 改一下顺序

把装饰器顺序换成：

```text
@retry(max_retries=2)
@log_call
def fetch_model_info(model_id):
```

这时日志会在每次重试内部执行。服务代码里装饰器顺序很重要。

## 什么时候适合用装饰器

适合：

1. 日志和追踪
2. 计时
3. 不稳定 I/O 的重试
4. 权限检查
5. 框架注册

如果包装层隐藏了关键业务逻辑，或者一个函数已经叠了太多层，就不适合继续加装饰器。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
python_pattern: decorator, iterator, generator, concurrency primitive, or metaprogramming hook
code_artifact: minimal runnable example plus printed output
use_case: where this pattern improves an AI app, pipeline, tool, or server
failure_check: hidden side effects, unreadable abstraction, race condition, or overengineering
Expected_output: small advanced-Python example with a practical AI-system use note
```

## 常见错误

- 忘记 `@wraps`，导致日志和框架看到的函数名都变成 `wrapper`。
- 重试所有异常，包括本应立即失败的校验错误或权限错误。
- 装饰器堆太多，执行顺序很难排查。

## 练习

在 `fetch_model_info` 前加一个 `require_role("admin")` 装饰器。非 admin 用户抛出 `PermissionError`，并且不要重试权限错误。

<details>
<summary>参考答案与讲解</summary>

好的实现会先处理权限，再让 retry 只处理临时失败。可以把 `require_role("admin")` 放在重试路径外层，或者修改 `retry`，让它遇到 `PermissionError` 时立即重新抛出。

预期行为是：

- admin 用户可以正常调用函数。
- 非 admin 用户会得到 `PermissionError`。
- 权限失败不会反复打印 retry 日志，因为权限错误不是临时网络错误。

</details>
