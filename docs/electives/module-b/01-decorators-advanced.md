---
title: "E.B.1 Advanced Decorator Usage"
sidebar_position: 8
description: "Use decorators to package logging, retry, timing, and permission logic without repeating it in every function."
keywords: [decorators, Python, wraps, retry, logging, timing, authorization]
---

# E.B.1 Advanced Decorator Usage

![Python decorator execution flow](/img/course/elective-python-decorator-flow-en.webp)

![Decorator cross-cutting logic layering diagram](/img/course/elective-decorator-crosscutting-layers-en.webp)

A decorator wraps a function with reusable outer behavior. Use it when many functions need the same logic, such as logging, timing, retry, permission checks, or tracing.

## What You Need

- Python 3.10+
- No external packages
- Basic understanding of functions

## Key Terms

- **Wrapper**: the inner function that runs before and after the original function.
- **Cross-cutting logic**: logic needed in many places but not part of the business task itself.
- **`functools.wraps`**: keeps the original function name and metadata after decoration.
- **Decorator order**: the top decorator runs first when the function is called.

## Run A Logging And Retry Decorator

Create `decorator_demo.py`:

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

Run it:

```bash
python decorator_demo.py
```

Expected output:

```text
[LOG] start fetch_model_info
[RETRY] attempt=1 error=temporary network error
[LOG] end fetch_model_info
{'model_id': 'demo-v1', 'status': 'ready'}
fetch_model_info
```

This shows three useful details: the business function stays short, retry behavior is centralized, and `wraps` preserves the function name.

## Change The Order

Swap the decorators:

```text
@retry(max_retries=2)
@log_call
def fetch_model_info(model_id):
```

Now logging runs inside each retry attempt. This is why decorator order matters in service code.

## When To Use Decorators

Use decorators for:

1. Logging and tracing
2. Timing
3. Retry around unstable I/O
4. Permission checks
5. Framework registration

Avoid decorators when the wrapper hides important business logic or when one function already has too many layers.

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
python_pattern: decorator, iterator, generator, concurrency primitive, or metaprogramming hook
code_artifact: minimal runnable example plus printed output
use_case: where this pattern improves an AI app, pipeline, tool, or server
failure_check: hidden side effects, unreadable abstraction, race condition, or overengineering
Expected_output: small advanced-Python example with a practical AI-system use note
```

## Common Mistakes

- Forgetting `@wraps`, then logs and frameworks see every function as `wrapper`.
- Retrying every exception, including validation or permission errors that should fail immediately.
- Stacking many decorators until the execution order is hard to debug.

## Practice

Add a `require_role("admin")` decorator before `fetch_model_info`. Make it raise `PermissionError` for non-admin users, and do not retry permission errors.
