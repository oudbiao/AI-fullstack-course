---
title: "1.1 Advanced Decorator Usage"
sidebar_position: 8
description: "Starting from real engineering needs such as logging, timing, retries, and permission control, understand why decorators are such a common pattern in Python service code."
keywords: [decorators, Python, wraps, retry, logging, timing, authorization]
---

# Advanced Decorator Usage

![Python decorator execution flow](/img/course/elective-python-decorator-flow.png)

![Decorator cross-cutting logic layering diagram](/img/course/elective-decorator-crosscutting-layers.png)

:::tip Reading the diagram
Decorators are best suited for cross-cutting logic such as logging, timing, retries, and permissions. When reading the diagram, focus on how the wrapper surrounds the original function, and why `functools.wraps` can preserve function identity and avoid debugging and framework recognition issues.
:::

:::tip Lesson focus
When many people first learn decorators, they tend to remember only that:

- the syntax looks a bit tricky
- it seems very advanced

But in real engineering, the most important value of decorators is actually very practical:

> **They wrap “cross-cutting logic” in one place instead of repeating it inside every function.**

Common examples of this cross-cutting logic include:

- logging
- timing
- retries
- permission checks

So in this lesson, we will not treat decorators as “fancy syntax.” Instead, we will look at them through real engineering problems.
:::

## Learning Objectives

- Understand the most common uses of decorators in engineering
- Learn how to write decorators with parameters and preserve metadata
- Understand why `functools.wraps` is important
- Master common logging, timing, and retry patterns through runnable examples

---

## 1. Why are decorators so common in engineering?

### 1.1 Because many pieces of logic repeatedly attach to the outside of different functions

For example, many of your functions may need to:

- print logs
- measure execution time
- catch exceptions
- perform permission checks

If you write all of that by hand in every function, the code will quickly turn into repetitive boilerplate.

### 1.2 The core value of decorators

They do not “magically modify functions,”  
instead they:

- receive a function
- wrap it with a layer of common logic
- return a new function

In other words:

> **Decorators reuse the “outer behavior” of functions.**

### 1.3 An analogy

A function is like a person doing real work.  
A decorator is like adding the same things to everyone:

- an ID badge
- clock-in records
- security checks
- a timer

The person does not change, but the workflow becomes more consistent.

---

## 2. Start with the most common logging decorator

The code below demonstrates:

- printing logs before and after a call
- preserving the original function metadata

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

### 2.1 Why is `wraps` so important?

Without `@wraps(fn)`,  
the metadata of the decorated function may be lost, such as:

- `__name__`
- `__doc__`

This can affect:

- debugging
- logging
- documentation generation
- the behavior of some frameworks

### 2.2 Why are logging decorators so common?

Because logging is a very typical form of cross-cutting logic:

- it is not part of the business logic itself
- yet many functions need it

---

## 3. Timing decorators: make performance issues visible first

Many engineering problems are not “the function is wrong,” but rather:

- it is too slow

A timing decorator can help you quickly find hotspots.

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

### 3.1 This kind of decorator is very useful in AI engineering

For example, you may want to quickly see:

- how long the tokenizer takes
- how long retrieval takes
- how long model inference takes

There is no need to write timing logic by hand in every function.

---

## 4. Decorators with parameters: why are they closer to real engineering?

Often, you do not just want to know whether a piece of logic is enabled,  
you also want to specify:

- how many retries to allow
- what permission level is required
- what the timeout threshold is

This is where parameterized decorators come in.

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

### 4.1 Why are there two layers of functions here?

Because:

- the first layer receives the decorator arguments
- the second layer receives the function to be decorated
- the third layer is the actual wrapper logic used at runtime

At first this may feel confusing,  
but once you remember the three layers of responsibility, it becomes much easier to follow.

---

## 5. Retry decorators: a very typical production code pattern

The example below simulates an unstable function  
and uses a decorator to centralize retry logic.

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

### 5.1 What is the engineering meaning of this code?

It shows that decorators are very suitable for packaging:

- retries
- rate limiting
- circuit breaking

These kinds of outer control logic.

### 5.2 Why should retry decorators be used carefully?

Because not all errors are suitable for retrying.  
For example:

- parameter errors
- permission errors

Retrying those only wastes resources.

---

## 6. Common pitfalls with decorators

### 6.1 Mistake 1: thinking “decorator” means “advanced” as soon as you see it

Decorators are not for showing off,  
but for reducing repeated logic.

### 6.2 Mistake 2: stacking too many decorators

If a function has too many decorators,  
debugging and understanding it both become harder.

### 6.3 Mistake 3: not using `wraps`

This causes metadata loss, which makes later troubleshooting painful.

---

## Summary

The most important thing in this lesson is not memorizing decorators as a syntax puzzle,  
but building an engineering judgment:

> **Decorators are best for packaging cross-cutting logic like logging, timing, retries, and permissions, so the business function itself stays cleaner.**

Once you establish this judgment,  
you will find it much easier to read framework code, service middleware, and library source code later.

---

## Exercises

1. Add a `label` parameter to `measure_time` and practice a parameterized decorator.
2. Think about it: what kinds of errors are suitable for a `retry` decorator, and what kinds are not?
3. Stack the logging, timing, and retry decorators on the same function and observe the execution order.
4. Explain in your own words: why are decorators especially suitable for “cross-cutting logic”?
