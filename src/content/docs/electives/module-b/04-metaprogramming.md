---
title: "E.B.4 Metaprogramming"
description: "Use practical metaprogramming patterns such as registries and descriptors without turning normal code into magic."
sidebar:
  order: 11
head:
  - tag: meta
    attrs:
      name: keywords
      content: "metaprogramming, type, registry, descriptor, dynamic class, Python"
---
![Python Metaprogramming Registry Map](/img/course/elective-metaprogramming-registry-map-en.webp)

Metaprogramming means using code to organize or generate code structure. In day-to-day Python engineering, the most useful version is usually not clever magic; it is automatic registration, field validation, and reducing repeated boilerplate.

## What You Need

- Python 3.10+
- No external packages
- Comfort with classes

## Key Terms

- **Registry**: a mapping that remembers available implementations.
- **Decorator registration**: using a decorator to add a class or function to a registry.
- **Descriptor**: an object that controls attribute read/write behavior.
- **`__set_name__`**: lets a descriptor know which attribute name it was assigned to.
- **Dynamic class**: a class created at runtime, often with `type`.

## Run A Registry And Descriptor Example

Create `metaprogramming_demo.py`:

```python
REGISTRY = {}


def register(name):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls

    return decorator


@register("csv")
class CsvLoader:
    def load(self):
        return "csv rows"


@register("json")
class JsonLoader:
    def load(self):
        return "json rows"


class NonEmpty:
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)

    def __set__(self, instance, value):
        if not value:
            raise ValueError("name cannot be empty")
        setattr(instance, self.private_name, value)


class JobConfig:
    name = NonEmpty()


loader = REGISTRY["json"]()
print(loader.load())
print(sorted(REGISTRY))

config = JobConfig()
config.name = "daily-import"
print(config.name)

try:
    config.name = ""
except ValueError as error:
    print("error:", error)
```

Run it:

```bash
python metaprogramming_demo.py
```

Expected output:

```text
json rows
['csv', 'json']
daily-import
error: name cannot be empty
```

The registry removes a manual mapping table. The descriptor keeps validation next to the field definition.

## When It Is Worth Using

Metaprogramming is useful when:

1. Many classes need automatic registration.
2. A framework needs to discover plugins.
3. Many fields share the same validation behavior.
4. Configuration should generate repeated structure.

Avoid it when a normal class or dictionary is clearer.

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

- Using dynamic tricks just to look advanced.
- Hiding behavior so deeply that debugging becomes painful.
- Removing small, harmless repetition at the cost of readability.

## Practice

Add a `yaml` loader and confirm `sorted(REGISTRY)` includes it. Then create an `IntegerRange(min_value, max_value)` descriptor for a `retry_count` field.

<details>
<summary>Reference implementation and walkthrough</summary>

The `yaml` loader should register itself through the same mechanism as the existing loaders, so `sorted(REGISTRY)` includes `yaml` without manually editing a separate mapping table.

For `IntegerRange`, the descriptor should reject non-integers and values outside the allowed range. A useful self-check sets `retry_count = 3`, then tries `retry_count = -1` or `"3"` and confirms a clear exception appears. The point is not cleverness; it is keeping repeated validation rules close to the field definition.

</details>
