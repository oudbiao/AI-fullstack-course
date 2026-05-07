---
title: "E.B.4 Metaprogramming"
sidebar_position: 11
description: "Use practical metaprogramming patterns such as registries and descriptors without turning normal code into magic."
keywords: [metaprogramming, type, registry, descriptor, dynamic class, Python]
---

# E.B.4 Metaprogramming

![Python Metaprogramming Registry Map](/img/course/elective-metaprogramming-registry-map-en.png)

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

## Common Mistakes

- Using dynamic tricks just to look advanced.
- Hiding behavior so deeply that debugging becomes painful.
- Removing small, harmless repetition at the cost of readability.

## Practice

Add a `yaml` loader and confirm `sorted(REGISTRY)` includes it. Then create an `IntegerRange(min_value, max_value)` descriptor for a `retry_count` field.
