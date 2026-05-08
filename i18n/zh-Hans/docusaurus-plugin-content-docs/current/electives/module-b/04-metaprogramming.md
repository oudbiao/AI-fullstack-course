---
title: "E.B.4 元编程"
sidebar_position: 11
description: "用注册表和描述符这类实用元编程模式减少重复，但不要把普通代码写成魔法。"
keywords: [metaprogramming, type, registry, descriptor, dynamic class, Python]
---

# E.B.4 元编程

![Python 元编程注册表图](/img/course/elective-metaprogramming-registry-map.webp)

元编程是用代码组织或生成代码结构。日常 Python 工程里最有用的元编程，通常不是炫技，而是自动注册、字段校验和减少重复模板。

## 准备内容

- Python 3.10+
- 不需要第三方包
- 熟悉类的基本写法

## 关键术语

- **Registry（注册表）**：记录可用实现的映射表。
- **装饰器注册**：用装饰器把类或函数加入注册表。
- **Descriptor（描述符）**：控制属性读取和写入行为的对象。
- **`__set_name__`**：让描述符知道自己被赋给了哪个属性名。
- **动态类**：运行时创建的类，常见方式是 `type`。

## 运行注册表和描述符示例

创建 `metaprogramming_demo.py`：

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

运行：

```bash
python metaprogramming_demo.py
```

预期输出：

```text
json rows
['csv', 'json']
daily-import
error: name cannot be empty
```

注册表省掉了手写映射表。描述符把字段校验放在字段定义附近。

## 什么时候值得用

适合：

1. 很多类需要自动注册。
2. 框架要发现插件。
3. 很多字段共享同一类校验行为。
4. 配置需要生成重复结构。

如果普通类或普通字典更清楚，就不要强行元编程。

## 常见错误

- 为了显得高级而使用动态技巧。
- 把行为藏得太深，导致调试痛苦。
- 为了消除一点点重复，牺牲整体可读性。

## 练习

加一个 `yaml` loader，确认 `sorted(REGISTRY)` 里包含它。然后为 `retry_count` 字段创建一个 `IntegerRange(min_value, max_value)` 描述符。
