---
title: "5.3 MCP Server 开发"
sidebar_position: 27
description: "从工具描述、参数校验、结果返回到最小 server 结构，理解一个 MCP Server 到底应该怎样暴露能力。"
keywords: [MCP server, tool server, schema, tool exposure, server development]
---

# MCP Server 开发

:::tip 本节定位
前两节我们已经知道：

- MCP 要解决什么问题
- MCP 架构里 client 和 server 各自负责什么

这一节开始真正落地 server 视角，回答：

> **如果我要自己写一个 MCP Server，我到底应该从哪里开始？**
:::

## 学习目标

- 理解 MCP Server 的最小职责边界
- 学会定义工具描述、参数结构和调用入口
- 理解为什么 server 开发的重点是“暴露能力”，而不是“把业务逻辑写死”
- 看懂一个最小可运行的 Mock MCP Server

---

## 一、MCP Server 真正在做什么？

### 1.1 它不是“另一个普通后端”

普通后端往往直接面向业务接口。  
而 MCP Server 更像：

> **把已有能力整理成一组可被 client 发现和调用的工具。**

所以它的核心关注点通常是：

- 有哪些工具
- 工具怎样描述
- 参数怎样校验
- 结果怎样统一返回

### 1.2 一个直觉类比

MCP Server 很像一个有前台的工具库管理员：

- 客户端来问“你这里有什么工具”
- Server 列出能力清单
- 客户端再说“我要用哪个”
- Server 按约定执行并返回结果

这和“直接把所有业务函数散着写”非常不一样。

---

## 二、先定义一个最小工具

### 2.1 一个工具最少得有哪几样？

至少要有：

- 名称
- 描述
- 参数说明
- 实际执行逻辑

### 2.2 一个最小工具描述示例

```python
search_docs_tool = {
    "name": "search_docs",
    "description": "搜索课程文档并返回相关内容",
    "parameters": {
        "query": {
            "type": "string",
            "description": "要搜索的关键词"
        }
    },
    "required": ["query"]
}

print(search_docs_tool)
```

你可以把这个结构理解成：

> 工具的对外说明书。 

---

## 三、工具描述为什么不能写得太随意？

### 3.1 一个坏描述

```python
bad_tool = {
    "name": "search",
    "description": "做搜索",
    "parameters": {"q": {"type": "string"}}
}

print(bad_tool)
```

问题在于：

- 名字太模糊
- 描述太空
- 参数含义不清楚

### 3.2 一个更稳的描述

```python
good_tool = {
    "name": "search_course_docs",
    "description": "搜索课程 FAQ、政策和学习路线文档",
    "parameters": {
        "query": {
            "type": "string",
            "description": "用户要查询的主题，比如 退款政策 或 证书"
        }
    },
    "required": ["query"]
}

print(good_tool)
```

这里更好的地方在于：

- 工具边界更清楚
- 参数语义更清楚
- client 更容易正确使用

---

## 四、Server 的最小两项能力：列工具 + 调工具

一个最小可用的 MCP Server，通常至少要能：

1. 列出可用工具
2. 接受某个工具调用

### 4.1 先写一个最小 Server

```python
class MockMCPServer:
    def __init__(self):
        self.tool_specs = [
            {
                "name": "search_docs",
                "description": "搜索课程文档",
                "parameters": {
                    "query": {"type": "string"}
                }
            }
        ]

    def list_tools(self):
        return self.tool_specs

server = MockMCPServer()
print(server.list_tools())
```

### 4.2 再加真正的执行逻辑

```python
class MockMCPServer:
    def __init__(self):
        self.kb = {
            "退款": "课程购买后 7 天内且学习进度低于 20% 可退款。",
            "证书": "完成所有项目并通过测试后可获得证书。"
        }

        self.tool_specs = [
            {
                "name": "search_docs",
                "description": "搜索课程文档",
                "parameters": {
                    "query": {"type": "string"}
                }
            }
        ]

    def list_tools(self):
        return self.tool_specs

    def call_tool(self, name, arguments):
        if name != "search_docs":
            return {"error": "unknown_tool"}

        query = arguments.get("query", "")
        for key, value in self.kb.items():
            if key in query:
                return {"result": value}
        return {"result": "未找到相关文档"}

server = MockMCPServer()
print(server.call_tool("search_docs", {"query": "退款政策是什么"}))
```

这已经是一个非常清楚的最小 server 骨架了。

---

## 五、参数校验为什么是 server 的责任之一？

### 5.1 因为 client 或模型都可能给错参数

例如：

```python
bad_call = {"query_text": "退款政策"}
```

如果 server 直接执行，就可能报错或产生奇怪行为。

### 5.2 一个最小校验版本

```python
def validate_search_docs(arguments):
    if "query" not in arguments:
        return False, "missing_query"
    if not isinstance(arguments["query"], str):
        return False, "query_must_be_string"
    return True, "ok"

print(validate_search_docs({"query": "退款政策"}))
print(validate_search_docs({"query_text": "退款政策"}))
```

### 5.3 为什么这一步一定不能省？

因为 server 是能力边界守门人。  
如果 server 不校验，整个工具系统就很难稳定。

---

## 六、一个更完整的最小 Server 版本

```python
class BetterMCPServer:
    def __init__(self):
        self.kb = {
            "退款": "课程购买后 7 天内且学习进度低于 20% 可退款。",
            "证书": "完成所有项目并通过测试后可获得证书。"
        }

    def list_tools(self):
        return [
            {
                "name": "search_docs",
                "description": "搜索课程文档",
                "parameters": {
                    "query": {"type": "string"}
                }
            }
        ]

    def validate(self, name, arguments):
        if name != "search_docs":
            return False, "unknown_tool"
        if "query" not in arguments:
            return False, "missing_query"
        if not isinstance(arguments["query"], str):
            return False, "query_must_be_string"
        return True, "ok"

    def call_tool(self, name, arguments):
        ok, msg = self.validate(name, arguments)
        if not ok:
            return {"error": msg}

        query = arguments["query"]
        for key, value in self.kb.items():
            if key in query:
                return {"result": value}
        return {"result": "未找到相关文档"}

server = BetterMCPServer()
print(server.list_tools())
print(server.call_tool("search_docs", {"query": "证书怎么获得"}))
print(server.call_tool("search_docs", {"wrong": "证书怎么获得"}))
```

### 6.2 这个版本比上一版强在哪？

它已经具备了：

- 工具列出
- 参数校验
- 统一调用入口
- 统一错误返回

这已经非常接近真实工程里 server 的核心职责。

---

## 七、MCP Server 开发里最常见的坑

### 7.1 把业务逻辑和协议逻辑混在一起

结果会变成：

- 工具描述不清
- 扩展困难
- 调试困难

### 7.2 工具粒度太粗或太细

- 太粗：一个工具什么都干
- 太细：client 调用复杂度爆炸

### 7.3 返回结构不统一

有时返回文本，有时返回 dict，有时直接抛异常，后面会很难接。

---

## 八、怎么判断一个 MCP Server 设计得够不够好？

可以先问四个问题：

1. client 能不能清楚知道有哪些工具
2. 参数要求是不是明确
3. 错误返回是不是统一
4. 加新工具时结构会不会越来越乱

如果这四个问题都答得比较稳，server 设计通常就已经不错了。

---

## 九、小结

这一节最重要的不是“把一个类写出来”，而是理解：

> **MCP Server 的本质，是把一组可执行能力，用清晰可发现、可校验、可调用的方式暴露出来。**

server 做得越清楚，client 侧越容易扩展，整个工具生态也越容易做大。

---

## 练习

1. 给 `BetterMCPServer` 再增加一个 `get_weather(city)` 工具。
2. 为这个新工具补上参数校验逻辑。
3. 想一想：工具粒度太粗和太细，各自会带来什么问题？
4. 用自己的话解释：为什么说 MCP Server 开发的核心不只是“执行工具”，更是“暴露清晰边界”？
