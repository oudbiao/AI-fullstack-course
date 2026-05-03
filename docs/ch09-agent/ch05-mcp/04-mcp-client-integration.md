---
title: "5.5 MCP Client Integration"
sidebar_position: 28
description: "From tool discovery and call dispatch to error handling and a minimal client implementation, understand how the client actually consumes the capabilities exposed by an MCP Server."
keywords: [MCP client, tool discovery, client integration, dispatch, protocol client]
---

# MCP Client Integration

:::tip Section Overview
So far, we have looked at MCP from the server side.
In this section, let’s switch perspectives and look at it from the client side:

> **How does the client discover, choose, and call the capabilities exposed by an MCP Server?**

This step is very important, because in real usage, the one that actually uses the tools is often not the server, but the client.
:::

## Learning Objectives

- Understand the core responsibilities of an MCP Client
- Learn to think about “discovering tools” and “calling tools” as two separate steps
- Understand a minimal MCP Client call flow
- Understand why the client side still needs selection strategies, failure handling, and caching

---

## 1. How are the responsibilities of Client and Server divided?

### 1.1 The Server provides capabilities

The Server is more like a “tool warehouse manager.” It is responsible for:

- Listing tools
- Exposing capabilities
- Executing calls

### 1.2 The Client consumes capabilities

The Client is more like the “person actually doing the work.” It is responsible for:

- Discovering tools
- Deciding which tool to call
- Organizing arguments
- Receiving results

So one very important point is:

> **An MCP Client is not a passive relay. It usually still has its own decision-making logic for calls.**

---

## 2. What should the Client learn first? Discover tools first

### 2.1 Why not hard-code everything?

If the client hard-codes all tools from the beginning:

- When server tools change, the code must change too
- If you switch to a different server, you must rewrite the client

That is the exact opposite of what MCP is trying to solve.

### 2.2 A minimal discovery example

```python
class MockMCPServer:
    def list_tools(self):
        return [
            {"name": "search_docs", "description": "Search course documents"},
            {"name": "get_weather", "description": "Query the weather"}
        ]

server = MockMCPServer()
tools = server.list_tools()

for tool in tools:
    print(tool)
```

### 2.3 What is this step teaching you?

It is teaching you this:

> The client must first know “what is available” before thinking about “how to use it.”

That is the value of the discovery phase.

---

## 3. After discovery, what else does the client need to do?

### 3.1 Choose a tool

Not every tool needs to be called.
The client usually needs to judge first:

- Whether the current problem needs a tool
- If it does, which tool to call

### 3.2 Organize arguments

Even if the correct tool is chosen, the arguments still need to be organized properly.

### 3.3 Handle errors

If:

- The server times out
- The tool does not exist
- Argument validation fails

the client cannot just crash. It also needs to decide:

- Whether to retry
- Whether to fall back
- Whether to switch tools

---

## 4. A minimal Client example

### 4.1 Runnable code

```python
class MockMCPServer:
    def list_tools(self):
        return [
            {"name": "search_docs", "description": "Search course documents"},
            {"name": "get_weather", "description": "Query the weather"}
        ]

    def call_tool(self, name, arguments):
        if name == "search_docs":
            return {"result": f"Search result: {arguments['query']}"}
        if name == "get_weather":
            return {"result": f"{arguments['city']} is sunny, 22 degrees right now"}
        return {"error": "unknown_tool"}

class MockMCPClient:
    def __init__(self, server):
        self.server = server
        self.tools = []

    def discover(self):
        self.tools = self.server.list_tools()
        return self.tools

    def call(self, name, arguments):
        return self.server.call_tool(name, arguments)

server = MockMCPServer()
client = MockMCPClient(server)

print(client.discover())
print(client.call("search_docs", {"query": "refund policy"}))
```

### 4.2 What is this code already showing?

It already shows the client’s two main functions:

1. Discovery
2. Calling

This is the minimal closed loop of an MCP Client.

---

## 5. The Client actually has a “strategy layer”

### 5.1 Why say the client is not just a protocol caller?

Because in real systems, the client often still needs to decide:

- Whether the current problem should go through MCP
- If so, which server / which tool has priority
- How to fall back after a failure

### 5.2 A simple tool selector

```python
def choose_tool(user_query, tools):
    tool_names = [t["name"] for t in tools]

    if "refund" in user_query and "search_docs" in tool_names:
        return {"name": "search_docs", "arguments": {"query": "refund policy"}}

    if "weather" in user_query and "get_weather" in tool_names:
        return {"name": "get_weather", "arguments": {"city": "Beijing"}}

    return None

tools = client.discover()
decision = choose_tool("What is the refund policy?", tools)
print(decision)
print(client.call(decision["name"], decision["arguments"]))
```

This shows that the client often also takes on a lightweight scheduling role.

---

## 6. Why is error handling especially important for the client?

### 6.1 Because the client is the first one to feel the failure

The server side may return:

- unknown_tool
- invalid_arguments
- timeout

And the client must decide what to do next.

### 6.2 A minimal error handling example

```python
def safe_call(client, name, arguments):
    result = client.call(name, arguments)
    if "error" in result:
        return {"ok": False, "fallback": "This tool is currently unavailable. Please try again later."}
    return {"ok": True, "data": result["result"]}

print(safe_call(client, "search_docs", {"query": "refund policy"}))
print(safe_call(client, "bad_tool", {}))
```

This step turns the system from:

- “Crash on the first error”

into:

- “Absorb errors gracefully”

---

## 7. Why do clients sometimes need caching too?

### 7.1 A very practical question

If you call `list_tools()` again for every request, wouldn’t that be wasteful?

In many cases:

- The tool list does not change that often
- Rediscovering every time adds latency

### 7.2 A minimal caching idea

```python
class CachedMCPClient(MockMCPClient):
    def discover_once(self):
        if not self.tools:
            self.tools = self.server.list_tools()
        return self.tools

cached_client = CachedMCPClient(server)
print(cached_client.discover_once())
print(cached_client.discover_once())
```

Although simple, it already shows:

> The client is not just a “relay”; it also has state and room for optimization.

---

## 8. Common pitfalls in client integration

### 8.1 Being able to call, but not choose

If the client does not do selection strategy, it is easy to end up with:

- Many tools, but no idea how to use them

### 8.2 Looking only at success, not at failure paths

Once the server fails, the system experience can suddenly get much worse.

### 8.3 Rediscovering tools every time

This may waste a lot of unnecessary overhead.

---

## Summary

The most important thing in this section is not to write a class that can call the server, but to understand:

> **The core of an MCP Client is not just “sending requests,” but turning “tool discovery, tool selection, argument organization, and result handling” into a stable consumption layer.**

The more mature the client is, the easier it becomes for the capabilities on the server side to be truly used by upper-layer systems.

---

## Exercises

1. Add a `read_file` tool to `MockMCPServer`, then extend the client’s selection logic.
2. Think about this: why are some systems suitable for rediscovering tools on every request, while others are better with caching?
3. Add a “retry once after failure” behavior to `safe_call()`.
4. Explain in your own words: why does an MCP Client usually also need a “strategy layer”?
