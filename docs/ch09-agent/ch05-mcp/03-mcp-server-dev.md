---
title: "9.5.4 MCP Server Development"
sidebar_position: 27
description: "From tool descriptions and parameter validation to result return and the minimal server structure, understand how an MCP Server should expose capabilities."
keywords: [MCP server, tool server, schema, tool exposure, server development]
---

# 9.5.4 MCP Server Development

:::tip Section overview
In the previous two sections, we already learned:

- What problem MCP is trying to solve
- What client and server are respectively responsible for in the MCP architecture

From this section on, we move into the real server-side implementation and answer:

> **If I want to write my own MCP Server, where should I start?**
:::

## Learning objectives

- Understand the minimum responsibility boundary of an MCP Server
- Learn how to define tool descriptions, parameter structures, and invocation entry points
- Understand why the focus of server development is “exposing capabilities,” not “hard-coding business logic”
- Read and understand a minimal runnable Mock MCP Server

---

## What is an MCP Server really doing?

### It is not “just another regular backend”

A regular backend usually exposes business APIs directly.
An MCP Server is more like:

> **Organizing existing capabilities into a set of tools that can be discovered and called by the client.**

So its core concerns are usually:

- What tools are available
- How each tool is described
- How parameters are validated
- How results are returned in a consistent way

### An intuitive analogy

An MCP Server is a bit like a tool library manager with a front desk:

- The client asks, “What tools do you have here?”
- The server lists its capability inventory
- The client then says, “Which one should I use?”
- The server executes according to the contract and returns the result

This is very different from “just writing all the business functions loosely scattered around.”

---

## First define a minimal tool

### What does a tool minimally need?

At minimum, it should have:

- A name
- A description
- Parameter specifications
- Actual execution logic

### A minimal tool description example

```python
search_docs_tool = {
    "name": "search_docs",
    "description": "Search course documents and return relevant content",
    "parameters": {
        "query": {
            "type": "string",
            "description": "The keyword to search for"
        }
    },
    "required": ["query"]
}

print(search_docs_tool)
```

Expected output:

```text
{'name': 'search_docs', 'description': 'Search course documents and return relevant content', 'parameters': {'query': {'type': 'string', 'description': 'The keyword to search for'}}, 'required': ['query']}
```

You can think of this structure as:

> The public-facing instruction manual for the tool.

---

## Why can’t tool descriptions be too casual?

### A bad description

```python
bad_tool = {
    "name": "search",
    "description": "Do search",
    "parameters": {"q": {"type": "string"}}
}

print(bad_tool)
```

Expected output:

```text
{'name': 'search', 'description': 'Do search', 'parameters': {'q': {'type': 'string'}}}
```

The problems are:

- The name is too vague
- The description is too empty
- The meaning of the parameter is unclear

### A more reliable description

```python
good_tool = {
    "name": "search_course_docs",
    "description": "Search course FAQ, policies, and learning path documents",
    "parameters": {
        "query": {
            "type": "string",
            "description": "The topic the user wants to query, such as refund policy or certificate"
        }
    },
    "required": ["query"]
}

print(good_tool)
```

Expected output:

```text
{'name': 'search_course_docs', 'description': 'Search course FAQ, policies, and learning path documents', 'parameters': {'query': {'type': 'string', 'description': 'The topic the user wants to query, such as refund policy or certificate'}}, 'required': ['query']}
```

What is better here:

- The tool boundary is clearer
- The parameter semantics are clearer
- The client is more likely to use it correctly

---

## The two minimum capabilities of a Server: list tools + call tools

A minimal usable MCP Server usually needs to be able to:

1. List available tools
2. Accept a tool invocation

### First, write a minimal Server

```python
class MockMCPServer:
    def __init__(self):
        self.tool_specs = [
            {
                "name": "search_docs",
                "description": "Search course documents",
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

Expected output:

```text
[{'name': 'search_docs', 'description': 'Search course documents', 'parameters': {'query': {'type': 'string'}}}]
```

### Then add real execution logic

```python
class MockMCPServer:
    def __init__(self):
        self.kb = {
            "refund": "You can request a refund within 7 days after purchase if your learning progress is below 20%.",
            "certificate": "You can receive a certificate after completing all projects and passing the tests."
        }

        self.tool_specs = [
            {
                "name": "search_docs",
                "description": "Search course documents",
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
        return {"result": "No relevant documents found"}

server = MockMCPServer()
print(server.call_tool("search_docs", {"query": "What is the refund policy?"}))
```

Expected output:

```text
{'result': 'You can request a refund within 7 days after purchase if your learning progress is below 20%.'}
```

This is already a very clear minimal server skeleton.

---

## Why is parameter validation one of the server’s responsibilities?

### Because the client or the model may pass incorrect parameters

For example:

```python
bad_call = {"query_text": "refund policy"}
```

If the server executes this directly, it may crash or behave strangely.

### A minimal validation version

```python
def validate_search_docs(arguments):
    if "query" not in arguments:
        return False, "missing_query"
    if not isinstance(arguments["query"], str):
        return False, "query_must_be_string"
    return True, "ok"

print(validate_search_docs({"query": "refund policy"}))
print(validate_search_docs({"query_text": "refund policy"}))
```

Expected output:

```text
(True, 'ok')
(False, 'missing_query')
```

### Why can’t we skip this step?

Because the server is the gatekeeper of the capability boundary.
If the server does not validate, the entire tool system becomes hard to keep stable.

![MCP Server Tool Contract Diagram](/img/course/ch09-mcp-server-tool-contract-map-en.webp)

:::tip Reading guide
Think of the MCP Server as the gatekeeper of the tool contract: it not only exposes list_tools, but also validates call_tool parameters, executes the real logic, standardizes returned results, and turns errors into structures the client can understand.
:::

---

## A more complete minimal Server version

```python
class BetterMCPServer:
    def __init__(self):
        self.kb = {
            "refund": "You can request a refund within 7 days after purchase if your learning progress is below 20%.",
            "certificate": "You can receive a certificate after completing all projects and passing the tests."
        }

    def list_tools(self):
        return [
            {
                "name": "search_docs",
                "description": "Search course documents",
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
        return {"result": "No relevant documents found"}

server = BetterMCPServer()
print(server.list_tools())
print(server.call_tool("search_docs", {"query": "How do I get a certificate?"}))
print(server.call_tool("search_docs", {"wrong": "How do I get a certificate?"}))
```

Expected output:

```text
[{'name': 'search_docs', 'description': 'Search course documents', 'parameters': {'query': {'type': 'string'}}}]
{'result': 'You can receive a certificate after completing all projects and passing the tests.'}
{'error': 'missing_query'}
```

![MCP Server validation result map](/img/course/ch09-mcp-server-validation-result-map-en.webp)

### What is better about this version than the previous one?

It already has:

- Tool listing
- Parameter validation
- A unified invocation entry point
- Unified error returns

This is already very close to the core responsibilities of a server in real-world engineering.

---

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
capability: resource, prompt, or tool exposed by server
contract: schema, transport, permissions, and error shape
call_trace: discovery, invocation, response, and failure handling
failure_check: incompatible schema, missing auth, unsafe tool, or server error
integration_action: validate server contract before adding autonomy
```

## The most common pitfalls in MCP Server development

### Mixing business logic with protocol logic

This leads to:

- Unclear tool descriptions
- Harder extension
- Harder debugging

### Tool granularity that is too coarse or too fine

- Too coarse: one tool does everything
- Too fine: client invocation complexity explodes

### Inconsistent return structures

Sometimes returning text, sometimes dicts, sometimes raising exceptions directly makes future integration very difficult.

---

## How do you know whether an MCP Server design is good enough?

You can start by asking four questions:

1. Can the client clearly know what tools are available?
2. Are the parameter requirements explicit?
3. Are error returns consistent?
4. Will the structure become messy when adding new tools?

If you can answer all four questions confidently, the server design is usually already pretty good.

---

## Summary

The most important thing in this section is not “writing a class,” but understanding:

> **The essence of an MCP Server is to expose a set of executable capabilities in a way that is clear to discover, validate, and call.**

The clearer the server is, the easier it is for the client side to expand, and the easier it is to grow the whole tool ecosystem.

---

## Exercises

1. Add a new `get_weather(city)` tool to `BetterMCPServer`.
2. Add parameter validation logic for this new tool.
3. Think about it: what problems do too coarse and too fine tool granularity each cause?
4. Explain in your own words: why is the core of MCP Server development not just “executing tools,” but “exposing clear boundaries”?
