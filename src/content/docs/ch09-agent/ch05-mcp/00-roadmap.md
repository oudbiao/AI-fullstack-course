---
title: "9.5.1 MCP Roadmap: Server, Client, Capability"
description: "A concise hands-on roadmap for MCP: understand the protocol layer, server/client responsibilities, tools, resources, prompts, and safe ecosystem integration."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "MCP guide, Model Context Protocol, Agent tool ecosystem, MCP Server"
---

# 9.5.1 MCP Roadmap: Server, Client, Capability

MCP is a protocol layer for connecting tools, resources, and prompt templates to model applications in a more standard way. It does not replace Agents or tools; it makes capabilities easier to expose and use consistently.

## See the MCP Boundary First

![MCP Host Client Server architecture diagram](/img/course/mcp-host-client-server-en.webp)

![MCP chapter learning order diagram](/img/course/ch09-mcp-chapter-flow-en.webp)

![MCP capability access bridge diagram](/img/course/ch09-mcp-capability-bridge-en.webp)

Function Calling focuses on structured calls. MCP focuses on how external capabilities are discovered, described, called, and governed through a protocol.

## Run a Capability Registry Check

Before implementing a real MCP Server, list what it exposes and what the Client may call.

```python
server = {
    "tools": ["search_docs"],
    "resources": ["course://ch09-agent"],
    "prompts": ["study_plan"],
}

client_request = "search_docs"

print("server_ready:", all(server.values()))
print("can_call:", client_request in server["tools"])
print("boundary:", "server exposes, client calls")
```

Expected output:

```text
server_ready: True
can_call: True
boundary: server exposes, client calls
```

If the boundary is vague, permissions and debugging will be vague too.

## Learn in This Order

| Step | Read | Practice Output |
|---|---|---|
| 1 | MCP concept | Explain why a protocol layer reduces integration mess |
| 2 | MCP architecture | Distinguish Host, Client, Server, tools, resources, prompts |
| 3 | Server development | Wrap one capability with clear input, output, and errors |
| 4 | Client integration | Discover and call server capabilities safely |
| 5 | Ecosystem | Connect MCP to IDEs, databases, browsers, knowledge bases, and Agents |

## Evidence to Keep

Keep this page's proof of learning as a small evidence card:

```text
capability: resource, prompt, or tool exposed by server
contract: schema, transport, permissions, and error shape
call_trace: discovery, invocation, response, and failure handling
failure_check: incompatible schema, missing auth, unsafe tool, or server error
integration_action: validate server contract before adding autonomy
```

## Pass Check

You pass this chapter when you can draw the Host-Client-Server relationship and explain what the Server exposes, what the Client calls, and where permissions are checked.

The exit mini project is a course-materials MCP Server design: one search tool, one resource URI pattern, one prompt template, and one failure-handling rule.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer describes the agent loop: goal, plan, tool call, observation, memory or state update, and stop condition.
2. The evidence should include a trace that another developer can inspect, not only the final answer.
3. A good self-check names one safety or reliability control such as tool schemas, permission boundaries, retries, evaluation cases, or a human-review point.

</details>
