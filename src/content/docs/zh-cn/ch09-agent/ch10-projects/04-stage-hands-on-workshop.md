---
title: "9.10.5 实操：构建一个可追踪的单 Agent 助手"
description: "第 9 章分步骤实操工作坊：构建一个包含规划、工具 schema 校验、权限边界、JSONL trace、评估用例和可选 OpenAI Agents SDK 升级的可追踪单 Agent 助手。"
sidebar:
  order: 57
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Agent 实操, 单 Agent, Agent trace, 工具调用, tool schema, Agent 评估, OpenAI Agents SDK"
---

# 9.10.5 实操：构建一个可追踪的单 Agent 助手

这个工作坊把第 9 章的主线压缩成一个可以运行的小项目。你会做一个小型单 Agent 助手：它能读取目标、选择工具、校验工具参数、拦截高风险动作、把每一步写入 trace，并运行一组小评估。

第一版只使用 Python 标准库。这是有意设计的。上框架、API key、MCP Server 或多 Agent 编排之前，你应该先亲眼看懂最核心的执行循环。

## 你将做出什么

![Agent 工作坊单 Agent 执行循环图](/img/course/ch09-workshop-single-agent-loop-flow-map.webp)

你会做一个学习规划 Agent，包含这些能力：

| 能力 | 你要实现什么 | 为什么重要 |
|---|---|---|
| 目标输入 | 接受 “prepare an AgentOps review plan” 这类用户目标 | Agent 面向目标行动，不只是回答一次问题 |
| 规划器 | 根据当前状态决定下一步动作 | 规划是从目标走向工具调用的桥 |
| 工具结构约束 | 校验必填字段、类型和未知参数 | 糟糕结构约束会导致错误工具调用 |
| 权限门 | 没有批准时拦截模拟的 `publish_report` 动作 | 真实 Agent 不能静默执行高风险动作 |
| 追踪日志 | 保存 `thought`、`action`、`arguments`、`observation`、`next_decision` | 排障需要过程证据 |
| 评估 | 固定测试成功、审批拦截、无证据三类情况 | 只有一次成功演示不够 |

:::tip[学习节奏]
请按顺序完成：先看图，复制代码，运行，对比输出，检查 trace，再读解释。不要一开始就做多 Agent 框架。先把一个 Agent 做到可观察、可控制。
:::
## 步骤 0：写代码前先看懂 Agent 循环

![Agent 行动闭环与 追踪 图](/img/course/ch09-agent-action-loop-trace-map.webp)

Agent 不是“加了工具的聊天机器人”。在这个工作坊里，Agent 指的是：

1. 它有一个 **目标**。
2. 它维护已经找到什么的 **状态**。
3. 它选择 **下一步行动**。
4. 它用经过校验的参数调用 **工具**。
5. 它读取工具返回的 **观察结果**。
6. 它更新状态，写入追踪记录，并判断是否继续。

新人最重要的观念是：Agent 出错时，不要只看最终回答。先看追踪记录。追踪记录会告诉你问题来自规划、工具结构约束、权限、观察结果处理，还是停止条件。

## 步骤 1：创建一个小项目目录

打开终端运行：

```bash
mkdir ch09_agent_workshop
cd ch09_agent_workshop
touch agent_workshop.py
```

你只需要 Python 3.10 或更新版本。第一个脚本不需要任何第三方包。

## 步骤 2：复制完整离线 Agent 脚本

![Agent 工作坊工具 结构约束 与权限门](/img/course/ch09-workshop-tool-schema-permission-map.webp)

复制代码前，先把这张图当成安全检查表。工具调用不能从模型决策直接跳到执行，它要经过：

- 结构约束 校验：必填字段、类型检查、未知参数检查；
- 权限校验：`read_only`、`write_limited` 或禁用；
- 追踪 记录：成功和被拦截的动作都要记录。

把下面代码复制到 `agent_workshop.py`：

```python
import json
import re
from pathlib import Path
from typing import Any


COURSE_MATERIALS = [
    {
        "id": "agentops",
        "title": "AgentOps control loop",
        "source": "ch09-agent/index.md#agentops",
        "text": (
            "A reliable Agent keeps a trace of the goal, plan, tool call, observation, cost, "
            "failure, recovery action, and final result. Traceable systems are easier to debug and evaluate."
        ),
    },
    {
        "id": "tool-safety",
        "title": "Tool safety boundaries",
        "source": "ch09-agent/ch03-tools/05-tool-safety.md",
        "text": (
            "Tools should be whitelisted by risk level. Read-only tools can run directly, "
            "write-limited tools need approval, and destructive tools should stay disabled by default."
        ),
    },
    {
        "id": "memory",
        "title": "Agent memory engineering",
        "source": "ch09-agent/ch04-memory/05-memory-engineering.md",
        "text": (
            "Memory should be written with clear scope, confidence, source, and update policy. "
            "Do not mix short-term task state with long-term user preference memory."
        ),
    },
    {
        "id": "evaluation",
        "title": "Agent evaluation basics",
        "source": "ch09-agent/ch08-eval-safety/01-evaluation-methods.md",
        "text": (
            "An Agent evaluation set should check task completion, tool accuracy, step count, "
            "permission violations, recovery behavior, and final answer quality."
        ),
    },
]


TOOL_SPECS = {
    "search_course": {
        "description": "Search the local Chapter 9 course notes.",
        "required": {"query": str},
        "optional": {"top_k": int},
        "risk": "read_only",
    },
    "make_study_plan": {
        "description": "Create a short study plan from retrieved evidence.",
        "required": {"goal": str, "evidence_ids": list},
        "optional": {},
        "risk": "read_only",
    },
    "publish_report": {
        "description": "Pretend to publish a report. This is high risk in the workshop.",
        "required": {"title": str, "body": str},
        "optional": {},
        "risk": "write_limited",
    },
}


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "should", "the", "to", "with",
    "what", "today",
}


def normalize(text: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]+", text.lower()) if token not in STOPWORDS]


def score(query: str, document: dict[str, str]) -> int:
    query_terms = set(normalize(query))
    doc_terms = normalize(document["title"] + " " + document["text"])
    return sum(1 for term in doc_terms if term in query_terms)


def validate_args(tool_name: str, args: dict[str, Any]) -> list[str]:
    spec = TOOL_SPECS[tool_name]
    allowed = set(spec["required"]) | set(spec["optional"])
    errors = []
    for key in args:
        if key not in allowed:
            errors.append(f"unknown argument: {key}")
    for key, expected_type in spec["required"].items():
        if key not in args:
            errors.append(f"missing required argument: {key}")
        elif not isinstance(args[key], expected_type):
            errors.append(f"{key} must be {expected_type.__name__}")
    for key, expected_type in spec["optional"].items():
        if key in args and not isinstance(args[key], expected_type):
            errors.append(f"{key} must be {expected_type.__name__}")
    return errors


def search_course(query: str, top_k: int = 2) -> list[dict[str, Any]]:
    hits = []
    for doc in COURSE_MATERIALS:
        doc_score = score(query, doc)
        if doc_score > 0:
            hits.append(
                {
                    "id": doc["id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "score": doc_score,
                    "summary": doc["text"].split(".")[0] + ".",
                }
            )
    return sorted(hits, key=lambda item: (-item["score"], item["id"]))[:top_k]


def make_study_plan(goal: str, evidence_ids: list[str]) -> list[dict[str, str]]:
    evidence = [doc for doc in COURSE_MATERIALS if doc["id"] in evidence_ids]
    if not evidence:
        return []
    tasks = [
        {
            "step": "1",
            "task": f"Restate the goal: {goal}",
            "evidence": evidence[0]["source"],
        },
        {
            "step": "2",
            "task": f"Read {evidence[0]['title']} and write three checkpoints.",
            "evidence": evidence[0]["source"],
        },
    ]
    if len(evidence) > 1:
        tasks.append(
            {
                "step": "3",
                "task": f"Compare with {evidence[1]['title']} and add one safety note.",
                "evidence": evidence[1]["source"],
            }
        )
    return tasks


def publish_report(title: str, body: str) -> dict[str, str]:
    return {"status": "published", "title": title, "url": "https://example.local/draft-agent-report"}


def call_tool(tool_name: str, args: dict[str, Any], approved_tools: set[str]) -> dict[str, Any]:
    if tool_name not in TOOL_SPECS:
        return {"status": "tool_not_found", "error": f"Unknown tool: {tool_name}"}

    errors = validate_args(tool_name, args)
    if errors:
        return {"status": "validation_error", "errors": errors}

    risk = TOOL_SPECS[tool_name]["risk"]
    if risk != "read_only" and tool_name not in approved_tools:
        return {
            "status": "blocked_by_approval",
            "error": f"{tool_name} is {risk}; human approval is required before running it.",
        }

    if tool_name == "search_course":
        return {"status": "ok", "data": search_course(**args)}
    if tool_name == "make_study_plan":
        return {"status": "ok", "data": make_study_plan(**args)}
    if tool_name == "publish_report":
        return {"status": "ok", "data": publish_report(**args)}
    return {"status": "tool_not_found", "error": f"No implementation for {tool_name}"}


def choose_next_step(state: dict[str, Any]) -> dict[str, Any]:
    goal = state["goal"]
    if not state["searched"]:
        return {
            "thought": "I need evidence before planning, so I will search the course notes first.",
            "action": "search_course",
            "arguments": {"query": goal, "top_k": 2},
        }
    if not state["evidence"]:
        return {"thought": "No permitted evidence was found, so I should stop.", "action": "finish", "arguments": {}}
    if not state["plan"]:
        return {
            "thought": "I have evidence, so I can create a short study plan with citations.",
            "action": "make_study_plan",
            "arguments": {"goal": goal, "evidence_ids": [item["id"] for item in state["evidence"]]},
        }
    if "publish" in goal.lower() and not state["publish_checked"]:
        return {
            "thought": "The user asked to publish, but publishing is a higher-risk action.",
            "action": "publish_report",
            "arguments": {"title": "Agent workshop plan", "body": json.dumps(state["plan"], ensure_ascii=False)},
        }
    return {"thought": "The goal is satisfied and no more tool calls are needed.", "action": "finish", "arguments": {}}


def append_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_agent(goal: str, approved_tools: set[str] | None = None, run_id: str = "demo", write_log: bool = True) -> dict[str, Any]:
    approved_tools = approved_tools or set()
    trace_path = Path("logs/agent_traces.jsonl")
    state = {"goal": goal, "searched": False, "evidence": [], "plan": [], "publish_checked": False}
    trace = []
    final = {"message": "No result yet."}
    status = "stopped_max_steps"

    for step_number in range(1, 6):
        decision = choose_next_step(state)
        action = decision["action"]
        if action == "finish":
            status = "completed" if state["plan"] else "no_evidence"
            final = {"message": "Done.", "tasks": state["plan"], "sources": [item["source"] for item in state["evidence"]]}
            break

        observation = call_tool(action, decision["arguments"], approved_tools)
        next_decision = "continue"

        if action == "search_course":
            state["searched"] = True
            state["evidence"] = observation.get("data", []) if observation["status"] == "ok" else []
            if not state["evidence"]:
                status = "no_evidence"
                final = {"message": "No course evidence matched the goal.", "tasks": [], "sources": []}
                next_decision = "stop_no_evidence"
        elif action == "make_study_plan" and observation["status"] == "ok":
            state["plan"] = observation["data"]
            next_decision = "continue_to_finish_or_publish"
        elif action == "publish_report":
            state["publish_checked"] = True
            if observation["status"] == "blocked_by_approval":
                status = "blocked_by_approval"
                final = {"message": observation["error"], "tasks": state["plan"], "sources": [item["source"] for item in state["evidence"]]}
                next_decision = "stop_for_human_approval"
            elif observation["status"] == "ok":
                status = "completed"
                final = {"message": "Published after approval.", "publication": observation["data"], "tasks": state["plan"]}
                next_decision = "finish"
        elif observation["status"] != "ok":
            status = observation["status"]
            final = {"message": observation.get("error", "Tool failed."), "tasks": state["plan"]}
            next_decision = "stop_on_tool_error"

        trace.append(
            {
                "run_id": run_id,
                "step": step_number,
                "thought": decision["thought"],
                "action": action,
                "arguments": decision["arguments"],
                "observation": observation,
                "next_decision": next_decision,
            }
        )
        if next_decision.startswith("stop") or next_decision == "finish":
            break

    if write_log:
        append_trace(trace_path, trace)
    return {"goal": goal, "status": status, "final": final, "trace": trace, "trace_file": str(trace_path)}


EVAL_CASES = [
    {
        "name": "safe_learning_plan",
        "goal": "Prepare a two-day review plan for AgentOps and tool safety",
        "expected_status": "completed",
    },
    {
        "name": "publish_without_approval",
        "goal": "Publish the AgentOps review plan to the class page",
        "expected_status": "blocked_by_approval",
    },
    {
        "name": "unknown_topic",
        "goal": "What is the cafeteria menu today?",
        "expected_status": "no_evidence",
    },
]


def evaluate() -> tuple[int, list[dict[str, Any]]]:
    rows = []
    passed = 0
    for case in EVAL_CASES:
        result = run_agent(case["goal"], run_id=f"eval-{case['name']}", write_log=False)
        ok = result["status"] == case["expected_status"]
        passed += int(ok)
        rows.append({"name": case["name"], "ok": ok, "status": result["status"]})
    return passed, rows


def main() -> None:
    trace_path = Path("logs/agent_traces.jsonl")
    if trace_path.exists():
        trace_path.unlink()

    print("STEP 1: run a safe learning-planning Agent")
    safe = run_agent("Prepare a two-day review plan for AgentOps and tool safety", run_id="demo-safe")
    print(f"status: {safe['status']}")
    print(f"final_tasks: {len(safe['final']['tasks'])}")
    print(f"first_action: {safe['trace'][0]['action']}")
    print(f"trace_file: {safe['trace_file']}")
    print()

    print("STEP 2: high-risk action is blocked")
    risky = run_agent("Publish the AgentOps review plan to the class page", run_id="demo-risk")
    print(f"status: {risky['status']}")
    print(f"blocked_tool: {risky['trace'][-1]['action']}")
    print()

    print("STEP 3: mini evaluation")
    passed, rows = evaluate()
    for row in rows:
        mark = "PASS" if row["ok"] else "FAIL"
        print(f"{row['name']}: {mark} ({row['status']})")
    print(f"passed: {passed}/{len(rows)}")


if __name__ == "__main__":
    main()
```

## 步骤 3：运行并对比输出

运行：

```bash
python3 agent_workshop.py
```

预期输出：

```text
STEP 1: run a safe learning-planning Agent
status: completed
final_tasks: 3
first_action: search_course
trace_file: logs/agent_traces.jsonl

STEP 2: high-risk action is blocked
status: blocked_by_approval
blocked_tool: publish_report

STEP 3: mini evaluation
safe_learning_plan: PASS (completed)
publish_without_approval: PASS (blocked_by_approval)
unknown_topic: PASS (no_evidence)
passed: 3/3
```

![Agent 工作坊运行 追踪 与评测结果图](/img/course/ch09-agent-workshop-run-trace-eval-result-map.webp)

如果你的输出一致，说明你已经完成第 9 章最小闭环：Agent 接收目标、检索证据、生成计划、拦截不安全发布动作、写入 trace 日志，并用固定用例评估。

## 步骤 4：检查 追踪 文件

![Agent 工作坊 追踪 JSONL 复盘图](/img/course/ch09-workshop-trace-jsonl-replay-map.webp)

运行：

```bash
head -n 1 logs/agent_traces.jsonl | python3 -m json.tool
```

你应该看到类似这样的记录：

```json
{
  "run_id": "demo-safe",
  "step": 1,
  "thought": "I need evidence before planning, so I will search the course notes first.",
  "action": "search_course",
  "arguments": {
    "query": "Prepare a two-day review plan for AgentOps and tool safety",
    "top_k": 2
  },
  "observation": {
    "status": "ok",
    "data": [
      {
        "id": "agentops",
        "title": "AgentOps control loop",
        "source": "ch09-agent/index.md#agentops",
        "score": 6,
        "summary": "A reliable Agent keeps a trace of the goal, plan, tool call, observation, cost, failure, recovery action, and final result."
      }
    ]
  },
  "next_decision": "continue"
}
```

如果你修改了资料，`score` 和返回数据可能不同，但关键字段应该稳定：`run_id`、`step`、`thought`、`action`、`arguments`、`observation` 和 `next_decision`。

## 步骤 5：像 Agent 流水线一样读代码

按这个顺序读脚本：

| 代码位置 | 要看什么 | 新手解释 |
|---|---|---|
| `COURSE_MATERIALS` | `id`、`source`、`text` | Agent 可以检索的小知识库 |
| `TOOL_SPECS` | `required`、`optional`、`risk` | 规划者 和工具之间的契约 |
| `validate_args()` | 缺字段、类型错误、未知字段 | 避免工具调用变成模糊猜测 |
| `call_tool()` | 执行前先做 结构约束 和权限检查 | 副作用之前先做安全 |
| `choose_next_step()` | search、plan、publish、finish 分支 | 这是一个极小 规划者 |
| `run_agent()` | 状态、追踪 行、停止行为 | Agent 循环在这里 |
| `EVAL_CASES` | 固定任务的预期状态 | 评估把演示变成可重复检查 |

这个脚本故意做成确定性流程。它暂时不追求“聪明”，而是训练控制骨架：以后换成更强模型或框架，也不应该丢掉这个骨架。

## 步骤 6：理解权限分支

![工具安全权限、沙箱与审计图](/img/course/ch09-tool-safety-permission-sandbox-map.webp)

关键安全行为在这里：

```python
risk = TOOL_SPECS[tool_name]["risk"]
if risk != "read_only" and tool_name not in approved_tools:
    return {
        "status": "blocked_by_approval",
        "error": f"{tool_name} is {risk}; human approval is required before running it.",
    }
```

`search_course` 和 `make_study_plan` 是 `read_only`，可以直接运行。`publish_report` 是 `write_limited`，除非显式传入批准，否则会被拦截。

试着在 `main()` 里做这个小改动：

```python
risky = run_agent(
    "Publish the AgentOps review plan to the class page",
    approved_tools={"publish_report"},
    run_id="demo-risk-approved",
)
```

再次运行脚本。高风险用例应该从 `blocked_by_approval` 变成 `completed`。这条规则很重要：不要为了让演示通过而删除安全检查，而是在合适的地方加入明确批准。

## 步骤 7：把评估当成计分卡读

![Agent 工作坊评估计分卡图](/img/course/ch09-workshop-evaluation-scorecard-map.webp)

`EVAL_CASES` 检查三种行为：

| 用例 | 预期行为 | 为什么重要 |
|---|---|---|
| `safe_learning_plan` | `completed` | 正常路径必须可用 |
| `publish_without_approval` | `blocked_by_approval` | 安全 Agent 会拦截高风险动作 |
| `unknown_topic` | `no_evidence` | 缺少证据时 Agent 应该停止 |

以后你改进 Agent 时，请保留这些用例。新增用例，而不是替换旧用例。回归测试能防止 Agent 项目变成一次性表演。

## 步骤 8：练习任务

按顺序完成：

| 难度 | 任务 | 通过标准 |
|---|---|---|
| 入门 | 新增一条 MCP 相关课程资料 | 包含 `MCP` 的目标能检索到它并引用来源 |
| 标准 | 给 `run_agent()` 新增 `max_steps` 参数 | 循环规划器会以 `stopped_max_steps` 停止 |
| 标准 | 在 `EVAL_CASES` 里加入 `validation_error` | 错误参数类型会在执行前被抓住 |
| 挑战 | 按 `run_id` 分文件保存 追踪 | 不用读完整日志也能复盘单次运行 |
| 挑战 | 用模型调用替换 `choose_next_step()` | 现有评估用例仍然通过 |

<details>
<summary>操作参考与检查点</summary>

1. MCP 材料任务可以新增一条包含 title、body、source id 的小型资料，然后验证包含 `MCP` 的 goal 能检索到它，并在最终答案中引用来源。
2. `max_steps` 应在下一次规划会超过限制前停止，返回 `stopped_max_steps`，并把停止原因写进 trace。
3. `validation_error` 用例应传入错误参数类型，并期待 dispatcher 在 tool execution 前拒绝它。
4. 按 `run_id` 保存 trace 时，可以把每次运行写入单独文件或分区路径，然后只读取该文件就能 replay 单次运行。
5. 如果用模型调用替换 `choose_next_step()`，现有安全和评估用例必须继续通过。只有在行为变好且 guardrails 不丢失时，升级才算成功。

</details>

## 步骤 9：可选 OpenAI Agents SDK 升级

![Agent 框架选型决策图](/img/course/ch09-framework-selection-decision-map.webp)

离线脚本是必做的新手路径。跑通之后，可以把手写 planner 换成当前 OpenAI Agents SDK。官方 quickstart 使用 `pip install openai-agents` 安装包，并以 `Agent`、`Runner` 和 `@function_tool` 作为核心构件。

安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install openai-agents
export OPENAI_API_KEY="your_api_key_here"
```

创建 `agent_sdk_upgrade.py`：

```python
from agents import Agent, Runner, function_tool


@function_tool
def search_course(query: str) -> str:
    """Search a tiny Chapter 9 note set."""
    if "AgentOps" in query or "tool safety" in query:
        return "AgentOps needs trace logs; tool safety needs risk levels and human approval."
    return "No matching course evidence."


agent = Agent(
    name="Traceable Study Agent",
    instructions=(
        "Help the learner make a short Chapter 9 study plan. "
        "Use search_course before answering. "
        "If the evidence is missing, say that the evidence is missing."
    ),
    tools=[search_course],
)

result = Runner.run_sync(agent, "Prepare a two-day review plan for AgentOps and tool safety")
print(result.final_output)
```

运行：

```bash
python3 agent_sdk_upgrade.py
```

真实项目里，请保留离线脚本训练出的习惯：

- 清楚定义工具结构约束；
- 记录工具调用和观察结果；
- 没有人类批准时拦截高风险工具；
- 保留能检查成功和安全边界的评估用例。

## 工作坊通关标准

![Agent 项目交付闭环图](/img/course/ch09-project-delivery-loop.webp)

当你能做到下面这些，就算完成本工作坊：

- 运行 `python3 agent_workshop.py` 并得到预期输出。
- 能解释目标、状态、工具结构约束、观察结果、追踪记录、审批和评估集。
- 能说明为什么 `publish_report` 没有批准会被拦截。
- 能检查 `logs/agent_traces.jsonl`，并解释其中一步。
- 能新增一个工具或评估用例，并且不破坏现有用例。
- 能说明 OpenAI Agents SDK 替换了哪一部分，哪些责任仍然属于你的应用代码。

请把这个小项目保留下来，作为第 9 章 baseline。后面遇到 MCP、LangGraph、CrewAI、AutoGen、部署或多 Agent 系统时，都可以回头对照这个脚本：框架替换了哪一部分，哪些边界仍然属于你自己？

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
项目目标：智能体应完成什么，以及必须不做什么
基线：在加入高级功能前的单智能体循环
追踪包：目标、计划、tool 调用、观察、记忆、评估
失败日志：一次失败或不安全的运行及其根因
交付物：README、运行命令、trace 截图/日志、下一步
```
