---
title: "9.2.4 ReAct 框架"
sidebar_position: 7
description: "从 Thought-Action-Observation 循环讲起，理解 ReAct 为什么能把推理和工具调用交织起来，以及它最适合哪类 Agent 任务。"
keywords: [ReAct, thought action observation, tool use, agent loop, reasoning and acting]
---

# 9.2.4 ReAct 框架

![ReAct 推理行动观察循环图](/img/course/react-reason-act-observe-loop.webp)

:::tip 本节定位
CoT 解决的是：

- 先拆步骤再回答

但 Agent 常常还会遇到另一类问题：

- 只靠脑内推导不够
- 必须去外部查、算、搜、看

这时系统需要的不只是“思考”，还要“行动”。
ReAct 的核心就是把两件事交织起来：

> **先想下一步该做什么，再调工具拿观察，再根据观察继续想。**
:::

## 学习目标

- 理解 ReAct 的核心循环：`Thought -> Action -> Observation`
- 理解它和纯 CoT 的差别
- 通过可运行示例看懂一个最小 ReAct agent loop
- 理解 ReAct 最适合什么问题，什么情况下会变得笨重

---

## 为什么只靠“想”还不够？

### 因为很多答案不在模型脑子里

例如：

- 今天北京天气怎么样？
- 某个订单现在是什么状态？
- 这两个数精确相加是多少？

这些问题都依赖：

- 实时外部信息
- 精确工具能力

如果模型只靠自己“猜”，
就会出现：

- 幻觉
- 过度自信
- 算错

### ReAct 的本质：边想边拿新信息

它的典型循环是：

1. `Thought`
   我现在缺什么信息？
2. `Action`
   我该调用哪个工具？
3. `Observation`
   工具返回了什么？
4. 再进入下一轮思考

这让 Agent 不再只是“脑补答案”，
而是可以逐步靠近真实环境。

### 一个类比：像做调查，而不是闭门写作

纯 CoT 更像在草稿纸上推题。
ReAct 更像做调查：

- 先想应该去查什么
- 去拿证据
- 再根据证据继续判断

---

## ReAct 和 CoT 的根本差别

### CoT 偏“内部推导”

核心问题是：

- 如何拆步骤
- 如何保持中间状态

### ReAct 偏“推导 + 外部交互”

它额外多了一层：

- 什么时候该向外部要信息

所以 ReAct 更像：

- CoT + Tool Loop

### 为什么这对 Agent 特别关键？

因为 Agent 不只是做静态问答。
它经常要：

- 查知识库
- 调数据库
- 算数
- 执行命令

这些都要求它在思考过程中不断接入外部世界。

---

## 先跑一个真正的 ReAct 最小闭环

下面这个例子会模拟一个小型电商助手。
用户问：

- 退款规则是什么？
- 订单金额 `299 + 15` 最终会退多少？

Agent 需要：

1. 先查退款政策
2. 再调用计算器
3. 最后整合出答案

```python
import ast
import operator

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_calculate(expression):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError("unsupported_expression")

    return visit(ast.parse(expression, mode="eval"))


def search_policy(topic):
    policies = {
        "refund": "未发货订单可直接申请退款，款项原路返回，通常 3 到 7 个工作日到账。",
    }
    return policies.get(topic, "未找到相关政策。")


def calculator(expression):
    return str(safe_calculate(expression))


def policy(state):
    trace = state["trace"]
    question = state["question"]

    if not any(item["action"] == "search_policy" for item in trace):
        return {
            "thought": "我需要先确认退款政策，再回答规则部分。",
            "action": "search_policy",
            "args": {"topic": "refund"},
        }

    if not any(item["action"] == "calculator" for item in trace):
        return {
            "thought": "我已经知道政策了，接下来计算退款金额 299 + 15。",
            "action": "calculator",
            "args": {"expression": "299 + 15"},
        }

    policy_text = next(item["observation"] for item in trace if item["action"] == "search_policy")
    amount = next(item["observation"] for item in trace if item["action"] == "calculator")

    return {
        "thought": "信息已经足够，可以给出最终回答。",
        "action": None,
        "answer": f"{policy_text} 本单预计退款金额为 {amount} 元。",
    }


TOOLS = {
    "search_policy": search_policy,
    "calculator": calculator,
}


def run_react(question, max_steps=5):
    state = {"question": question, "trace": []}

    for _ in range(max_steps):
        decision = policy(state)

        if decision["action"] is None:
            return state["trace"], decision["answer"]

        tool_name = decision["action"]
        observation = TOOLS[tool_name](**decision["args"])

        state["trace"].append(
            {
                "thought": decision["thought"],
                "action": tool_name,
                "args": decision["args"],
                "observation": observation,
            }
        )

    return state["trace"], "达到最大步数，未能完成任务。"


trace, answer = run_react("退款规则是什么？订单金额 299 + 15 最终会退多少？")

print("trace:")
for item in trace:
    print(item)
print("\nfinal answer:")
print(answer)
```

预期输出：

```text
trace:
{'thought': '我需要先确认退款政策，再回答规则部分。', 'action': 'search_policy', 'args': {'topic': 'refund'}, 'observation': '未发货订单可直接申请退款，款项原路返回，通常 3 到 7 个工作日到账。'}
{'thought': '我已经知道政策了，接下来计算退款金额 299 + 15。', 'action': 'calculator', 'args': {'expression': '299 + 15'}, 'observation': '314'}

final answer:
未发货订单可直接申请退款，款项原路返回，通常 3 到 7 个工作日到账。 本单预计退款金额为 314 元。
```

![ReAct 退款工具 trace 结果图](/img/course/ch09-react-refund-tool-trace-result-map.webp)

:::tip 读图提示
把图当成一份 trace 账本来读：第一次 observation 给出退款政策，第二次 observation 给出计算金额，最终回答要整合两者，而不是只复读最后一个工具结果。
:::

### 这段代码最应该怎么读？

建议按这个顺序：

1. 先看 `policy`
   理解 agent 每轮如何决定“下一步”
2. 再看 `TOOLS`
   理解外部能力从哪来
3. 最后看 `run_react`
   理解完整循环如何把 trace 逐步积累起来

### `trace` 为什么这么重要？

因为 ReAct 不是一次出答案，
而是逐步推进。

有了 trace，你才能知道：

- 想了什么
- 调了什么
- 看到了什么
- 为什么最后会给出这个答案

这对调试非常关键。

### 为什么 ReAct 往往比“直接一次调用工具”更强？

因为真实问题经常不是一步完成。
工具调用顺序可能依赖前一步结果。

例如这里：

- 要先确认政策
- 再算金额
- 再组织回答

这就是 ReAct 最擅长的结构。

---

## ReAct 什么时候最好用？

### 任务需要多轮观察

例如：

- 先搜再算
- 先查再比
- 先看状态再决定下一步

### 工具调用顺序不是固定死的

如果每个任务都严格是：

1. 查 A
2. 查 B
3. 输出

那普通 workflow 也许就够了。

ReAct 更适合：

- 当前一步结果会影响下一步选择

### 你需要过程可追踪

因为 ReAct 天然有：

- thought
- action
- observation

这让它很适合做：

- 调试
- 回放
- 错误分析

---

## ReAct 最常见的问题是什么？

### 循环太长

如果 agent 老是在：

- 想
- 调
- 再想
- 再调

就会出现：

- 慢
- 贵
- 容易跑偏

### 工具选错

ReAct 不保证每轮都选对工具。
它可能会：

- 查错知识源
- 重复调用
- 调一个其实没必要的工具

### Observation 整合失败

即使工具返回了对的信息，
agent 也可能：

- 忽略关键字段
- 误读结果
- 最后整合错

这说明 ReAct 的难点不只是“有没有工具”，
还有“能不能读懂工具输出”。

---

## 工程上怎么让 ReAct 更稳？

### 让 action schema 足够清楚

工具描述越清楚，
agent 越不容易乱调。

### 限制最大步数

避免无意义循环的一个最简单办法就是：

- 明确 `max_steps`

### 对 observation 做结构化

如果工具返回的是乱糟糟的一大段自然语言，
agent 更容易误读。

更稳的方式通常是：

- 返回结构化字段

例如：

- `{"refund_days": "3-7", "channel": "original_payment"}`

---

## 常见误区

### 误区一：ReAct 就是“会调用工具”

不够准确。
ReAct 的关键是：

- 推理和行动交替推进

### 误区二：只要有 trace，就一定可靠

trace 可追踪，但不自动保证正确。

### 误区三：所有 Agent 都应该用 ReAct

不一定。
如果流程高度固定，
显式工作流可能更简单、更稳。

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
task_goal: what the agent is trying to solve
plan_or_trace: reasoning steps, plan, ReAct trace, or execution graph
observation: what changed after each action
failure_check: hallucinated step, stale observation, loop, or unverified conclusion
eval_action: compare against expected result and revise the plan
```

## 小结

这节最重要的不是把 `ReAct` 当成一个流行名词，
而是理解它为什么重要：

> **当任务需要边思考边向外部世界拿信息时，ReAct 能把“推理”和“行动”组织成一条逐步收集证据、逐步靠近答案的循环。**

只要这层理解清楚了，
你后面再看更复杂的 Agent 轨迹、工具策略和多步执行框架，就会更顺。

---

## 练习

1. 给示例再加一个工具，例如 `check_order_status`，让 agent 多一步判断。
2. 为什么说 ReAct 更适合“下一步动作依赖上一轮 observation”的任务？
3. 如果工具输出很乱，ReAct 为什么更容易出错？
4. 想一个更适合固定 workflow、而不太适合 ReAct 的任务。
