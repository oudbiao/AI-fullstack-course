---
title: "3.8 实战：多工具协作 Agent"
sidebar_position: 18
description: "把工具发现、策略、安全和多步推理串成一个完整实践，做一个能处理退款工单的多工具 Agent。"
keywords: [multi-tool agent, orchestration, tool chain, agent practice, refund assistant]
---

# 实战：多工具协作 Agent

:::tip 本节定位
前面几节我们已经分别讲了：

- 工具 schema
- 调用策略
- 常见工具
- 安全与高级模式

这一节要把它们真正串起来。  
我们不再只讲某一个工具，而是做一个完整的小型 Agent：

> **用户提交退款工单后，Agent 先查订单状态，再查政策，再计算金额，最后给出可执行答复。**

这就是一个典型的多工具协作任务。
:::

## 学习目标

- 理解多工具 Agent 和单工具 Agent 的主要差别
- 看懂一个完整的“发现 -> 选择 -> 执行 -> 整合 -> 输出”闭环
- 理解多工具协作里状态管理为什么是关键
- 学会用最小项目方式展示一个多工具 Agent

---

## 一、多工具协作难在哪里？

### 1.1 难点不只是“工具更多了”

真正困难的地方通常有三层：

1. 先后顺序
2. 中间状态传递
3. 错误后的处理

例如退款场景里：

- 不知道订单状态，政策判断就可能错
- 不知道订单金额，计算退款额就没法做
- 工具失败后，最终答复也必须改变

### 1.2 一个类比：像接力赛而不是单人跑

单工具任务像一个人直接完成动作。  
多工具任务像接力赛：

- 前一棒的结果要交给下一棒
- 某一棒掉棒，后面都受影响

### 1.3 所以多工具系统最怕“状态散掉”

如果每一轮都不清楚当前已经知道什么，  
系统就很容易：

- 重复调用
- 漏关键信息
- 最后整合错

---

## 二、这个实战例子要解决什么问题？

我们做一个最小但完整的退款工单助手。  
用户问题是：

- 我的订单还能退款吗？
- 预计退多少钱？
- 多久到账？

这个任务至少要用到三类工具：

1. `get_order_status`
2. `search_refund_policy`
3. `calculator`

而且它们之间有明显顺序：

- 先看订单状态
- 再匹配政策
- 再算金额

---

## 三、先跑一个完整闭环示例

下面这段代码会完整展示：

1. 工具注册
2. 状态跟踪
3. 决策策略
4. 多轮执行
5. 最终回答

```python
TOOLS = {
    "get_order_status": lambda order_id: {
        "order_id": order_id,
        "status": "未发货",
        "amount": 299,
        "shipping_fee": 15,
    },
    "search_refund_policy": lambda keyword: {
        "policy_text": "未发货订单可直接申请退款，款项原路返回，通常 3 到 7 个工作日到账。"
    },
    "calculator": lambda expression: {
        "result": eval(expression, {"__builtins__": {}}, {})
    },
}


def decide_next_action(state):
    if "order_info" not in state:
        return {"tool": "get_order_status", "arguments": {"order_id": state["order_id"]}}

    if "policy" not in state:
        return {"tool": "search_refund_policy", "arguments": {"keyword": "退款"}}

    if "refund_amount" not in state:
        order = state["order_info"]
        expression = f"{order['amount']} + {order['shipping_fee']}"
        return {"tool": "calculator", "arguments": {"expression": expression}}

    return None


def apply_observation(state, tool_name, observation):
    if tool_name == "get_order_status":
        state["order_info"] = observation
    elif tool_name == "search_refund_policy":
        state["policy"] = observation["policy_text"]
    elif tool_name == "calculator":
        state["refund_amount"] = observation["result"]


def build_final_answer(state):
    order = state["order_info"]
    if order["status"] != "未发货":
        return "该订单当前不满足直接退款条件，请联系人工客服进一步处理。"

    return (
        f"订单 {state['order_id']} 当前状态为{order['status']}。"
        f"{state['policy']} "
        f"预计退款金额为 {state['refund_amount']} 元。"
    )


def run_agent(order_id, max_steps=5):
    state = {"order_id": order_id, "trace": []}

    for _ in range(max_steps):
        decision = decide_next_action(state)
        if decision is None:
            return state["trace"], build_final_answer(state)

        tool_name = decision["tool"]
        observation = TOOLS[tool_name](**decision["arguments"])

        state["trace"].append(
            {
                "tool": tool_name,
                "arguments": decision["arguments"],
                "observation": observation,
            }
        )

        apply_observation(state, tool_name, observation)

    return state["trace"], "达到最大步数，任务未完成。"


trace, answer = run_agent("ORD-1001")
print("trace:")
for item in trace:
    print(item)
print("\nanswer:")
print(answer)
```

### 3.1 这段代码和前面分散示例最大的差别是什么？

它已经不再是：

- 单一工具 demo

而是完整表现出：

- 决策顺序
- 状态累积
- 多工具配合
- 最终整合

也就是说，它已经接近一个真正的多工具 Agent 骨架。

### 3.2 为什么 `state` 这么关键？

因为每次工具调用后，系统都要知道：

- 现在已经知道了什么
- 还缺什么
- 下一步该补哪块信息

如果没有统一状态，  
多工具协作几乎一定会乱。

### 3.3 为什么最终回答不是直接拿最后一个 observation？

因为多工具系统的目标，通常不是原样转述某次工具输出。  
它真正要做的是：

- 把多个 observation 整合成用户可理解的结论

这正是 Agent 层的价值。

---

## 四、这类系统最容易失败在哪？

### 4.1 工具顺序错

例如还没查订单状态，  
就先查退款金额或直接给结论。

### 4.2 中间状态没保存

会导致：

- 重复查同一工具
- 结果覆盖错
- 后面步骤用不到前面结果

### 4.3 某个工具失败后，系统还假装继续成功

这是多工具系统里很危险的一类 bug。  
例如：

- 政策没查到
- 但系统还是编了一个退款规则

所以失败路径也必须是设计的一部分。

---

## 五、怎样把这个 demo 进一步做成作品？

### 5.1 第一步：让工具更真实

例如把：

- mock 订单状态
- mock 政策文档

换成：

- 数据库查询
- 文档检索

### 5.2 第二步：加失败处理

例如：

- 工具超时
- 订单不存在
- 政策未命中

系统都应有明确退路。

### 5.3 第三步：加入评估集

你可以准备：

- 可退款订单
- 不可退款订单
- 金额边界样例
- 工具失败样例

这样系统就不只是“能跑”，  
而是“能测”。

### 5.4 第四步：把 trace 可视化

如果你把工具调用轨迹展示出来，  
这个项目会非常适合做作品集演示。

---

## 六、常见误区

### 6.1 误区一：多工具就是把多个函数按顺序连起来

不够。  
真正难的是：

- 顺序判断
- 状态传递
- 失败恢复

### 6.2 误区二：工具越多，Agent 就越强

工具变多只会让：

- 选择难度
- 状态管理复杂度

一起上升。

### 6.3 误区三：最终答得像人就说明系统好

多工具系统更该看：

- trace 是否合理
- 工具是否必要
- observation 是否被正确整合

---

## 小结

这节最重要的，不是做出一个“会连续调三个函数”的 demo，  
而是建立一个多工具 Agent 的核心认识：

> **多工具协作的本质，是围绕共享状态把多个外部能力按正确顺序组织起来，并在失败和不确定时保持系统可控。**

只要这层理解稳了，  
你后面做更复杂的：

- 企业助手
- 研究 Agent
- 代码 Agent

都会知道问题真正难在哪。

---

## 练习

1. 给示例再加一个 `notify_user` 工具，只有在退款条件成立时才发送通知。
2. 为什么说多工具 Agent 的核心不是“工具多”，而是“状态管理稳”？
3. 如果 `search_refund_policy` 返回空结果，你会怎么改这套流程？
4. 想一想：这个 demo 里哪些部分最适合拿去做作品集展示？
