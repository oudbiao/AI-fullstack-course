---
title: "4.7 实战：完整记忆系统"
sidebar_position: 24
description: "做一个可运行的多层记忆 Agent：短期窗口、长期偏好、情景记录与流程记忆协同工作，并展示查询、写入、压缩和回复生成闭环。"
keywords: [memory practice, short term, long term, episodic memory, procedural memory, agent]
---

# 实战：完整记忆系统

:::tip 本节定位
前几节我们把记忆拆成了概念和策略。  
这一节直接做一个能跑的“小系统”，把这些层串起来：

- 短期记忆：最近对话和当前状态
- 长期记忆：用户偏好和稳定信息
- 情景记忆：一次次任务经历
- 程序记忆：固定工作流步骤

目标不是做大而全，而是先把“完整记忆闭环”跑通。
:::

## 学习目标

- 学会把多层记忆放进同一套 Agent 状态机
- 学会设计“什么时候写入哪层记忆”的规则
- 学会让记忆真正参与回答，而不是只做存档
- 通过一个可运行示例建立可复用的项目骨架

---

## 一、我们要做的系统长什么样？

### 1.1 目标场景

我们继续沿用售后助手场景。  
用户会连续问：

- 退款条件
- 退款进度
- 回答风格要求

系统要做到两件事：

1. 当前会话内保持连贯
2. 下次再来还能记住用户偏好

### 1.2 四层记忆分工

这个示例里我们这样分工：

- `short_term`  
  最近 N 轮消息 + 当前任务状态
- `long_term`  
  用户长期偏好
- `episodic`  
  每次处理任务后的总结条目
- `procedural`  
  预定义流程模板，例如退款处理步骤

### 1.3 评估目标

这个实战示例最重要的检查点是：

- 能否正确写入偏好
- 能否在后续回答时引用偏好
- 能否留下可检索的情景记录
- 能否在回答前引用程序记忆流程

---

## 二、先跑一个完整可执行版本

下面代码会模拟两轮对话：

1. 第一轮用户提出“请简洁回答”并问退款条件
2. 第二轮用户再问进度，系统要自动沿用简洁风格

它会打印：

- 回复结果
- 四层记忆快照

```python
from collections import deque
from dataclasses import dataclass, asdict


def get_refund_policy():
    return "退款规则：购买后7天内且学习进度低于20%可申请退款，款项原路返回，通常3-7个工作日到账。"


def get_order_status(order_id):
    mock = {
        "ORD-1001": {"status": "未发货", "progress": 0.12, "amount": 299},
        "ORD-1002": {"status": "已发货", "progress": 0.35, "amount": 499},
    }
    return mock.get(order_id, {"status": "未知", "progress": None, "amount": None})


@dataclass
class Episode:
    user_id: str
    topic: str
    summary: str


class MemoryAgent:
    def __init__(self, short_window=4):
        self.short_term_messages = deque(maxlen=short_window)
        self.short_term_state = {}
        self.long_term_profile = {}
        self.episodic_memory = []
        self.procedural_memory = {
            "refund_workflow": [
                "读取订单状态",
                "读取退款政策",
                "判断是否满足条件",
                "返回结论和到账说明",
            ]
        }

    def _remember_short(self, role, content):
        self.short_term_messages.append({"role": role, "content": content})

    def _update_profile(self, user_id, message):
        if "简洁" in message:
            self.long_term_profile.setdefault(user_id, {})["style"] = "concise"
        if "详细" in message:
            self.long_term_profile.setdefault(user_id, {})["style"] = "detailed"

    def _style_for_user(self, user_id):
        return self.long_term_profile.get(user_id, {}).get("style", "default")

    def _format_answer(self, text, style):
        if style == "concise":
            return text[:70] + ("..." if len(text) > 70 else "")
        if style == "detailed":
            return text + " 若你愿意，我可以再补充具体操作步骤和常见失败原因。"
        return text

    def _write_episode(self, user_id, topic, summary):
        self.episodic_memory.append(Episode(user_id=user_id, topic=topic, summary=summary))

    def handle(self, user_id, user_message, order_id):
        self._remember_short("user", user_message)
        self._update_profile(user_id, user_message)

        self.short_term_state["active_workflow"] = "refund_workflow"
        self.short_term_state["order_id"] = order_id

        workflow = self.procedural_memory["refund_workflow"]
        order_info = get_order_status(order_id)
        policy = get_refund_policy()

        if order_info["status"] == "未知":
            answer = "我暂时查不到该订单，请确认订单号后重试。"
        elif order_info["progress"] is not None and order_info["progress"] < 0.2:
            answer = (
                f"订单 {order_id} 当前学习进度为 {order_info['progress']*100:.0f}%，"
                f"符合退款进度条件。{policy}"
            )
        else:
            answer = (
                f"订单 {order_id} 当前学习进度为 {order_info['progress']*100:.0f}%，"
                "已超过退款进度阈值，当前不满足直接退款条件。"
            )

        style = self._style_for_user(user_id)
        final_answer = self._format_answer(answer, style)
        self._remember_short("assistant", final_answer)

        self._write_episode(
            user_id=user_id,
            topic="refund",
            summary=f"workflow={workflow}; order={order_id}; style={style}; result={final_answer}",
        )

        return final_answer

    def snapshot(self, user_id):
        return {
            "short_term_messages": list(self.short_term_messages),
            "short_term_state": dict(self.short_term_state),
            "long_term_profile": self.long_term_profile.get(user_id, {}),
            "episodic_memory_tail": [asdict(x) for x in self.episodic_memory[-2:]],
            "procedural_memory": self.procedural_memory,
        }


agent = MemoryAgent(short_window=4)

print("round1:")
print(agent.handle("u_001", "请简洁回答，我想看退款条件", "ORD-1001"))
print("\nround2:")
print(agent.handle("u_001", "那多久到账？", "ORD-1001"))

print("\nmemory snapshot:")
print(agent.snapshot("u_001"))
```

### 2.1 这段代码体现了哪四层协作？

1. `short_term_messages`  
   保留近期对话
2. `long_term_profile`  
   记住用户风格偏好
3. `episodic_memory`  
   每次完成任务后落一条“经历记录”
4. `procedural_memory`  
   定义退款任务流程模板

这四层都被用到了，不再是“讲概念但没运行”。

### 2.2 为什么第二轮还能保持简洁风格？

因为第一轮用户说了“请简洁回答”，  
系统把它写入了长期偏好：

- `long_term_profile["u_001"]["style"] = "concise"`

所以第二轮即使用户没再重复，回复也会继续沿用。

### 2.3 情景记忆在这里有什么价值？

每轮处理完成后，系统都会写一条 episode summary。  
这让我们后续可以回答：

- 用户之前经历过哪些退款判断
- 当时依据是什么

这对复盘和解释很有用。

---

## 三、这个系统还可以怎样扩展？

### 3.1 给长期记忆加“可信度”和“更新时间”

避免很旧、低可信的信息持续影响回答。

### 3.2 给情景记忆加检索

例如按 topic 和关键词查过去经历，  
给复杂问题提供历史参照。

### 3.3 给程序记忆做版本化

流程一旦改变，可追踪：

- 哪次对话用的是哪版流程

这对审计和回放很重要。

---

## 四、实战里最容易踩的坑

### 4.1 把所有信息都当长期记忆写入

结果会变成：

- 检索噪声越来越高

### 4.2 没有“写入门槛”

例如用户随口一句话就写长期，  
系统很容易学到错误偏好。

### 4.3 只存记忆，不让记忆参与决策

这样系统看起来“有记忆”，  
但实际上回答没有任何变化。

---

## 五、小结

这节最重要的是把“完整记忆系统”落成一个可执行闭环：

> **短期维持当前任务，长期保留稳定偏好，情景沉淀历史经历，程序记住可复用流程。**

当这四层协同起来，Agent 才能从“一次性问答器”变成“持续可用的任务系统”。

---

## 练习

1. 给示例加一个 `user_blacklist_topic` 长期偏好，看系统能否在回答中规避不相关话题。
2. 让 `episodic_memory` 支持按 `topic` 检索最近一条记录。
3. 把 `procedural_memory` 改成多流程版本，例如 `refund_workflow` 和 `invoice_workflow`。
4. 想一想：哪些信息最适合只放短期、不放长期？
