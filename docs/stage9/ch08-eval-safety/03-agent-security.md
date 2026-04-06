---
title: "8.4 Agent 安全与对齐"
sidebar_position: 46
description: "从越权、提示注入、工具误用到状态污染，理解 Agent 安全为什么比普通聊天系统更复杂。"
keywords: [agent security, alignment, prompt injection, tool misuse, permissions]
---

# Agent 安全与对齐

:::tip 本节定位
普通聊天系统主要风险常常停留在：

- 说错

而 Agent 风险更进一步：

- 做错

因为一旦接入工具和状态，错误会从文本层升级到动作层。  
这就是为什么 Agent 安全会比纯聊天复杂很多。
:::

## 学习目标

- 理解 Agent 和普通聊天系统在风险上的关键差异
- 学会按攻击面拆解 Agent 安全问题
- 理解提示注入、越权调用、状态污染等典型问题
- 建立“安全是系统设计问题”的意识

---

## 先建立一张地图

Agent 安全这节最适合新人的理解顺序不是“只盯 Prompt Injection”，而是先看清攻击面：

```mermaid
flowchart LR
    A["输入层"] --> B["提示注入"]
    A2["工具层"] --> B2["越权调用 / 参数误用"]
    A3["状态层"] --> B3["状态污染"]
    A4["权限层"] --> B4["过度授权"]
```

所以这节真正想解决的是：

- Agent 风险为什么会从“说错”升级到“做错”
- 为什么安全必须从系统攻击面整体去看

### 一个更适合新人的总类比

你可以把 Agent 安全理解成：

- 给一个会动手做事的助理设边界

普通聊天系统更像：

- 助理只能开口说话

Agent 更像：

- 助理还能查系统、调工具、改状态

这时风险就不只是：

- 说错一句话

而是：

- 真把事情做坏了

## 一、Agent 安全为什么更复杂？

因为 Agent 常常会：

- 调工具
- 读写状态
- 接触外部系统

这意味着失败不只是：

- 回答不准

还可能是：

- 调错接口
- 泄露数据
- 写坏状态

---

## 二、典型攻击面有哪些？

### 1. Prompt Injection

通过输入诱导系统违背原规则。

### 2. Tool Misuse

诱导调用不该调用的工具。

### 3. State Pollution

把恶意内容写进长期记忆或会话状态。

### 4. Over-Permission

系统权限过大，导致一旦出错后果更严重。

### 2.1 一个很适合初学者先记的风险表

| 风险面 | 最值得先问什么 |
|---|---|
| Prompt Injection | 输入会不会诱导系统违背原规则 |
| Tool Misuse | 工具会不会被调错或被恶意利用 |
| State Pollution | 恶意内容会不会进入长期状态 |
| Over-Permission | 一旦出错，最大后果会不会太大 |

这个表很适合新人，因为它会把“Agent 安全”从一个笼统词重新拆成几个清楚的攻击面。

---

## 三、先看一个最小权限分级示例

```python
tool_permissions = {
    "search_docs": "low",
    "get_user_profile": "medium",
    "delete_account": "high",
}


def can_execute(tool_name, user_role):
    if tool_permissions[tool_name] == "high" and user_role != "admin":
        return False
    return True


print(can_execute("search_docs", "guest"))
print(can_execute("delete_account", "guest"))
print(can_execute("delete_account", "admin"))
```

### 3.1 这个例子最想表达什么？

Agent 安全的核心不是只靠模型“懂事”，  
还要靠系统层明确：

- 哪些动作谁可以做

### 3.2 一个新人最该先做的风险盘点

如果你正在做一个 Agent，第一步往往不是先加规则，  
而是先列清楚：

- 它能读什么
- 它能写什么
- 它能调用哪些高风险工具
- 它会不会把内容写进长期状态

这份清单本身就非常有价值。

### 3.3 再看一个最小“动作确认”示例

```python
def require_confirmation(tool_name, risk_level):
    if risk_level == "high":
        return {"allow": False, "next_step": "human_confirmation"}
    return {"allow": True, "next_step": "execute"}


print(require_confirmation("delete_account", "high"))
print(require_confirmation("search_docs", "low"))
```

这个示例很适合初学者，因为它会帮助你看到：

- Agent 安全不只是“能不能做”
- 还包括“什么时候必须先停下来确认”

## 四、一个新人可直接照抄的安全排查顺序

更稳的顺序通常是：

1. 先列出所有工具
2. 再给工具分权限等级
3. 再检查哪些状态会被写入
4. 最后再补提示注入和输出层规则

这样会比只在 Prompt 上打补丁更有效。

## 如果把它做成项目或系统设计，最值得展示什么

最值得展示的通常不是：

- “我们有安全机制”

而是：

1. 工具权限分级表
2. 高风险动作的确认链路
3. 状态写入边界
4. 输入攻击和工具误用分别怎么防

这样别人会更容易看出：

- 你理解的是 Agent 的系统攻击面
- 不只是加了几条文本过滤规则

---

## 五、最常见误区

### 1. 只做输出过滤，不管工具权限

### 2. 以为安全只是 prompt 问题

### 3. 忽略状态和记忆层

## 六、什么时候“安全问题”其实是架构问题？

很常见的是下面这些情况：

- 工具权限没有分级
- 所有工具都对所有请求开放
- 状态写入没有边界
- 高风险动作没有确认机制

这时候问题已经不是“模型回答得对不对”，而是系统设计本身太松。

---

## 小结

这节最重要的是建立一个判断：

> **Agent 安全的复杂度来自工具、状态和权限的加入，因此它必须按系统攻击面整体设计，而不是只在输出层补丁式修补。**

## 这节最该带走什么

- Agent 安全是系统攻击面问题，不只是文本安全问题
- 工具、权限、状态和输入都要一起看
- 先收权限，再收行为，往往比单纯输出过滤更稳

---

## 练习

1. 列出你当前 Agent 里最危险的三个工具。
2. 为什么说状态污染也是安全问题？
3. 如果一个工具权限过大，最坏后果会是什么？
4. 想一想：为什么 Agent 安全不等于普通聊天安全的放大版？
