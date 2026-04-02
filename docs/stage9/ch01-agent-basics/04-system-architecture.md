---
title: "1.4 Agent 系统架构"
sidebar_position: 4
description: "从 Planner、Tool、Memory、State、Guardrails 等组件出发，理解一个可落地 Agent 系统的基本架构。"
keywords: [agent architecture, planner, tools, memory, state, guardrails, observability]
---

# Agent 系统架构

## 学习目标

完成本节后，你将能够：

- 说清楚一个 Agent 系统常见的核心组件
- 理解单 Agent 架构的基本执行闭环
- 跑通一个带状态与工具注册的迷你架构示例
- 知道真实系统为什么离不开观测和护栏

---

## 一、Agent 不是“一个模型”这么简单

### 1.1 最基础的误解：以为 Agent 只是大模型 + Prompt

真实可用的 Agent 系统，通常至少还需要：

- 工具层
- 状态管理
- 执行循环
- 错误处理
- 安全限制

模型很重要，但它更像大脑的一部分，而不是整个系统。

### 1.2 你可以把 Agent 想成一个小型操作系统

里面有：

- 决策中心
- 工具箱
- 记事本
- 执行记录
- 安全规则

这也是为什么 Agent 一旦进入工程化阶段，就不再只是“写 prompt”。

---

## 二、常见核心组件

### 2.1 Planner / 决策器

负责决定：

- 当前任务怎么拆
- 下一步做什么
- 要不要调用工具

在简单系统里，这部分可能直接由 LLM 负责。  
在更强控制场景里，也可能由规则 + LLM 混合完成。

### 2.2 Tool Layer / 工具层

这是 Agent 能行动的关键。

工具可能包括：

- 搜索
- 数据库查询
- API 调用
- 文件读写
- 计算器

如果没有工具，很多 Agent 其实只能“说”，不能“做”。

---

## 三、状态、记忆和上下文

### 3.1 State：当前任务进行到哪

状态通常记录：

- 用户目标
- 已执行步骤
- 中间结果
- 失败重试次数

这和“长期记忆”不是一回事。  
它更像当前任务的工作区。

### 3.2 Memory：跨回合保留什么

记忆更偏向：

- 用户偏好
- 历史项目
- 长期上下文

很多基础 Agent 并不一定一开始就需要复杂记忆，但几乎都会需要状态。

---

## 四、一个标准执行闭环

### 4.1 最小闭环：感知、决策、行动、观察

```mermaid
flowchart LR
    A["输入目标"] --> B["决定下一步"]
    B --> C["调用工具 / 生成动作"]
    C --> D["拿到结果并更新状态"]
    D --> B
    D --> E["满足结束条件后输出"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style B fill:#fff3e0,stroke:#e65100,color:#333
    style C fill:#f3e5f5,stroke:#6a1b9a,color:#333
    style D fill:#fffde7,stroke:#f9a825,color:#333
    style E fill:#e8f5e9,stroke:#2e7d32,color:#333
```

### 4.2 为什么这个闭环很关键？

因为 Agent 的本质不是“一次性回答”，而是：

> 根据中间结果不断调整动作。

这就是它和普通聊天机器人的根本差别之一。

---

## 五、一个迷你可运行架构示例

下面这个例子里，我们显式写出：

- 工具注册表
- 状态
- 决策逻辑
- 执行循环

```python
def tool_weather(city):
    data = {"北京": "晴 22 度", "上海": "多云 25 度"}
    return data.get(city, "暂无该城市天气")

def tool_calc(expression):
    return str(eval(expression, {"__builtins__": {}}))

TOOLS = {
    "weather": tool_weather,
    "calc": tool_calc
}

def decide_next_action(state):
    query = state["query"]
    if state["done"]:
        return None

    if "天气" in query and not state["steps"]:
        city = "北京" if "北京" in query else "上海"
        return {"tool": "weather", "args": city}

    if "计算" in query and not state["steps"]:
        expression = query.replace("计算", "").strip()
        return {"tool": "calc", "args": expression}

    return {"tool": None, "args": None}

def run_agent(query):
    state = {
        "query": query,
        "steps": [],
        "observations": [],
        "done": False
    }

    while not state["done"]:
        action = decide_next_action(state)
        if action is None or action["tool"] is None:
            state["done"] = True
            break

        tool_name = action["tool"]
        args = action["args"]
        result = TOOLS[tool_name](args)

        state["steps"].append(action)
        state["observations"].append(result)
        state["done"] = True

    if state["observations"]:
        return state, state["observations"][-1]
    return state, "没有可执行动作"

state, answer = run_agent("计算 23 * 8")
print("状态:", state)
print("最终答案:", answer)
```

这个例子很小，但已经包含了 Agent 架构的核心味道。

---

## 六、Guardrails：为什么护栏必不可少？

### 6.1 因为 Agent 会行动

行动就意味着风险。

比如：

- 调错工具
- 重复执行
- 越权访问
- 误删数据

所以真实系统里常见护栏包括：

- 工具白名单
- 参数校验
- 最大步数限制
- 人工确认节点

### 6.2 一个最简单的护栏例子

```python
def safe_eval(expression):
    allowed_chars = set("0123456789+-*/(). ")
    if not set(expression) <= allowed_chars:
        return "表达式包含不允许的字符"
    return str(eval(expression, {"__builtins__": {}}))

print(safe_eval("3 * (4 + 5)"))
print(safe_eval("__import__('os').system('rm -rf /')"))
```

护栏的核心思想不是“让系统完全不会错”，而是把错误范围收窄。

---

## 七、Observability：为什么要能看见 Agent 在做什么？

### 7.1 因为多步系统不透明就很难调试

你至少希望能看到：

- 每一步决定了什么
- 调了哪个工具
- 工具返回了什么
- 为什么结束

### 7.2 最小观测信息通常包括

- 输入
- action
- observation
- 最终输出
- 时间成本

很多 Agent 项目最后卡住，不是因为模型太弱，而是因为系统根本看不清自己哪里错了。

---

## 八、单 Agent 和多 Agent

### 8.1 单 Agent 通常先学会

大多数系统应该先把单 Agent 打稳：

- 更好调试
- 更易收敛
- 架构更清楚

### 8.2 多 Agent 不是默认升级路线

只有当任务真的适合分工时，多 Agent 才值得考虑。

例如：

- 规划 Agent
- 执行 Agent
- 评审 Agent

如果任务不复杂，多 Agent 反而会增加协调成本。

---

## 九、初学者常见误区

### 9.1 先上多 Agent，再想清楚单 Agent

这通常会让调试难度直线上升。

### 9.2 把状态和记忆混成一件事

状态更偏当前任务，记忆更偏跨任务积累。

### 9.3 没有日志和回放机制

一旦系统出错，就只能靠猜。

---

## 小结

这一节最重要的认识是：

> Agent 架构的关键，不只是“能不能调用工具”，而是能不能把决策、执行、状态、护栏和观测组织成一个稳定闭环。

这也是为什么真正的 Agent 工程，既是模型问题，也是系统设计问题。

---

## 练习

1. 给迷你 Agent 再增加一个 `docs_search` 工具。
2. 给 `run_agent()` 增加“最大步数”限制。
3. 想一想：如果工具经常超时，架构层面应该补哪些机制？
