---
title: "3.1 Function Calling 详解"
sidebar_position: 11
description: "从 schema 设计、参数校验、错误恢复到多步调用，深入理解 Function Calling 在 Agent 工具层里的真实工程价值。"
keywords: [Function Calling, Tool Calling, 参数校验, Agent, schema, 工具调度]
---

# Function Calling 详解

:::tip 本节定位
上一节你已经知道 Function Calling 是“模型产出结构化工具调用”。  
这一节我们不再停留在“会调用”，而要进入真正重要的问题：

> **怎样把函数调用做得稳定、可控、可扩展？**

这才是它在 Agent 系统里真正的工程价值。
:::

## 学习目标

- 理解 Function Calling 的完整工程链路
- 学会设计更稳的工具 schema
- 理解参数校验、失败处理和错误恢复
- 看懂一个多步工具调用的小型闭环
- 分清“模型做决定”和“程序做执行”各自负责什么

---

## 一、为什么要把 Function Calling 单独深挖？

### 1.1 初级版本只解决“能不能调”

最简单的函数调用系统只要求：

- 模型选对工具
- 参数大致对

这在 demo 阶段通常够了。

### 1.2 真正上线后会立刻遇到更难的问题

比如：

- 工具很多，模型经常选错
- 参数经常缺字段
- 某些调用必须做权限控制
- 工具失败后怎么恢复？
- 多步调用时怎样避免无限循环？

所以真正的 Function Calling，不只是一个 JSON 结构，而是一套工程机制。

---

## 二、先把完整链路看清楚

### 2.1 Function Calling 的标准闭环

```mermaid
flowchart LR
    A["用户输入"] --> B["模型选择工具并产出参数"]
    B --> C["程序校验调用是否合法"]
    C --> D["执行工具"]
    D --> E["拿到结果"]
    E --> F["继续回复 / 继续下一步调用"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style B fill:#fff3e0,stroke:#e65100,color:#333
    style C fill:#f3e5f5,stroke:#6a1b9a,color:#333
    style D fill:#fffde7,stroke:#f9a825,color:#333
    style E fill:#e8f5e9,stroke:#2e7d32,color:#333
    style F fill:#ffebee,stroke:#c62828,color:#333
```

### 2.2 哪些环节分别归谁负责？

| 环节 | 谁负责 |
|---|---|
| 识别要不要调工具 | 模型 |
| 产出调用结构 | 模型 |
| 校验参数是否合法 | 程序 |
| 执行工具 | 程序 |
| 根据结果继续下一步 | 模型 / 工作流 / Agent 调度器 |

这是一个特别关键的边界：

> **模型负责“决定”，程序负责“保证执行安全和稳定”。**

---

## 三、Schema 设计为什么会直接影响效果？

### 3.1 一个坏 schema 长什么样？

```python
bad_schema = {
    "name": "search",
    "description": "做一些查询",
    "parameters": {
        "q": {"type": "string"}
    }
}

print(bad_schema)
```

这个 schema 的问题在于：

- 工具名太模糊
- 描述太空
- 参数语义不清

模型拿到这种 schema，很容易糊涂。

### 3.2 一个更好的 schema

```python
good_schema = {
    "name": "search_course_policy",
    "description": "查询课程政策类文档，例如退款、证书、学习顺序",
    "parameters": {
        "keyword": {
            "type": "string",
            "description": "需要检索的主题关键词，例如 退款 或 证书"
        }
    },
    "required": ["keyword"]
}

print(good_schema)
```

更好的 schema 往往具备：

- 工具名明确
- 描述具体
- 参数命名有语义
- 必填项清楚

---

## 四、参数校验：你不能把模型当作永远可靠的调用器

### 4.1 一个典型错误

```python
tool_call = {
    "name": "search_course_policy",
    "arguments": {}
}
```

如果你直接执行：

```python
search_course_policy(**tool_call["arguments"])
```

那程序大概率会出错。

### 4.2 一个最小校验器

```python
def validate_tool_call(call):
    if "name" not in call:
        return False, "missing_name"
    if "arguments" not in call:
        return False, "missing_arguments"

    if call["name"] == "search_course_policy":
        args = call["arguments"]
        if "keyword" not in args:
            return False, "missing_keyword"
        if not isinstance(args["keyword"], str):
            return False, "keyword_must_be_string"

    return True, "ok"

print(validate_tool_call({"name": "search_course_policy", "arguments": {"keyword": "退款"}}))
print(validate_tool_call({"name": "search_course_policy", "arguments": {}}))
```

这一步不是“锦上添花”，而是上线系统的基础防线。

---

## 五、一个更完整的可运行版本

### 5.1 先定义工具

```python
def search_course_policy(keyword):
    docs = {
        "退款": "课程购买后 7 天内且学习进度低于 20% 可申请退款。",
        "证书": "完成所有必修项目并通过结课测试后可获得证书。"
    }
    return docs.get(keyword, "未找到相关政策")

def calculate(expression):
    return str(eval(expression, {"__builtins__": {}}))
```

### 5.2 定义调度器和校验

```python
def validate_tool_call(call):
    if "name" not in call or "arguments" not in call:
        return False, "invalid_call_structure"

    if call["name"] == "search_course_policy":
        args = call["arguments"]
        if "keyword" not in args or not isinstance(args["keyword"], str):
            return False, "invalid_policy_arguments"

    if call["name"] == "calculate":
        args = call["arguments"]
        if "expression" not in args or not isinstance(args["expression"], str):
            return False, "invalid_calculate_arguments"

    return True, "ok"

def dispatch(call):
    if call["name"] == "search_course_policy":
        return search_course_policy(**call["arguments"])
    if call["name"] == "calculate":
        return calculate(**call["arguments"])
    return "unknown_tool"
```

### 5.3 模拟“模型决定工具调用”

```python
def mock_model(user_query):
    if "退款" in user_query:
        return {
            "name": "search_course_policy",
            "arguments": {"keyword": "退款"}
        }
    if "证书" in user_query:
        return {
            "name": "search_course_policy",
            "arguments": {"keyword": "证书"}
        }
    if "计算" in user_query:
        return {
            "name": "calculate",
            "arguments": {"expression": user_query.replace("计算", "").strip()}
        }
    return None
```

### 5.4 串成完整闭环

```python
queries = [
    "退款政策是什么？",
    "证书怎么拿？",
    "计算 12 * (3 + 2)"
]

for q in queries:
    print("用户问题:", q)
    call = mock_model(q)
    print("模型产出:", call)

    valid, msg = validate_tool_call(call)
    print("校验结果:", valid, msg)

    if valid:
        result = dispatch(call)
        print("工具执行结果:", result)
    else:
        print("调用被拒绝")

    print("-" * 50)
```

这个示例已经比“单纯打印 tool_call”更接近真实系统。

---

## 六、多步调用时，真正难点在哪里？

### 6.1 难点不是再多调用一次，而是状态管理

比如用户问：

> “先查退款政策，再帮我算一下 3000 元打七折是多少钱。”

这时系统可能需要：

1. 调 `search_course_policy`
2. 再调 `calculate`
3. 最后合并回答

问题在于：

- 中间结果怎么保存
- 下一步什么时候停
- 出错时怎么处理

### 6.2 一个最小多步例子

```python
def multi_step_agent(query):
    steps = []

    if "退款" in query:
        call_1 = {"name": "search_course_policy", "arguments": {"keyword": "退款"}}
        steps.append(("tool_call", call_1))
        result_1 = dispatch(call_1)
        steps.append(("tool_result", result_1))

    if "打七折" in query:
        call_2 = {"name": "calculate", "arguments": {"expression": "3000 * 0.7"}}
        steps.append(("tool_call", call_2))
        result_2 = dispatch(call_2)
        steps.append(("tool_result", result_2))

    return steps

for step in multi_step_agent("先查退款，再算 3000 元打七折"):
    print(step)
```

这就是为什么 Function Calling 讲到后面，迟早会和 Agent 结合起来。

---

## 七、失败处理和恢复为什么重要？

### 7.1 工具失败是常态，不是例外

真实系统里，工具失败非常常见：

- 参数错
- 接口超时
- 网络异常
- 数据为空

### 7.2 一个简单失败兜底

```python
def safe_dispatch(call):
    try:
        valid, msg = validate_tool_call(call)
        if not valid:
            return {"error": msg}
        return {"result": dispatch(call)}
    except Exception as e:
        return {"error": str(e)}

print(safe_dispatch({"name": "calculate", "arguments": {"expression": "2 + 3"}}))
print(safe_dispatch({"name": "calculate", "arguments": {"wrong": "2 + 3"}}))
```

一个成熟系统通常不会因为一次工具失败就直接崩掉。

---

## 八、Function Calling 深水区真正要关注什么？

### 8.1 不是“能不能调”，而是“能不能稳稳调”

真正重要的问题包括：

- schema 够不够清晰
- 参数校验够不够严格
- 工具权限有没有分层
- 多步调用如何收敛
- 错误能不能回放和定位

### 8.2 工具层是 Agent 工程的可靠性底盘

如果工具层不稳，后面这些都会跟着摇：

- 推理链
- 多步执行
- 记忆系统
- 多 Agent 协同

所以 Function Calling 虽然看起来像“结构化输出”，本质上却是 Agent 工程里的关键基础设施。

---

## 九、初学者最常踩的坑

### 9.1 把 schema 当文案写

schema 不是说明书摆设，而是调用边界本身。

### 9.2 没有校验就直接执行

这是非常危险的。

### 9.3 工具层没有日志

一旦调用错了、参数错了、执行炸了，根本不知道哪里出问题。

---

## 小结

这一节最重要的不是会写一个 `{"name": ..., "arguments": ...}`，而是理解：

> **Function Calling 的真正价值，在于把模型的决策能力，安全地接到工程系统的执行能力上。**

当你开始关注 schema 设计、参数校验、失败恢复和多步状态时，才算真正进入了工具层工程。

---

## 练习

1. 把本节工具系统再加一个 `get_weather(city)`，并补上对应 schema 与校验。
2. 故意构造一个参数错误的 tool call，看看校验器是否能拦住。
3. 把 `multi_step_agent()` 扩展成“最多执行 3 步”，避免无限循环。
4. 想一想：为什么 Function Calling 在 Agent 系统里比在普通聊天机器人里更关键？
