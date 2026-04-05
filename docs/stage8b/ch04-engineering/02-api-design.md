---
title: "4.3 API 设计与服务化"
sidebar_position: 18
description: "从请求结构、响应结构、幂等性、错误处理到版本管理，理解 LLM 服务 API 怎样设计得更稳。"
keywords: [API design, service design, idempotency, request schema, response schema, versioning]
---

# API 设计与服务化

:::tip 本节定位
做 LLM 应用时，很多人能写出一个本地脚本，但一到服务化就开始混乱。  
真正的问题不是“会不会写接口”，而是：

> **这个接口能不能长期稳定地被别人调用。**

这一节就是在回答这个问题。
:::

## 学习目标

- 理解一个 LLM 服务 API 最基本应该定义哪些内容
- 学会设计清晰的请求和响应结构
- 理解幂等性、错误返回、trace_id、版本管理这些服务化关键概念
- 看懂一个最小 API 处理闭环

---

## 一、为什么 API 设计不是“随便包个 JSON”？

### 1.1 一个差的接口长什么样？

```python
bad_request = {
    "msg": "退款政策是什么"
}

bad_response = {
    "text": "7 天内可退款"
}
```

问题在哪？

- `msg` 是什么？用户消息？系统消息？
- 没有 trace_id
- 没有错误结构
- 没有版本信息
- 没有上下文字段

### 1.2 一个好的 API 设计在做什么？

本质上它在回答：

- 输入长什么样
- 输出长什么样
- 错了时怎么表示
- 调一次和调十万次还能不能稳定

也就是说，API 设计不是“写个入口”，而是在定义：

> **系统和外部世界的契约。**

---

## 二、先设计请求结构

### 2.1 最小请求结构通常至少要有这些

- `query`
- `user_id`（可选）
- `session_id`（多轮时）
- `metadata`（可选）

### 2.2 一个更清楚的请求对象

```python
request = {
    "query": "退款政策是什么？",
    "user_id": 1,
    "session_id": "sess_001",
    "metadata": {
        "channel": "web"
    }
}

print(request)
```

这里你已经能感受到：

- 查询内容是什么
- 谁发来的
- 属于哪个会话
- 额外上下文是什么

这就比“只传一段字符串”强很多。

---

## 三、再设计响应结构

### 3.1 为什么响应也必须规范？

因为真实调用方往往不只是人，还有：

- 前端
- 其他服务
- 日志系统
- 评估系统

它们都需要稳定消费返回结果。

### 3.2 一个更稳的响应结构

```python
response = {
    "trace_id": "trace_001",
    "answer": "课程购买后 7 天内且学习进度低于 20% 可申请退款。",
    "sources": [
        {"id": "doc_001", "section": "退款政策"}
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 35
    }
}

print(response)
```

### 3.3 为什么这几个字段有价值？

- `trace_id`：方便追踪链路
- `answer`：真正的业务输出
- `sources`：便于引用和校验
- `usage`：便于成本分析

---

## 四、错误响应也必须设计

### 4.1 很多系统只设计成功返回

但工程里更常见的问题其实是：

- 参数不合法
- 上游超时
- 权限不足
- 知识库为空

### 4.2 一个统一错误结构

```python
error_response = {
    "trace_id": "trace_002",
    "error": {
        "code": "INVALID_ARGUMENT",
        "message": "query 不能为空"
    }
}

print(error_response)
```

这一步非常重要，因为它让调用方明确知道：

- 发生了什么错
- 错误类别是什么
- 是否值得重试

---

## 五、一个最小可运行的服务化处理函数

### 5.1 用纯 Python 先模拟 API handler

```python
def handle_chat(request):
    trace_id = "trace_demo_001"

    if "query" not in request or not request["query"].strip():
        return {
            "trace_id": trace_id,
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "query 不能为空"
            }
        }

    answer = f"系统回复：{request['query']}"
    return {
        "trace_id": trace_id,
        "answer": answer,
        "sources": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8}
    }

print(handle_chat({"query": "退款政策是什么？"}))
print(handle_chat({"query": ""}))
```

### 5.2 这段代码其实在教什么？

它在教你：

1. 请求先校验
2. 每次请求都有 trace_id
3. 成功和失败都要有统一结构

这已经是服务化设计最核心的一层了。

---

## 六、幂等性为什么重要？

### 6.1 什么是幂等性？

简单理解：

> 同一个请求重复调用多次，结果应该保持一致或可控。 

这在这些场景特别重要：

- 重试
- 超时后重新发起
- 网络抖动

### 6.2 哪些接口更要考虑幂等性？

尤其是：

- 工单创建
- 支付发起
- 订单变更

而纯问答接口通常天然更像“只读操作”，幂等性更容易做。

---

## 七、版本管理为什么不能后补？

### 7.1 API 一旦被别人接入，就很难随便改字段

如果今天返回：

- `answer`

明天改成：

- `response_text`

调用方就会直接挂。

### 7.2 一个简单版本策略

```python
api_info = {
    "version": "v1",
    "endpoint": "/api/v1/chat"
}

print(api_info)
```

哪怕是最小项目，也建议早点有版本意识。

---

## 八、一个更接近真实服务的 FastAPI 例子

如果你想看更贴近真实后端的写法，可以看下面这个最小版本。

:::info 运行环境
```bash
pip install fastapi uvicorn
```
:::

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/v1/chat")
def chat(payload: dict):
    if "query" not in payload or not payload["query"].strip():
        return {
            "trace_id": "trace_demo_002",
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "query 不能为空"
            }
        }

    return {
        "trace_id": "trace_demo_002",
        "answer": f"系统回复：{payload['query']}"
    }
```

这段代码虽然简单，但已经非常接近真实服务的最小雏形。

---

## 九、初学者最常踩的坑

### 9.1 请求结构太随意

一开始省事，后面会非常痛苦。

### 9.2 错误结构不统一

这会让前端和其他服务越来越难接。

### 9.3 没有 trace_id

出了问题很难追链路。

### 9.4 一开始就把接口绑死在单一业务逻辑上

这样后面扩展非常困难。

---

## 十、小结

这一节最重要的不是把接口跑起来，而是理解：

> **API 设计的核心，是让输入、输出、错误和链路追踪都变成稳定的系统契约。**

一旦契约清楚，服务才能真正长期稳定地被别人依赖。

---

## 练习

1. 给 `handle_chat()` 增加一个 `session_id` 字段支持。
2. 设计一个统一的错误码枚举，比如 `INVALID_ARGUMENT`、`TIMEOUT`、`NOT_FOUND`。
3. 想一想：如果这是一个“创建工单”接口，你会怎样考虑幂等性？
4. 用自己的话解释：为什么说 API 设计本质上是在定义系统契约？
