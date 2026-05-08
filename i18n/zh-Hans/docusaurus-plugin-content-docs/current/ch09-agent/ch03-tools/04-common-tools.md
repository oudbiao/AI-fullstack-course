---
title: "9.3.5 常见工具集成"
sidebar_position: 14
description: "从搜索、计算器、数据库、文件系统到浏览器，理解 Agent 里最常见的工具类型以及怎样把它们接进系统。"
keywords: [tool integration, search, calculator, database, filesystem, browser, Agent]
---

# 9.3.5 常见工具集成

:::tip 本节定位
讲工具层时，如果只停留在抽象 schema，很容易发虚。
这一节我们把镜头拉近一点，直接看：

> **Agent 系统里最常见的工具到底有哪些，它们分别怎么接？**

你会发现，很多工具虽然名字不同，但接入方式其实很有共性。
:::

## 学习目标

- 认识 Agent 中最常见的几类工具
- 理解每类工具分别适合解决什么问题
- 看懂一个统一工具注册与调度示例
- 理解工具集成时最常见的失败点和工程注意事项

---

## 为什么要把工具分类型来看？

### 因为“工具”这个词太宽了

搜索是工具，计算器是工具，数据库查询是工具，文件读写也是工具。
如果一股脑都看成“一个函数”，你很快就会混乱。

更实用的做法是先分几类：

1. 检索类
2. 计算类
3. 数据访问类
4. 文件 / 环境操作类
5. 外部服务调用类

### 为什么分类有帮助？

因为不同类型工具的关注点不同：

- 搜索类看召回质量
- 计算类看精确性和安全
- 数据库类看权限和过滤
- 文件类看路径边界
- 外部服务类看超时和重试

也就是说：

> 不同工具虽然都叫工具，但工程风险完全不一样。

---

## 最常见的五类工具

### 搜索 / 检索类

适合：

- 查文档
- 查知识库
- 查网页

特点：

- 输入通常是 query
- 输出通常是一组候选结果

### 计算类

适合：

- 四则运算
- 统计指标
- 小型数据转换

特点：

- 输出必须稳定精确
- 安全风险要格外小心

### 数据访问类

适合：

- 查数据库
- 查订单
- 查用户状态

特点：

- 参数和权限最关键
- 很多业务逻辑在这一层决定

### 文件 / 环境操作类

适合：

- 读文件
- 写文件
- 列目录
- 执行代码

特点：

- 风险高
- 边界控制极其重要

### 外部服务调用类

适合：

- 发邮件
- 调第三方 API
- 提交工单

特点：

- 失败率、超时、重试都很常见

---

## 一个统一的工具注册表

真实系统里，常常不会把工具散落在各处，而是统一注册。

### 最小可运行示例

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


def search_docs(keyword):
    docs = {
        "退款": "课程购买后 7 天内可申请退款",
        "证书": "完成项目并通过测试后可获得证书"
    }
    return docs.get(keyword, "未找到相关文档")

def calculator(expression):
    return safe_calculate(expression)

def get_user_status(user_id):
    mock_db = {
        1: {"name": "Alice", "progress": 0.15},
        2: {"name": "Bob", "progress": 0.35}
    }
    return mock_db.get(user_id, {"error": "user_not_found"})

TOOLS = {
    "search_docs": search_docs,
    "calculator": calculator,
    "get_user_status": get_user_status
}

print(TOOLS.keys())
```

预期输出：

```text
dict_keys(['search_docs', 'calculator', 'get_user_status'])
```

### 为什么统一注册很重要？

因为后面你会需要：

- 统一描述 schema
- 统一做权限控制
- 统一打日志
- 统一调度和统计

如果工具没有注册表，系统会越来越难维护。

---

## 一个统一调度器

### 最小调度器示例

```python
def dispatch(call):
    name = call["name"]
    arguments = call["arguments"]

    if name not in TOOLS:
        return {"error": "unknown_tool"}

    try:
        result = TOOLS[name](**arguments)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

calls = [
    {"name": "search_docs", "arguments": {"keyword": "退款"}},
    {"name": "calculator", "arguments": {"expression": "12 * 7"}},
    {"name": "get_user_status", "arguments": {"user_id": 1}}
]

for call in calls:
    print(call, "->", dispatch(call))
```

预期输出：

```text
{'name': 'search_docs', 'arguments': {'keyword': '退款'}} -> {'result': '课程购买后 7 天内可申请退款'}
{'name': 'calculator', 'arguments': {'expression': '12 * 7'}} -> {'result': 84}
{'name': 'get_user_status', 'arguments': {'user_id': 1}} -> {'result': {'name': 'Alice', 'progress': 0.15}}
```

### 这段代码教会你什么？

它教会你：

- 不同工具可以共享统一调用入口
- 程序端可以统一做错误处理
- 后面要扩工具时，结构也不会乱

---

## 不同类型工具到底要注意什么？

### 搜索类工具

重点关注：

- query 是否改写
- 返回多少条结果
- 结果是否要 rerank

### 计算类工具

重点关注：

- 安全
- 精度
- 表达式是否合法

一个简单的安全计算器示例：

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


def safe_calculator(expression):
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return {"error": "invalid_expression"}
    return {"result": safe_calculate(expression)}

print(safe_calculator("3 * (4 + 5)"))
print(safe_calculator("__import__('os').system('rm -rf /')"))
```

预期输出：

```text
{'result': 27}
{'error': 'invalid_expression'}
```

### 数据库类工具

重点关注：

- 权限
- 参数完整性
- 查询边界

例如，不要让模型随意生成任意 SQL 再直接执行。

### 文件类工具

重点关注：

- 路径白名单
- 写入权限
- 是否需要人工确认

### 外部服务类工具

重点关注：

- 超时
- 重试
- 幂等性

---

## 一个更贴近 Agent 的工具组合例子

### 场景：判断用户能不能退款

这件事可能需要两个工具：

1. 查用户学习进度
2. 查退款政策

```python
def refund_eligibility_agent(user_id):
    status = get_user_status(user_id)
    if "error" in status:
        return {"error": "用户不存在"}

    policy = search_docs("退款")
    progress = status["progress"]

    can_refund = progress < 0.2
    return {
        "user": status["name"],
        "progress": progress,
        "policy": policy,
        "can_refund": can_refund
    }

print(refund_eligibility_agent(1))
print(refund_eligibility_agent(2))
```

预期输出：

```text
{'user': 'Alice', 'progress': 0.15, 'policy': '课程购买后 7 天内可申请退款', 'can_refund': True}
{'user': 'Bob', 'progress': 0.35, 'policy': '课程购买后 7 天内可申请退款', 'can_refund': False}
```

### 这段代码真正说明了什么？

它说明：

> 工具集成不是每个工具单独存在，而是经常要协同完成一个目标。

这也是为什么后面 Agent 会越来越依赖工具编排能力。

---

## 工具集成最常见的失败点

### schema 对不上

例如：

- 工具需要 `user_id`
- 模型却传了 `id`

### 返回值格式不统一

如果有的工具返回字符串，有的返回 dict，有的返回 list，系统会越来越难接。

### 没有统一错误处理

一个工具返回 `None`，另一个抛异常，第三个返回 `"failed"`，后面逻辑很容易乱。

### 没有日志和回放

线上一出错，就很难知道到底是哪类工具出了问题。

---

## 一个实用建议：统一工具返回格式

最稳妥的做法之一是统一工具输出结构，例如都返回：

```python
{
  "ok": True,
  "data": ...
}
```

或者：

```python
{
  "ok": False,
  "error": ...
}
```

一个小示例：

```python
def wrapped_search(keyword):
    try:
        result = search_docs(keyword)
        return {"ok": True, "data": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

print(wrapped_search("退款"))
```

预期输出：

```text
{'ok': True, 'data': '课程购买后 7 天内可申请退款'}
```

这会让后面 Agent 层更容易做统一判断。

---

## 初学者最常踩的坑

### 把所有工具都接进来再说

工具越多，系统越复杂。
更稳妥的做法是：

- 先接最刚需的 2~3 个

### 不区分高风险工具和低风险工具

文件删除、支付操作、数据库写入，和搜索文档不是一个风险等级。

### 工具接口没有统一约定

这是很多 Agent 系统越做越乱的直接原因。

---

## 小结

这一节最重要的不是背“有哪些工具”，而是理解：

> **常见工具集成的关键，不只是把工具接进来，而是把它们用统一接口、统一错误处理、统一边界约束组织起来。**

只有这样，工具层才会成为 Agent 的能力放大器，而不是故障制造器。

---

## 练习

1. 给本节工具注册表再加一个 `get_weather(city)` 工具。
2. 把所有工具的返回值统一成 `{"ok": ..., "data": ..., "error": ...}` 格式。
3. 想一想：为什么数据库写入工具和搜索工具不应该放在同一个权限等级？
4. 用自己的话解释：为什么说工具注册表和统一调度器是 Agent 工程里非常重要的两个结构？
