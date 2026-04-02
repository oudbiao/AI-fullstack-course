---
title: "5.3 结构化输出"
sidebar_position: 17
description: "从为什么自然语言输出不够稳，到 JSON 约束、字段设计和校验，真正理解结构化输出在 LLM 工程里的价值。"
keywords: [structured output, JSON, schema, validation, prompt engineering, LLM]
---

# 结构化输出

:::tip 本节定位
很多人第一次用大模型时，默认让它输出一段自然语言。  
但一旦你要把模型接进程序系统，很快就会碰到一个现实问题：

> **自然语言虽然灵活，但不稳定。**

结构化输出就是在解决“让模型的回答更像程序接口”这件事。
:::

## 学习目标

- 理解为什么结构化输出对 LLM 应用非常重要
- 学会设计一个简单但清晰的 JSON 输出格式
- 理解字段设计、约束说明和校验逻辑
- 看懂一个从 Prompt 到 JSON 解析的最小闭环
- 分清“结构化输出”和“Function Calling”的区别与联系

---

## 一、为什么光有自然语言不够？

### 1.1 一个很常见的脆弱场景

假设你想让模型识别用户意图：

用户输入：

> “我想了解退款政策”

如果模型返回：

> “这个用户大概率是想问退款相关内容，建议转到退款模块。”

人当然能看懂。  
但程序会很难稳定使用这段话。

因为程序更希望拿到的是：

```json
{
  "intent": "refund_policy",
  "confidence": 0.92
}
```

### 1.2 真正的问题是什么？

问题不在于模型不会回答，而在于：

> **自然语言输出太自由，程序很难稳定消费。**

所以当模型输出要继续传给：

- 前端
- 后端
- 工作流
- 数据库

时，结构化输出几乎就变成刚需。

---

## 二、结构化输出到底是什么？

### 2.1 一句话理解

> **结构化输出 = 让模型按预先约定的字段和格式输出结果。**

最常见格式包括：

- JSON
- 列表
- 表格
- 固定字段对象

### 2.2 为什么 JSON 最常见？

因为它同时满足：

- 人能读
- 程序能解析
- 结构清楚

所以在 LLM 应用里，JSON 通常是结构化输出的第一选择。

---

## 三、结构化输出最核心的设计点是什么？

### 3.1 字段要少而清楚

初学者很容易犯的错是：

- 一上来设计 20 个字段
- 但每个字段含义都不稳定

更好的原则是：

> **先用最少字段表达最关键结果。**

例如意图识别：

```json
{
  "intent": "refund_policy",
  "confidence": 0.92
}
```

就已经够用了。

### 3.2 字段命名要稳定

如果今天叫：

- `intent`

明天又叫：

- `user_intent`

后天又叫：

- `task_type`

那程序端会越来越混乱。

所以结构化输出的第一原则之一是：

> 字段名要稳定。 

---

## 四、一个最小可运行示例：从字符串 JSON 到程序解析

### 4.1 先看最小解析

```python
import json

text = '{"intent": "refund_policy", "confidence": 0.92}'
data = json.loads(text)

print(data)
print("intent =", data["intent"])
print("confidence =", data["confidence"])
```

### 4.2 这段代码虽然简单，但意义很大

它在教你：

1. 结构化输出不是“看起来像 JSON”，而是要能被真正解析
2. 解析后，程序就可以稳定取字段

也就是说，结构化输出的价值不在“更好看”，而在：

> **后续程序真的能用。**

---

## 五、一个更贴近真实任务的小例子：用户意图识别

### 5.1 假设你要求模型输出这个结构

```json
{
  "intent": "refund_policy",
  "needs_human": false,
  "confidence": 0.92
}
```

### 5.2 模拟模型输出 + 程序解析

```python
import json

mock_model_output = """
{
  "intent": "refund_policy",
  "needs_human": false,
  "confidence": 0.92
}
"""

data = json.loads(mock_model_output)

if data["intent"] == "refund_policy" and not data["needs_human"]:
    print("进入退款政策自动处理流程")
else:
    print("转人工或进入其他流程")

print(data)
```

这就已经是结构化输出在真实工作流里的典型使用方式了。

---

## 六、Prompt 要怎么写，结构化输出才更稳？

### 6.1 不要只说“请输出 JSON”

更稳妥的写法通常包括：

- 明确字段名
- 明确字段类型
- 明确只能输出 JSON
- 明确不要附加解释

例如：

```text
请根据用户输入进行意图识别，并严格输出 JSON。

字段要求：
- intent: string，可选值为 refund_policy / certificate / other
- needs_human: boolean
- confidence: float，范围 0 到 1

不要输出任何额外解释，只输出 JSON。
```

### 6.2 为什么这会更稳？

因为你不是只在“提需求”，而是在：

> **给模型定义输出合同。**

合同越清楚，结果越稳。

---

## 七、为什么结构化输出仍然需要校验？

### 7.1 因为模型不是编译器

即使你 prompt 写得很好，模型也可能：

- 漏字段
- 写错类型
- 多输出解释文字
- JSON 格式不闭合

### 7.2 一个最小校验示例

```python
import json

def validate_output(text):
    try:
        data = json.loads(text)
    except Exception:
        return False, "invalid_json"

    required = ["intent", "needs_human", "confidence"]
    for field in required:
        if field not in data:
            return False, f"missing_{field}"

    if not isinstance(data["intent"], str):
        return False, "intent_type_error"
    if not isinstance(data["needs_human"], bool):
        return False, "needs_human_type_error"
    if not isinstance(data["confidence"], (int, float)):
        return False, "confidence_type_error"

    return True, data

good = '{"intent":"refund_policy","needs_human":false,"confidence":0.92}'
bad = '{"intent":"refund_policy","confidence":"high"}'

print(validate_output(good))
print(validate_output(bad))
```

这一步特别重要，因为它让你的系统从：

- “模型大概会这么输出”

变成：

- “程序明确知道输出是否合格”

---

## 八、结构化输出和 Function Calling 有什么关系？

### 8.1 相同点

它们都在做一件事：

> 把模型输出从自由文本，变成程序更容易接住的格式。 

### 8.2 不同点

粗略地说：

- **结构化输出**：更广泛，重点是“结果格式稳定”
- **Function Calling**：更进一步，重点是“输出的是工具调用意图”

例如：

- 结构化输出：输出分类结果 JSON
- Function Calling：输出 `{name, arguments}` 去调工具

所以可以理解成：

> Function Calling 是结构化输出的一种更偏执行型形态。 

---

## 九、真实项目里最常见的坑

### 9.1 字段设计过多

字段越多，模型越容易错，后处理也越复杂。

### 9.2 字段含义不稳定

比如 `confidence` 有时写 0~1，有时写百分比，这种设计很危险。

### 9.3 不做解析与校验

很多 demo 看起来能跑，但一接程序就崩，问题通常出在这里。

### 9.4 输出结构和业务流程脱节

如果 JSON 虽然完整，但不能直接驱动后续流程，那结构化输出就没有真正服务业务。

---

## 十、小结

这一节最重要的不是记住 JSON 语法，而是理解：

> **结构化输出的本质，是把模型回答变成程序可以稳定消费的中间结果。**

当你开始把模型接进真实系统时，这往往比“回答写得更漂亮”更重要。

---

## 练习

1. 设计一个“课程问答路由”的 JSON 输出格式，至少包含 `intent`、`confidence`、`needs_human` 三个字段。
2. 故意构造一个缺字段的 JSON，看看校验器是否能拦住。
3. 想一想：什么时候应该用结构化输出，什么时候直接自然语言就够了？
4. 用自己的话解释：为什么说结构化输出是 Prompt 工程走向工程化的关键一步？
