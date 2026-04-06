---
title: "4.4 日志与监控"
sidebar_position: 19
description: "从结构化日志、关键指标、trace 到告警思路，理解 LLM 服务为什么一定要可观测。"
keywords: [logging, monitoring, tracing, metrics, observability, LLM ops]
---

# 日志与监控

:::tip 本节定位
很多 LLM 应用本地 Demo 跑起来都不错，但一到线上就会暴露一个问题：

> **出了问题，你根本不知道是哪里坏了。**

日志与监控的价值，不在“多记点东西”，而在：

> 让系统出问题时，你有办法定位、解释、回放、修复。
:::

## 学习目标

- 理解日志、指标、追踪三者分别解决什么问题
- 学会设计结构化日志字段
- 理解 LLM 系统里最值得监控的指标有哪些
- 看懂一个最小日志 + 监控示例

---

## 先建立一张地图

日志与监控更适合按“发生了什么 -> 整体表现怎样 -> 单条请求经历了什么”来理解：

```mermaid
flowchart LR
    A["日志"] --> B["记录单点事件"]
    B --> C["指标"]
    C --> D["看整体趋势"]
    D --> E["Trace"]
    E --> F["还原单条请求链路"]
```

所以这节真正想解决的是：

- 出问题时你到底先看哪一层
- 为什么日志、指标、trace 缺一块都会很难排障

---

## 一、为什么这件事特别重要？

### 1.1 LLM 系统的故障比普通接口更隐蔽

普通接口错误通常比较直接：

- 500
- 超时
- 参数错

而 LLM 系统还会有这些“软故障”：

- 回答变差
- 检索飘了
- token 成本暴涨
- 只在某些场景失误

所以如果你没有观测能力，系统经常会变成：

> 看起来还活着，但其实已经坏了一半。 

### 1.2 日志与监控到底在解决什么？

可以先粗略分三层：

- 日志：发生了什么
- 指标：发生得多不多、快不快、贵不贵
- 追踪：一条请求完整走了哪些步骤

### 1.3 一个更适合新人的总类比

你可以把可观测性理解成：

- 给系统装仪表盘、行车记录仪和维修日志

没有这些东西时，系统坏了你只能说：

- 好像有点不对

有了之后你才可能知道：

- 哪里开始异常
- 是偶发还是持续
- 是单条请求的问题，还是整个系统的问题

---

## 二、先看日志：最基础也最常被写坏

### 2.1 什么叫“结构化日志”？

相比只打一行字符串：

```python
print("request received")
```

更有价值的是记录结构化字段：

- request_id
- user_id
- stage
- latency_ms
- model_name

### 2.2 一个最小结构化日志示例

```python
log = {
    "trace_id": "trace_001",
    "stage": "retrieval",
    "query": "退款政策是什么",
    "latency_ms": 120,
    "top_k": 3
}

print(log)
```

这种日志最大的好处是：

> 后面你可以按字段查、按字段聚合，而不只是人工看文本。 

---

## 三、指标：系统整体表现的体温计

### 3.1 最值得监控的几类指标

对 LLM 系统来说，最常见的指标包括：

- 请求量
- 错误率
- 平均延迟
- P95 / P99 延迟
- token 使用量
- 工具调用次数
- 检索命中率

### 3.2 一个最小指标汇总示例

```python
requests = [
    {"latency_ms": 800, "tokens": 600, "ok": True},
    {"latency_ms": 1200, "tokens": 750, "ok": True},
    {"latency_ms": 3000, "tokens": 900, "ok": False}
]

avg_latency = sum(r["latency_ms"] for r in requests) / len(requests)
error_rate = sum(not r["ok"] for r in requests) / len(requests)
avg_tokens = sum(r["tokens"] for r in requests) / len(requests)

print("avg_latency_ms =", avg_latency)
print("error_rate     =", error_rate)
print("avg_tokens     =", avg_tokens)
```

这就是监控面板最小的雏形。

### 3.3 一个很适合初学者先记的指标表

| 指标 | 更像在回答什么问题 |
|---|---|
| 请求量 | 系统忙不忙 |
| 错误率 | 系统是否经常失败 |
| 平均 / P95 延迟 | 用户是否在等太久 |
| token 使用量 | 成本是否异常 |
| 检索命中率 | RAG 链路是否在变差 |
| 工具调用成功率 | Agent 行动层是否稳定 |

这个表很适合新人，因为它会把“指标很多”重新压回到几个能理解的问题上。

---

## 四、追踪（trace）：一条请求到底经历了什么？

### 4.1 为什么 LLM 系统特别需要 trace？

因为一条请求往往不只过一个模块，而可能经过：

1. API 接入
2. 检索
3. 工具调用
4. 模型生成
5. 后处理

如果最后回答错了，你需要知道：

- 是检索错了
- 还是模型生成错了
- 还是工具层挂了

### 4.2 一个最小 trace 示例

```python
trace = [
    {"trace_id": "trace_001", "stage": "api_in", "latency_ms": 20},
    {"trace_id": "trace_001", "stage": "retrieval", "latency_ms": 120},
    {"trace_id": "trace_001", "stage": "llm_generate", "latency_ms": 850},
    {"trace_id": "trace_001", "stage": "response_out", "latency_ms": 15}
]

for item in trace:
    print(item)
```

trace 的核心价值是：

> 让你看到“同一个请求的完整旅程”。 

### 4.3 第一次做线上排障时，最稳的默认顺序

更稳的顺序通常是：

1. 先看指标有没有整体异常
2. 再看日志里具体哪一类请求在出错
3. 最后顺着 trace 看完整链路

这样会比一开始就翻满屏日志更容易定位问题。

---

## 五、一个更贴近真实的最小观测闭环

```python
import time

def timed_stage(name, fn, *args, **kwargs):
    start = time.time()
    result = fn(*args, **kwargs)
    latency_ms = int((time.time() - start) * 1000)
    log = {
        "trace_id": "trace_demo_001",
        "stage": name,
        "latency_ms": latency_ms
    }
    print(log)
    return result

def fake_retrieve(query):
    time.sleep(0.1)
    return ["退款政策"]

def fake_llm(docs):
    time.sleep(0.2)
    return f"根据 {docs} 生成回答"

docs = timed_stage("retrieval", fake_retrieve, "退款政策是什么")
answer = timed_stage("llm_generate", fake_llm, docs)
print(answer)
```

这个例子虽然小，但已经把：

- trace_id
- stage
- latency

这几个核心字段带起来了。

---

## 六、LLM 系统最值得额外监控的东西

相比传统 API，LLM 系统通常还值得多监控这些：

### 6.1 token 成本

因为它直接决定：

- 钱花了多少
- prompt 是否越来越长

### 6.2 检索质量

例如：

- top-1 是否命中
- 检索为空比例

### 6.3 工具调用质量

例如：

- 工具调用成功率
- 参数校验失败率
- 重试率

### 6.4 回答质量信号

例如：

- 用户追问率
- 用户纠错率
- 点踩率

这类指标不能替代离线评估，但非常重要。

---

## 七、告警为什么不能只看“服务挂没挂”？

### 7.1 LLM 系统很多问题不会直接 500

例如：

- 回答质量持续下降
- token 用量突然翻倍
- 检索命中率掉得很厉害

这些问题系统可能仍然“活着”，但业务已经明显坏了。

### 7.2 所以告警最好分两层

- 基础可用性告警
  - 错误率
  - 超时率

- 业务质量告警
- 检索命中率下降
- 平均 token 数异常上升
- 用户负反馈异常增加

### 7.3 一个很适合初学者先记的告警分层表

| 告警类型 | 典型例子 |
|---|---|
| 可用性告警 | 错误率高、超时率高 |
| 成本告警 | token 暴涨、调用次数异常 |
| 质量告警 | 检索命中率下降、用户追问率上升 |

这个表很适合新人，因为它会提醒你：

- LLM 系统的“坏掉”不只是一种坏法

---

## 八、一个很实用的日志字段清单

如果你在做 LLM 服务，最实用的一组字段通常包括：

| 字段 | 作用 |
|---|---|
| trace_id | 串起整条链路 |
| user_id / session_id | 定位用户或会话 |
| stage | 当前在哪个环节 |
| latency_ms | 这一步耗时多少 |
| model_name | 用了哪个模型 |
| prompt_tokens / completion_tokens | 成本分析 |
| tool_name | 调了什么工具 |
| retrieval_topk | 检索设置 |
| error_code | 失败类型 |

不是每一条日志都要全打，但这份清单很适合作为设计起点。

---

## 九、初学者最常踩的坑

### 9.1 只打字符串，不打字段

后面很难聚合分析。

### 9.2 只记成功，不记失败

这会让错误定位非常痛苦。

### 9.3 没有 trace_id

出了问题无法追完整链路。

### 9.4 只监控系统可用性，不监控业务质量

这是 LLM 项目里特别常见的问题。

---

## 小结

这一节最重要的不是“学会打日志”，而是理解：

> **日志、指标、trace 共同组成了系统的可观测性，它们决定你能不能真正维护一个上线后的 LLM 服务。**

没有观测，很多故障只能靠猜；  
有了观测，系统才真正可维护。

## 如果把它做成项目或系统设计，最值得展示什么

最值得展示的通常不是：

- “我接了日志系统”

而是：

1. 一条请求的 trace
2. 一组关键指标
3. 一个典型错误案例怎么被定位出来
4. 质量告警和可用性告警是怎么分层的

这样别人会更容易看出：

- 你理解的是可观测性闭环
- 不只是会打印日志

---

## 练习

1. 给本节的 `timed_stage()` 再加一个 `error_code` 字段。
2. 设计一个你自己的日志结构，专门记录检索阶段。
3. 想一想：如果服务错误率没变，但用户追问率突然上升，这通常意味着什么？
4. 用自己的话解释：为什么 LLM 系统的告警不能只看 500 和超时？
