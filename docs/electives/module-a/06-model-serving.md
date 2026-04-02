---
title: "1.6 模型服务化"
sidebar_position: 6
description: "从请求队列、批处理、版本路由、健康检查到服务指标，理解模型服务化为什么是一套完整的工程系统。"
keywords: [model serving, batching, request queue, version routing, health check, deployment]
---

# 模型服务化

:::tip 本节定位
“把模型跑起来”和“把模型服务化”是两件不同的事。

前者更像：

- 在脚本里调用一次模型

后者更像：

- 接请求
- 排队
- 调度
- 返回结果
- 监控与升级

这节课要解决的，就是从“能调用模型”走到“能提供模型服务”这一步。
:::

## 学习目标

- 理解模型服务化的核心组成部分
- 理解请求队列、批处理和版本路由在服务里的作用
- 通过可运行示例搭建一个最小 serving 流程
- 建立模型服务上线前该关注哪些指标的意识

---

## 一、为什么模型服务化不是“包个 API”就结束？

### 1.1 因为真实请求不是一条一条均匀到来的

服务会遇到：

- 突发高峰
- 大小不一的请求
- 慢请求和快请求混杂

这意味着你需要：

- 队列
- 调度
- 超时
- 指标

### 1.2 因为模型会升级

上线后你还会遇到：

- 新版本灰度
- 模型回滚
- 多版本共存

所以服务化不只是“把当前模型暴露出去”，  
还包括未来如何维护。

### 1.3 一个类比

单次推理像你自己在厨房做一顿饭。  
模型服务化更像开餐厅：

- 客人会排队
- 厨房要调度
- 菜要稳定出品

---

## 二、服务化最核心的几个组件

### 2.1 请求入口

负责：

- 接收请求
- 参数校验
- 身份认证

### 2.2 队列

负责：

- 缓冲请求
- 平滑流量

### 2.3 批处理器

负责把多个小请求合并，提升吞吐。

### 2.4 模型执行器

真正完成：

- 预处理
- 推理
- 后处理

### 2.5 版本路由与健康检查

负责：

- 哪个请求去哪个模型版本
- 某个实例是否可接流量

---

## 三、先跑一个最小 serving 流程

这个示例会模拟：

1. 请求进入队列
2. 批处理器按批次出队
3. 模型执行器统一处理
4. 按模型版本返回结果

```python
from collections import deque


request_queue = deque(
    [
        {"id": "req1", "text": "退款规则", "model_version": "v1"},
        {"id": "req2", "text": "发票规则", "model_version": "v1"},
        {"id": "req3", "text": "地址修改", "model_version": "v2"},
        {"id": "req4", "text": "证书说明", "model_version": "v2"},
    ]
)


def batch_pop(queue, batch_size):
    batch = []
    while queue and len(batch) < batch_size:
        batch.append(queue.popleft())
    return batch


def model_executor(batch):
    outputs = []
    for item in batch:
        outputs.append(
            {
                "id": item["id"],
                "model_version": item["model_version"],
                "answer": f"[{item['model_version']}] processed:{item['text']}",
            }
        )
    return outputs


all_outputs = []
while request_queue:
    batch = batch_pop(request_queue, batch_size=2)
    print("batch:", batch)
    outputs = model_executor(batch)
    all_outputs.extend(outputs)

print("\noutputs:")
for item in all_outputs:
    print(item)
```

### 3.1 这段代码最值得学什么？

它说明模型服务化最基础的运行模式：

- 请求不是直接一个个立刻跑
- 而是先进队列，再按策略调度

### 3.2 为什么批处理对服务很重要？

因为很多模型在批量处理时吞吐更高。  
如果每条请求都单独跑：

- 资源利用率可能很差

### 3.3 为什么这里显式保留 `model_version`？

因为真实服务里，多版本共存很常见。  
没有版本字段，灰度和回滚会非常麻烦。

---

## 四、服务上线后最该看哪些指标？

### 4.1 延迟

至少看：

- 平均延迟
- P95 延迟

### 4.2 吞吐

例如：

- 每秒请求数
- 每秒批次数

### 4.3 错误率

包括：

- 请求失败
- 超时
- 模型内部异常

### 4.4 批处理效率

例如：

- 平均 batch size

如果长期偏小，  
说明批处理策略可能没有真正发挥作用。

---

## 五、模型服务化最容易踩的坑

### 5.1 误区一：只看模型推理时间

真实延迟通常还包括：

- 排队
- 预处理
- 后处理
- 网络开销

### 5.2 误区二：批处理越大越好

batch 太大可能提升吞吐，  
但也可能把单请求延迟拉高。

### 5.3 误区三：没有版本路由就直接替换线上模型

这样一旦出问题，回滚会非常痛苦。

---

## 小结

这节最重要的是建立一个服务视角：

> **模型服务化不是“写一个 API 调模型”，而是围绕队列、批处理、版本、健康检查和指标，把模型做成可持续运维的服务。**

只要这层理解清楚了，  
后面你做边缘部署和综合项目时就更容易串起来。

---

## 练习

1. 把示例里的 `batch_size` 改成 `1` 和 `3`，观察输出组织方式怎么变。
2. 想一想：为什么模型服务里一定要显式带上版本信息？
3. 如果你更看重单请求延迟，而不是总吞吐，批处理策略该怎么调？
4. 你会在模型服务上线后优先看哪三个指标？为什么？
