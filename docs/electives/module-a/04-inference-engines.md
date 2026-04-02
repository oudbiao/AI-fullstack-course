---
title: "1.4 推理引擎"
sidebar_position: 4
description: "从 ONNX Runtime、TensorRT、OpenVINO 这类推理引擎的角色和差异讲起，理解为什么部署不只是“导出模型”这么简单。"
keywords: [inference engine, ONNX Runtime, TensorRT, OpenVINO, execution graph, deployment]
---

# 推理引擎

:::tip 本节定位
训练好的模型并不会自动变成高性能线上服务。  
中间往往还隔着一层非常关键的系统组件：

- 推理引擎

它负责把模型图真正高效地跑在某类硬件上。

所以这节课要回答的是：

> **为什么部署里经常不是“直接加载模型权重就推理”，而是要先过一层推理引擎。**
:::

## 学习目标

- 理解推理引擎在部署链路中的角色
- 区分不同推理引擎大致适合什么硬件和场景
- 通过可运行示例理解“延迟、吞吐、适配性”这三类指标
- 建立选引擎时的第一层判断

---

## 一、推理引擎到底在做什么？

### 1.1 它不是模型本身

模型回答的是：

- 网络结构和参数是什么

推理引擎回答的是：

- 这套结构怎样在目标设备上更高效地执行

### 1.2 它通常会做哪些事？

常见包括：

- 图优化
- 算子融合
- 内存规划
- 后端 kernel 选择

### 1.3 一个类比

模型像菜谱。  
推理引擎像厨房调度系统。

同一个菜谱，  
在不同厨房用不同流程做，速度和质量都会不同。

---

## 二、为什么推理引擎会有那么多种？

### 2.1 因为硬件不一样

常见目标环境包括：

- 通用 CPU
- NVIDIA GPU
- Intel CPU / NPU
- 边缘设备

### 2.2 因为优化目标不一样

有的更看重：

- 易用性

有的更看重：

- 极致性能

### 2.3 所以没有“绝对最强引擎”

更合理的问法是：

- 这类模型、这类硬件、这类目标下，哪个引擎更合适？

---

## 三、先用一个小示例理解“引擎选择”

这个例子不会真的跑 ONNX Runtime 或 TensorRT，  
但会很直接地模拟：

- 不同引擎在不同场景下的延迟、吞吐和适配分

```python
engines = [
    {"name": "onnxruntime", "latency_ms": 32, "throughput_qps": 31, "hardware_fit": 8},
    {"name": "tensorrt", "latency_ms": 14, "throughput_qps": 70, "hardware_fit": 10},
    {"name": "openvino", "latency_ms": 26, "throughput_qps": 38, "hardware_fit": 9},
]


def score(engine, prefer_low_latency=True):
    latency_score = 100 / engine["latency_ms"]
    throughput_score = engine["throughput_qps"] / 10
    hardware_score = engine["hardware_fit"]

    if prefer_low_latency:
        return round(latency_score * 0.5 + throughput_score * 0.2 + hardware_score * 0.3, 2)
    return round(latency_score * 0.2 + throughput_score * 0.5 + hardware_score * 0.3, 2)


latency_first = sorted(
    [{**e, "score": score(e, True)} for e in engines],
    key=lambda x: x["score"],
    reverse=True,
)

throughput_first = sorted(
    [{**e, "score": score(e, False)} for e in engines],
    key=lambda x: x["score"],
    reverse=True,
)

print("latency_first:")
for item in latency_first:
    print(item)

print("\nthroughput_first:")
for item in throughput_first:
    print(item)
```

### 3.1 这段代码在教什么？

它在提醒你：

- 引擎选择不是单一指标排序

如果你更看重：

- 低延迟

和更看重：

- 高吞吐

最后排名可能不同。

### 3.2 为什么这比只记“TensorRT 更快”有用？

因为真实决策从来不只是：

- 谁理论最快

还包括：

- 能不能接进当前链路
- 是否支持目标模型
- 是否值得为了这点性能增加复杂度

---

## 四、几个常见引擎的大方向区别

### 4.1 ONNX Runtime

更像通用型选手。  
优点通常是：

- 生态广
- 兼容性强
- 上手相对平衡

### 4.2 TensorRT

更像 NVIDIA 生态下的高性能路线。  
常见特点：

- GPU 场景强
- 调优空间大
- 工程门槛相对更高

### 4.3 OpenVINO

更偏 Intel 生态和特定硬件适配。  
常见特点：

- 某些 CPU / Intel 设备上表现不错
- 适合特定部署环境

### 4.4 这三者怎么选？

不要先问“谁更火”，  
而要先问：

- 我的硬件是什么
- 我的模型格式是什么
- 我更看重延迟还是易维护

---

## 五、推理引擎会直接影响哪些部署结果？

### 5.1 延迟

用户最先感知到的就是：

- 快不快

### 5.2 吞吐

服务侧更关心：

- 同一时间能扛多少请求

### 5.3 资源利用率

例如：

- 显存是不是更省
- CPU 利用率是不是更合理

### 5.4 维护复杂度

性能更高的路线，  
有时也意味着：

- 导出更复杂
- 调试更难
- 平台绑定更重

---

## 六、最常见误区

### 6.1 误区一：推理引擎只是“换个库跑”

不是。  
它往往会改变：

- 图执行方式
- 优化策略
- 硬件利用率

### 6.2 误区二：最快的引擎就是最好的引擎

如果兼容性差、调试复杂、部署门槛太高，  
“最快”未必就是最优。

### 6.3 误区三：先选引擎，再看硬件

更合理的顺序通常是反过来：

- 先看硬件和目标约束
- 再选引擎

---

## 小结

这节最重要的，不是记下几个引擎名字，  
而是建立一个部署判断：

> **推理引擎是在模型和硬件之间做高效执行适配的系统层，它的价值不只是“跑起来”，而是“更快、更省、更适配”。**

只要这层理解清楚了，后面学服务化和边缘部署时就会更自然。

---

## 练习

1. 调整示例里的打分权重，看看“更看重硬件适配”时排序怎么变。
2. 为什么说选推理引擎本质上是“硬件 + 模型 + 目标”的联合决策？
3. 如果你部署在 NVIDIA GPU 上，为什么 TensorRT 往往更值得优先考虑？
4. 想一想：如果团队维护能力一般，但项目需要尽快上线，你会更偏向通用型还是极致优化型引擎？
