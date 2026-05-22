---
title: "E.A.3 模型优化技术"
description: "把模型优化当成延迟、内存、准确率和运维风险之间的可度量取舍。"
sidebar:
  order: 3
head:
  - tag: meta
    attrs:
      name: keywords
      content: "模型优化, 量化, 剪枝, 蒸馏, 融合, batching, 部署"
---
![模型优化路线图](/img/course/elective-model-optimization-map.webp)

![模型优化取舍仪表盘](/img/course/elective-optimization-tradeoff-dashboard.webp)

优化不是“把模型压到越小越好”，而是在改善一个约束时，同时检查你失去了什么。

## 运行一个很小的量化误差检查

```python
values = [0.1234, 0.5678, 0.9012]
quantized = [round(value * 255) / 255 for value in values]
errors = [abs(original - compressed) for original, compressed in zip(values, quantized)]

print([round(value, 4) for value in quantized])
print(f"max_error={max(errors):.4f}")
```

预期输出：

```text
[0.1216, 0.5686, 0.902]
max_error=0.0018
```

这是最小优化习惯：压缩，测误差，再判断误差是否可接受。

## 选择合适的优化路径

| 技术 | 什么时候适合 | 上线前检查 |
|---|---|---|
| 量化 | 延迟和内存太高 | 真实验证样本上的准确率下降 |
| 剪枝 | 很多权重或通道没用 | runtime 是否真的变快 |
| 蒸馏 | 小模型可以模仿大模型 | student 是否在边界样本失败 |
| 算子融合 | runtime overhead 高 | 引擎是否支持融合后的图 |
| Batching / scheduling | 多请求一起到达 | 长尾延迟和队列等待 |

## 实用顺序

1. 先测 baseline 延迟、内存、准确率。
2. 一次只尝试一种优化。
3. 记录前后指标。
4. 保留失败样本。
5. 只有取舍清楚时才上线。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
部署目标：本地推理、边缘设备、模型服务器或优化实验
工件：C++ 代码片段、基准测试、模型工件、服务配置或部署说明
指标：延迟、内存、吞吐量、模型大小、准确率下降或可靠性
失败检查：ABI/构建问题、硬件不匹配、量化损失或服务瓶颈
期望产出：可复现的部署或优化证据，而不只是理论笔记
```

## 通过检查

你能说明一种优化的收益、可能代价，以及真实部署前要看哪个指标，就算通过。

<details>
<summary>检查思路与讲解</summary>

一个合格答案会点名一种优化方法、它带来的收益、潜在代价，以及上线前要看的指标。例如量化可以降内存，但要检查验证集精度和失败样本；剪枝可能让模型更小，但还要确认 runtime 真的变快。

不要只说“越小越好”。要说明压缩后保留了什么、丢掉了什么，以及为什么这个取舍在真实部署里可接受。

</details>
