---
title: "E.A.5 边缘设备部署"
sidebar_position: 5
description: "通过内存、功耗、延迟和离线要求，判断模型能不能可靠地跑在边缘设备上。"
keywords: [edge deployment, Jetson, Raspberry Pi, memory budget, latency, offline inference]
---

# E.A.5 边缘设备部署

![边缘部署约束决策图](/img/course/elective-edge-deployment-constraint-map.webp)

边缘部署指模型运行在用户、摄像头、机器或传感器附近。第一问题通常不是模型准不准，而是设备能不能长期稳定运行这套系统。

## 准备内容

- Python 3.10+
- 不需要第三方包
- 一个目标场景，例如摄像头分类、工厂检测、离线表单识别

## 四个检查点

- **内存**：模型、运行时、输入缓存和服务本身都要占 RAM。
- **功耗**：能跑一次不代表能长时间不降频、不过热。
- **延迟**：有些任务要即时响应，有些可以等待。
- **离线模式**：网络不稳定时，设备仍要有本地兜底能力。

## 运行兼容性筛选器

创建 `edge_fit.py`：

```python
devices = [
    {"name": "edge-a", "memory_mb": 512, "power_w": 8, "offline": True},
    {"name": "edge-b", "memory_mb": 2048, "power_w": 15, "offline": False},
    {"name": "edge-c", "memory_mb": 4096, "power_w": 25, "offline": True},
]

model = {
    "name": "int8-small-classifier",
    "memory_mb": 700,
    "power_w": 10,
    "latency_ms": 65,
    "requires_offline": True,
}

for device in devices:
    reasons = []

    if device["memory_mb"] < model["memory_mb"]:
        reasons.append("memory")
    if device["power_w"] < model["power_w"]:
        reasons.append("power")
    if model["requires_offline"] and not device["offline"]:
        reasons.append("offline")

    status = "FIT" if not reasons else "CHECK " + ",".join(reasons)
    print(device["name"], status)
```

运行：

```bash
python edge_fit.py
```

预期输出：

```text
edge-a CHECK memory,power
edge-b CHECK offline
edge-c FIT
```

从左到右读结果：`edge-c` 不一定最快或最便宜，但它是唯一满足部署约束的设备。

## 让它更接近真实

把 `model["memory_mb"]` 从 `700` 改成 `350`，再运行一次。`edge-a` 仍然会失败，因为功耗不够。这说明边缘部署是多约束问题。

## 边缘部署检查清单

在说设备“可以上线”前，至少验证：

1. 能从冷启动正常运行。
2. 连续跑 30 分钟以上没有明显内存增长。
3. 网络断开时仍有可用兜底。
4. 保留足够日志，方便远程排障。
5. 有简单的回滚或替换方案。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
部署目标：本地推理、边缘设备、模型服务器或优化实验
工件：C++ 代码片段、基准测试、模型工件、服务配置或部署说明
指标：延迟、内存、吞吐量、模型大小、准确率下降或可靠性
失败检查：ABI/构建问题、硬件不匹配、量化损失或服务瓶颈
期望产出：可复现的部署或优化证据，而不只是理论笔记
```

## 常见错误

- 先选模型，再硬塞到小设备上。
- 只测一次推理，不测长时间运行。
- 默认设备一直在线。
- 忘记日志、缓存、输入图片也会占内存。

## 练习

给每个设备加 `price_usd`，选出通过全部检查且最便宜的设备。再加第二个模型，对比哪个设备能同时支持两个模型。

<details>
<summary>参考实现与讲解</summary>

答案应该先按约束过滤设备，再只在通过的设备里比较价格。一个便宜但内存、功耗或离线要求不达标的设备，不是有效部署目标。

第二个模型可以用共享检查，例如如果两个模型要同时运行，就检查 `device["memory_mb"] >= model_a["memory_mb"] + model_b["memory_mb"]`；如果一次只运行一个模型，就分别比较。最后要写清取舍：最佳设备是满足真实运行约束后最便宜的那个。

</details>
