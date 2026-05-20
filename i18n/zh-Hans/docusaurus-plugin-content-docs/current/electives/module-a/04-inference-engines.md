---
title: "E.A.4 推理引擎"
sidebar_position: 4
description: "通过模型格式、硬件、延迟、吞吐和维护成本，学会选择合适的推理引擎。"
keywords: [inference engine, ONNX Runtime, TensorRT, OpenVINO, execution graph, deployment]
---

# E.A.4 推理引擎

![推理引擎与硬件适配图](/img/course/elective-inference-engine-hardware.webp)

![推理引擎选型矩阵图](/img/course/elective-inference-engine-selection-matrix.webp)

推理引擎是训练好的模型和真实硬件之间的运行层。模型说明“要算什么”，推理引擎决定“怎样在 CPU、GPU、NPU 或边缘硬件上高效执行”。

这一节先做一个选型小练习。不要把某个引擎记成永远最好，要看部署约束。

## 准备内容

- Python 3.10+
- 不需要第三方包
- 花 5 分钟运行并修改一个评分脚本

## 关键术语

- **Latency（延迟）**：一次请求从进入到拿到结果要等多久。
- **Throughput（吞吐）**：系统每秒能完成多少请求。
- **Backend（后端）**：面向具体硬件的执行路径，例如 CPU、CUDA、TensorRT、OpenVINO。
- **ONNX**：常用的模型交换格式。
- **Operator（算子）**：模型图里的一个操作，例如矩阵乘法、卷积、归一化。

## 运行引擎选择器

创建 `engine_selector.py`：

```python
engines = [
    {
        "name": "ONNX Runtime",
        "hardware": ["cpu", "nvidia"],
        "formats": ["onnx"],
        "latency": "medium",
        "ops": "easy",
    },
    {
        "name": "TensorRT",
        "hardware": ["nvidia"],
        "formats": ["onnx", "engine"],
        "latency": "low",
        "ops": "hard",
    },
    {
        "name": "OpenVINO",
        "hardware": ["cpu", "intel"],
        "formats": ["onnx", "ir"],
        "latency": "low",
        "ops": "medium",
    },
]

need = {"hardware": "nvidia", "format": "onnx", "latency": "low"}

for engine in engines:
    score = 0
    score += 2 if need["hardware"] in engine["hardware"] else -3
    score += 2 if need["format"] in engine["formats"] else -2
    score += 1 if need["latency"] == engine["latency"] else 0
    score -= 1 if engine["ops"] == "hard" else 0
    engine["score"] = score

best = max(engines, key=lambda item: item["score"])

for engine in engines:
    print(engine["name"], engine["score"])

print("selected:", best["name"])
```

运行：

```bash
python engine_selector.py
```

预期输出：

```text
ONNX Runtime 4
TensorRT 4
OpenVINO 0
selected: ONNX Runtime
```

这里 ONNX Runtime 和 TensorRT 分数相同，脚本选择了第一个。这个结果是故意保留的：真实部署里，如果更快的路线会增加构建和维护成本，简单路线反而可能更适合作为第一版。

## 改一个约束

把：

```python
need = {"hardware": "nvidia", "format": "onnx", "latency": "low"}
print(need)
```

第一个片段的预期输出：

```text
{'hardware': 'nvidia', 'format': 'onnx', 'latency': 'low'}
```

改成：

```python
need = {"hardware": "intel", "format": "onnx", "latency": "low"}
print(need)
```

第二个片段的预期输出：

```text
{'hardware': 'intel', 'format': 'onnx', 'latency': 'low'}
```

再次运行。预期结果：

```text
ONNX Runtime -1
TensorRT -2
OpenVINO 5
selected: OpenVINO
```

核心结论很简单：硬件变了，引擎选择也会变。

## 实用选型顺序

做高级调优前，先按这个顺序判断：

1. 确认目标硬件。
2. 确认引擎能加载的模型格式。
3. 检查是否有不支持的算子。
4. 用相同输入尺寸比较延迟和吞吐。
5. 选择能达标且最容易维护的引擎。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
deployment_target: local inference, edge device, model server, or optimization experiment
artifact: C++ snippet, benchmark, model artifact, serving config, or deployment note
metric: latency, memory, throughput, model size, accuracy drop, or reliability
failure_check: ABI/build issue, hardware mismatch, quantization loss, or serving bottleneck
Expected_output: reproducible deployment or optimization evidence, not only theory notes
```

## 常见错误

- 因为 TensorRT 快，就不考虑团队是否能维护 engine 构建流程。
- 用很小的输入测试，生产输入变大后才发现很慢。
- 到上线前一周才发现有算子不支持。

## 练习

给每个引擎加一个 `memory` 字段；如果超过设备内存，就扣 1 分。然后分别用 CPU-only、NVIDIA GPU、Intel 设备三种场景重新选择。
