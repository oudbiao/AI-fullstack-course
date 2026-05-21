---
title: "E.A C++ 与模型部署路线图"
sidebar_position: 0
description: "C++ 与模型部署选修模块的简明实操路线图：从运行时基础到优化、推理引擎、边缘部署、服务化和交付项目。"
---

# E.A C++ 与模型部署路线图

当 Python 模型已经能跑，但延迟、内存、打包或服务成本变成真正问题时，再回来学这个选修模块。

## 先看部署路线

![C++ 与模型部署模块学习地图](/img/course/elective-cpp-deployment-module-map.webp)

![C++ 运行时内存图](/img/course/elective-cpp-runtime-memory.webp)

核心问题很简单：你能不能把模型输出变成快速、可度量、可部署的推理路径。

## 跑最小 C++ 推理步骤

创建 `demo.cpp`：

```cpp
#include <iostream>
#include <vector>

int main() {
    std::vector<float> logits = {1.2f, 0.3f, 2.1f};
    int best_index = 0;

    for (int i = 1; i < static_cast<int>(logits.size()); ++i) {
        if (logits[i] > logits[best_index]) {
            best_index = i;
        }
    }

    std::cout << "best_class=" << best_index << "\n";
    std::cout << "score=" << logits[best_index] << "\n";
    return 0;
}
```

运行：

```bash
c++ -std=c++17 demo.cpp -o demo
./demo
```

预期输出：

```text
best_class=2
score=2.1
```

这是最小部署习惯：输入类似 tensor 的数值，计算决策，打印可复现结果。

## 按这个顺序学

| 步骤 | 课程 | 练习产物 |
|---|---|---|
| 1 | [E.A.1 C++ 基础](./01-cpp-basics.md) | 编译并运行一个小推理辅助程序 |
| 2 | [E.A.2 C++ 进阶](./02-cpp-advanced.md) | 解释 ownership、RAII 和安全释放资源 |
| 3 | [E.A.3 模型优化](./03-model-optimization.md) | 比较延迟、内存和精度取舍 |
| 4 | [E.A.4 推理引擎](./04-inference-engines.md) | 根据硬件和模型格式选择引擎 |
| 5 | [E.A.5 边缘部署](./05-edge-deployment.md) | 说出边缘约束并导出检查表 |
| 6 | [E.A.6 模型服务化](./06-model-serving.md) | 设计带版本和指标的服务 |
| 7 | [E.A.7 项目](./07-projects.md) | 交付一个小型部署证据包 |

## 通过标准

你能编译一个 C++ 示例，解释部署取舍，记录延迟或内存证据，并把结果接到 [选修实操工作坊](../hands-on-elective-workshop.md)，就算通过本模块。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
部署目标：本地推理、边缘设备、模型服务器或优化实验
工件：C++ 代码片段、基准测试、模型工件、服务配置或部署说明
指标：延迟、内存、吞吐量、模型大小、准确率下降或可靠性
失败检查：ABI/构建问题、硬件不匹配、量化损失或服务瓶颈
期望产出：可复现的部署或优化证据，而不只是理论笔记
```

<details>
<summary>检查思路与讲解</summary>

一个合格答案会说明为什么这里适合用 C++：更稳定的运行时、更可控的内存，以及更贴近部署目标的路径。证据可以是一次编译输出、一条延迟或内存记录，再加上一句说明它为什么能接到后续实操工作坊。

如果只写“能运行”，还不够。要把可复现产物和部署取舍一起留下来。

</details>
