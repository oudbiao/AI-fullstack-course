---
title: "A.8 学习卡住时的急救指南"
sidebar_position: 5
---

# A.8 学习卡住时的急救指南

![学习卡点排查地图](/img/course/appendix-troubleshooting-rescue-map.webp)

![最小复现与提问流程图](/img/course/appendix-debug-mre-help-flow.webp)

卡住时，先把“我学不会”变成“我能定位这个失败”。

## 先判断问题类型

| 现象 | 可能问题 | 第一反应 |
|---|---|---|
| `ModuleNotFoundError` | 环境错了或依赖没装 | 查 Python 和 `pip` 路径 |
| 找不到文件 | 工作目录或相对路径错了 | 打印 `Path.cwd()` |
| 代码能跑但结果奇怪 | 输入、标签或指标有问题 | 打印样本和中间值 |
| 训练不变好 | 数据、loss、学习率或标签格式问题 | 先让模型过拟合极小数据 |
| GPU 显存爆了 | batch、输入或模型太大 | 先减小 batch size |
| 项目太大无从下手 | 没有最小闭环 | 定义一个输入、一个处理、一个输出 |

## 先跑这些检查

```bash
python --version
which python
pip --version
pip list
pwd
ls
```

如果使用 NVIDIA GPU：

```bash
nvidia-smi
```

路径问题可以这样查：

```python
from pathlib import Path

print(Path.cwd())
print(Path("data").exists())
```

预期输出：

你当前所在目录会不同，形式大致是：

```text
/your/current/project
False
```

## 按这个顺序调试代码

1. 打印前 2 条输入和标签。
2. 打印 shape、长度和值范围。
3. 打印进入模型前的一个中间结果。
4. 打印计算指标前的一个模型输出。
5. 最后才改模型或参数。

最小检查示例：

```python
texts = ["refund request", "invoice copy", "shipping delay"]
labels = ["support", "billing", "support"]

print("samples:", len(texts))
print("first texts:", texts[:2])
print("first labels:", labels[:2])
print("label set:", sorted(set(labels)))
```

预期输出：

```text
samples: 3
first texts: ['refund request', 'invoice copy']
first labels: ['support', 'billing']
label set: ['billing', 'support']
```

## 用完整问题请求帮助

```text
我正在做什么：
我期待看到什么：
实际发生了什么：
完整错误最后 20 行：
我已经尝试过什么：
最小可复现代码：
```

## 养成最小复现习惯

项目很乱时，先缩到能跑：

```python
def predict(x):
    return x * 2

data = [1, 2, 3]
preds = [predict(x) for x in data]
print(preds)
```

预期输出：

```text
[2, 4, 6]
```

然后一层一层把真实逻辑加回去。哪一层加回去后坏了，就检查哪一层。

## 暂停还是继续？

| 情况 | 更好的动作 |
|---|---|
| 已经随机尝试 30 分钟 | 停下来写假设 |
| 复制命令但说不清作用 | 停下来检查环境 |
| 已经有 1-2 个清晰假设 | 继续测试 |
| 知道下一个可观察结果 | 继续推进 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
症状：准确的错误信息、命令、输入和环境
最小复现：仍然会失败的最小代码或命令
假设：依赖、路径、数据、API、模型或浏览器/运行时问题
下一次探测：先检查一个命令或日志，再改很多东西
期望产出：一份可复现的 bug 说明和经过测试的修复或回退方案
```
