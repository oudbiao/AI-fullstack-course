---
title: "6.4.1 RNN 路线图：按顺序处理序列"
sidebar_position: 0
description: "紧凑版 RNN 路线图：序列输入、隐藏状态、RNN、LSTM、GRU 和序列实战。"
keywords: [RNN 指南, 序列模型, LSTM, GRU, hidden state]
---

# 6.4.1 RNN 路线图：按顺序处理序列

RNN 面向有顺序的数据：文本、时间序列、点击流、传感器读数，以及任何前面步骤会影响后面步骤的输入。

## 先看序列流

![RNN 序列模型章节关系图](/img/course/ch06-rnn-chapter-flow.webp)

![RNN 隐藏状态滚动记忆图](/img/course/ch06-rnn-hidden-state-rolling-memory-map.webp)

| 概念 | 第一层意思 |
|---|---|
| sequence length | 时间步数量 |
| input size | 每一步的特征数 |
| hidden state | 滚动记忆 |
| LSTM / GRU | 门控记忆控制 |
| batch first | `[batch, seq_len, features]` 这种形状风格 |

## 跑一次 GRU 形状检查

创建 `rnn_first_loop.py`，安装 `torch` 后运行。

```python
import torch

sequence = torch.randn(2, 3, 5)
gru = torch.nn.GRU(input_size=5, hidden_size=4, batch_first=True)
outputs, hidden = gru(sequence)

print("sequence_shape:", tuple(sequence.shape))
print("outputs_shape:", tuple(outputs.shape))
print("hidden_shape:", tuple(hidden.shape))
```

预期输出：

```text
sequence_shape: (2, 3, 5)
outputs_shape: (2, 3, 4)
hidden_shape: (1, 2, 4)
```

读作：2 条序列，每条 3 步，每步 5 个特征。GRU 输出隐藏表示大小为 `4`。

## 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [6.4.2 RNN 基础](./01-rnn-basics.md) | 序列输入、隐藏状态、形状 |
| 2 | [6.4.3 LSTM 与 GRU](./02-lstm-gru.md) | 门控、长依赖、记忆控制 |
| 3 | [6.4.4 序列建模实战](./03-sequence-practice.md) | 滑动窗口、训练/评估循环 |

## 留下的证据

保留一条序列 shape 笔记：

```text
输入: [batch, seq_len, features]
输出：每一步一个隐藏表示
隐藏状态：压缩的滚动记忆
门控原因：LSTM/GRU 有助于保留或丢弃信息
基线：将序列模型与一个简单的朴素规则比较
```

## 通过标准

能读懂 `[batch, seq_len, features]`，把 hidden state 解释成滚动记忆，并知道 LSTM/GRU 是为长依赖而引入，就算通过。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要把 tensor、模型层、loss、`backward()` 和 optimizer 更新连成一个训练闭环。
2. 证据应包含可运行的小实验、tensor shape 检查，以及能解释的 loss 或验证曲线。
3. 自检时要能指出一个失败模式，例如 shape 不匹配、loss 不下降、过拟合、数据泄漏，或只会说 Attention/Transformer 名词却讲不出数据流。

</details>
