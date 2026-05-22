---
title: "6.5.1 Transformer 路线图：Attention 让 token 互相看见"
description: "紧凑版 Transformer 路线图：attention、QKV、全局上下文、Transformer block 和现代 LLM 基础。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Transformer 指南, attention mechanism, QKV, self-attention, deep learning"
---
Transformer 是从深度学习走向现代 LLM 的桥。第一直觉很简单：每个 token 可以决定哪些其他 token 更重要。

## 先看 Attention 流程

![Transformer 章节关系图](/img/course/ch06-transformer-chapter-flow.webp)

![Transformer 全局上下文建模图](/img/course/ch06-transformer-global-context-map.webp)

| 概念 | 第一层意思 |
|---|---|
| token | 序列里的一个位置 |
| Q / K / V | token 的 query、key、value 视角 |
| attention weight | 一个 token 看另一个 token 的程度 |
| block | attention 加前馈层反复精炼表示 |
| mask | 生成时防止看见未来 token |

## 跑一次 Attention 形状检查

创建 `transformer_first_loop.py`，安装 `torch` 后运行。

```python
import torch

attention = torch.nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
tokens = torch.randn(1, 4, 8)
output, weights = attention(tokens, tokens, tokens)

print("tokens_shape:", tuple(tokens.shape))
print("output_shape:", tuple(output.shape))
print("attention_shape:", tuple(weights.shape))
```

预期输出：

```text
tokens_shape: (1, 4, 8)
output_shape: (1, 4, 8)
attention_shape: (1, 4, 4)
```

`attention_shape` 是 `[batch, query_position, key_position]`：4 个位置里的每个位置都能看 4 个位置。

## 按这个顺序学

| 顺序 | 阅读 | 先抓住什么 |
|---|---|---|
| 1 | [6.5.2 Attention 机制](/zh-cn/ch06-deep-learning/ch05-transformer/01-attention-mechanism/) | QKV、attention 权重、mask |
| 2 | [6.5.3 Transformer 架构](/zh-cn/ch06-deep-learning/ch05-transformer/02-transformer-architecture/) | block 结构、残差、前馈层 |

## 留下的证据

保留一条 attention 桥接笔记：

```text
token 张量形状：[batch, seq_len, embed_dim]
注意力形状：[batch, query_position, key_position]
QKV 含义：Q/K 负责匹配，V 携带内容
mask 原因：生成不能看到未来 token
LLM 桥接：解码器块把 token 上下文转换为下一个 token 的 logits
```

## 通过标准

能读懂 attention 权重形状，解释为什么 attention 带来全局上下文，并把 mask 和文本生成联系起来，就算通过。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要把 tensor、模型层、loss、`backward()` 和 optimizer 更新连成一个训练闭环。
2. 证据应包含可运行的小实验、tensor shape 检查，以及能解释的 loss 或验证曲线。
3. 自检时要能指出一个失败模式，例如 shape 不匹配、loss 不下降、过拟合、数据泄漏，或只会说 Attention/Transformer 名词却讲不出数据流。

</details>
