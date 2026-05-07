---
title: "6.6.1 生成模型路线图：采样、解码、审查"
sidebar_position: 0
description: "紧凑版生成模型路线图：潜在向量、GAN、VAE、生成输出和评估习惯。"
keywords: [生成模型指南, GAN, VAE, latent vector, deep learning]
---

# 6.6.1 生成模型路线图：采样、解码、审查

生成模型不是只预测标签，而是创造新样本。实用闭环是：采样潜在编码，解码输出，审查结果，比较版本。

## 先看生成流程

![生成模型章节关系图](/img/course/ch06-generative-chapter-flow.png)

![GAN 对抗平衡图](/img/course/ch06-gan-adversarial-balance-map.png)

| 概念 | 第一层意思 |
|---|---|
| latent vector | 用于生成的紧凑隐藏输入 |
| decoder / generator | 把潜在编码变成输出 |
| discriminator | 在 GAN 中判断真实还是生成 |
| VAE | 学习更平滑的潜在空间 |
| review | 生成结果仍需要人和指标检查 |

## 跑一个极小 decoder

创建 `generative_first_loop.py`，安装 `torch` 后运行。

```python
import torch

torch.manual_seed(0)
latent = torch.randn(2, 4)
decoder = torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.Tanh())
generated = decoder(latent)

print("latent_shape:", tuple(latent.shape))
print("generated_shape:", tuple(generated.shape))
print("value_range:", round(generated.min().item(), 3), round(generated.max().item(), 3))
```

预期输出：

```text
latent_shape: (2, 4)
generated_shape: (2, 6)
value_range: -0.863 0.695
```

这还不是真正的生成器，只是展示核心形状直觉：小的 latent vector 可以被解码成更大的输出。

## 按这个顺序学

| 顺序 | 阅读 | 先抓住什么 |
|---|---|---|
| 1 | [6.6.2 GAN](./01-gan.md) | generator、discriminator、对抗平衡 |
| 2 | [6.6.3 VAE](./02-vae.md) | encoder、decoder、潜在空间 |

## 通过标准

能解释预测标签和生成样本的区别，并说明为什么生成结果需要审查而不能盲信，就算通过。
