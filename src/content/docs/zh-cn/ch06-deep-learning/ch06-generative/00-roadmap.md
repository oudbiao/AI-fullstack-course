---
title: "6.6.1 生成模型路线图：采样、解码、审查"
description: "紧凑版生成模型路线图：潜在向量、GAN、VAE、生成输出和评估习惯。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "生成模型指南, GAN, VAE, latent vector, deep learning"
---
生成模型不是只预测标签，而是创造新样本。实用闭环是：采样潜在编码，解码输出，审查结果，比较版本。

## 先看生成流程

![生成模型章节关系图](/img/course/ch06-generative-chapter-flow.webp)

![GAN 对抗平衡图](/img/course/ch06-gan-adversarial-balance-map.webp)

| 概念 | 第一层意思 |
|---|---|
| latent vector | 用于生成的紧凑隐藏输入 |
| decoder / generator | 把潜在编码变成输出 |
| discriminator | 在 GAN 中判断真实还是生成 |
| VAE | 学习更平滑的潜在空间 |
| review | 生成结果仍需要人和指标检查 |

## GAN、VAE 和扩散模型怎么选？

第一次学生成模型时，先不要把它们当成互相替代的名字。它们的取舍不一样：

| 模型路线 | 适合先理解成 | 主要优势 | 常见难点 |
|---|---|---|---|
| GAN | 生成器和判别器对抗训练 | 样本可能很锐利，生成速度快 | 训练不稳定、模式崩塌 |
| VAE | 学一个可采样的 latent space | latent 连续、适合插值和异常检测 | 输出可能偏平滑，细节弱 |
| Diffusion | 从噪声逐步去噪 | 质量和可控性强，生态成熟 | 推理步数多，成本较高 |

所以本小章先讲 GAN 和 VAE，是为了看懂生成模型的两个基础思想：对抗生成和可采样 latent。后面 ch12 的扩散模型会继续回答“为什么逐步去噪成为图像生成主线”。

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

![极小 decoder 运行结果图](/img/course/ch06-generative-tiny-decoder-result-map.webp)

## 按这个顺序学

| 顺序 | 阅读 | 先抓住什么 |
|---|---|---|
| 1 | [6.6.2 GAN](/zh-cn/ch06-deep-learning/ch06-generative/01-gan/) | generator、discriminator、对抗平衡 |
| 2 | [6.6.3 VAE](/zh-cn/ch06-deep-learning/ch06-generative/02-vae/) | encoder、decoder、潜在空间 |

## 留下的证据

保留一条生成结果复盘笔记：

```text
潜在形状: 进入生成器/解码器的紧凑代码是什么
输出形状：输出的是哪种类似样本的对象
质量检查：它看起来合理吗，或能重建得好吗？
多样性检查：输出是否有变化，还是在塌缩？
信任规则：生成的输出始终需要复查
```

## 通过标准

能解释预测标签和生成样本的区别，并说明为什么生成结果需要审查而不能盲信，就算通过。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要把 tensor、模型层、loss、`backward()` 和 optimizer 更新连成一个训练闭环。
2. 证据应包含可运行的小实验、tensor shape 检查，以及能解释的 loss 或验证曲线。
3. 自检时要能指出一个失败模式，例如 shape 不匹配、loss 不下降、过拟合、数据泄漏，或只会说 Attention/Transformer 名词却讲不出数据流。

</details>
