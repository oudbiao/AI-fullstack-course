---
title: "12.2.1 图像生成路线图：提示词、控制、审核"
description: "图像生成章的简明实操路线图：设计提示词记录、保存参数、选择生成模式，并审核输出结果。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "图像生成指南, 扩散模型, Stable Diffusion, ControlNet, LoRA"
---
图像生成不是写一句提示词就结束，而是一套工作流：明确意图，记录提示词和参数，选择控制方式，对候选图做比较和审核。

## GAN、VAE、Diffusion 为什么先后出现

图像生成的发展可以先看成一条问题推动的历史线。每种方法都不是凭空替代前一种，而是在补前一种的短板。

```text
VAE -> GAN -> Diffusion -> Latent Diffusion / Stable Diffusion
```

| 方法 | 当时主要想解决什么 | 典型代价或短板 |
|---|---|---|
| VAE | 学一个连续、可采样的 latent space，可以重建、插值和生成相似样本 | 生成结果常偏平滑，细节不够锐利 |
| GAN | 让生成器和判别器对抗，追求更真实、更锐利的图像 | 训练不稳定，容易模式崩塌，控制难 |
| Diffusion | 从噪声逐步去噪，训练更稳定，质量和多样性更好 | 采样步数多，推理成本较高 |
| Latent Diffusion / Stable Diffusion | 在潜变量空间去噪，降低图像生成成本，并接入文本条件控制 | 需要理解 text encoder、U-Net、VAE、scheduler 和提示词记录 |

因此学习图像生成时，不要只背“哪个模型更强”。更重要的是看清：VAE 让 latent space 可采样，GAN 追求逼真图像，Diffusion 用逐步去噪换来稳定和质量，Stable Diffusion 把这套能力工程化到可控工作流里。

## 先看流程图

![图像生成章节学习流程图](/img/course/ch12-image-gen-chapter-flow.webp)

![Stable Diffusion 应用模式选择图](/img/course/ch12-sd-application-mode-selector-map.webp)

![Stable Diffusion 微调路线选择图](/img/course/ch12-sd-finetuning-route-choice-map.webp)

先养成一个习惯：记录你要什么、用了哪种模式、哪些 seed 或参数影响结果，以及导出前必须审核什么。

## 建一个提示词记录

```python
import json

brief = {
    "topic": "RAG basics",
    "audience": "beginners",
    "style": "clean editorial cover",
}
prompt = f"{brief['style']} for {brief['topic']}, friendly visual metaphor for {brief['audience']}, clear layout"
record = {
    "mode": "text-to-image",
    "prompt": prompt,
    "negative_prompt": "blurry, watermark, unreadable text",
    "seed": 42,
    "review": ["legibility", "copyright", "brand safety"],
}

print(json.dumps(record, indent=2))
```

预期输出：

```text
{
  "mode": "text-to-image",
  "prompt": "clean editorial cover for RAG basics, friendly visual metaphor for beginners, clear layout",
  "negative_prompt": "blurry, watermark, unreadable text",
  "seed": 42,
  "review": [
    "legibility",
    "copyright",
    "brand safety"
  ]
}
```

![图像生成提示词记录运行结果图](/img/course/ch12-image-prompt-record-result-map.webp)

如果提示词记录无法复现，后面就很难稳定改图。

## 按这个顺序学

| 步骤 | 阅读内容 | 练习产物 |
|---|---|---|
| 1 | 扩散直觉 | 解释加噪、去噪、seed、采样 |
| 2 | Stable Diffusion 组件 | 画出 text encoder、U-Net、VAE、latent space |
| 3 | 应用与控制 | 对比 text-to-image、image-to-image、inpainting、ControlNet、LoRA |

## 通过标准

你能写出提示词记录，解释为什么选择某种生成模式，保存 3 个候选图备注，并在导出前标记至少 1 个审核风险，就算通过本章。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要说清涉及哪些模态、输入输出契约是什么，以及文字、图像、音频或视频证据如何对齐。
2. 证据应包含真实媒体产物或 trace，并附上质量、安全和失败案例说明。
3. 自检时要能判断任务需要的是生成、理解、检索、工具编排还是人工复核，而不是把所有多模态问题都当成同一种 demo。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
提示词记录：提示词、负面要求、参考、seed/model，以及版本号
候选输出：生成或模拟的结果及选择原因
技术备注：扩散步、潜变量、cross-attention、LoRA 或应用模式
失败检查：提示漂移、风格不匹配、产物、版权、肖像或复核失败
期望产出：选定图片/版本记录加被拒候选说明
```
