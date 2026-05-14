---
title: "12.2.1 图像生成路线图：提示词、控制、审核"
sidebar_position: 0
description: "图像生成章的简明实操路线图：设计提示词记录、保存参数、选择生成模式，并审核输出结果。"
keywords: [图像生成指南, 扩散模型, Stable Diffusion, ControlNet, LoRA]
---

# 12.2.1 图像生成路线图：提示词、控制、审核

图像生成不是写一句提示词就结束，而是一套工作流：明确意图，记录提示词和参数，选择控制方式，对候选图做比较和审核。

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
