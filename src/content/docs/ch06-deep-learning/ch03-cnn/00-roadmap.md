---
title: "6.3.1 CNN Roadmap: Turn Images Into Feature Maps"
description: "A compact CNN roadmap: convolution, channels, feature maps, classic architectures, transfer learning, and image classification practice."
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "CNN guide, convolution, image classification, transfer learning, feature map"
---
CNNs learn local visual patterns. Instead of reading an image as one flat row of numbers, they scan small regions and build feature maps.

## Look at the Image Flow First

![CNN chapter relationship diagram](/img/course/ch06-cnn-chapter-flow-en.webp)

![CNN receptive field growth map](/img/course/ch06-cnn-receptive-field-growth-map-en.webp)

| Concept | First meaning |
|---|---|
| channel | color or learned feature dimension |
| kernel | small sliding filter |
| feature map | output after filters scan the image |
| pooling / stride | shrink spatial size |
| transfer learning | reuse a pretrained vision backbone |

## Run One Convolution

Create `cnn_first_loop.py` and run it after installing `torch`.

```python
import torch

image = torch.randn(1, 3, 32, 32)
conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
features = conv(image)

print("input_shape:", tuple(image.shape))
print("feature_shape:", tuple(features.shape))
```

Expected output:

```text
input_shape: (1, 3, 32, 32)
feature_shape: (1, 8, 32, 32)
```

Read the shape as `[batch, channels, height, width]`. The convolution changed `3` input channels into `8` learned feature channels.

## Learn in This Order

| Order | Read | What to practice |
|---|---|---|
| 1 | [6.3.2 Convolution Basics](/ch06-deep-learning/ch03-cnn/01-convolution-basics/) | kernel, stride, padding, channel |
| 2 | [6.3.3 CNN Structure](/ch06-deep-learning/ch03-cnn/02-cnn-structure/) | conv block, pooling, classifier head |
| 3 | [6.3.4 Classic Architectures](/ch06-deep-learning/ch03-cnn/03-classic-architectures/) | LeNet, AlexNet, VGG, ResNet intuition |
| 4 | [6.3.5 Transfer Learning](/ch06-deep-learning/ch03-cnn/04-transfer-learning/) | frozen backbone, fine-tuning |
| 5 | [6.3.6 Image Classification Practice](/ch06-deep-learning/ch03-cnn/05-image-classification-practice/) | dataset, training, prediction examples |

## Evidence to Keep

Keep one CNN shape note:

```text
input: [batch, channels, height, width]
conv_output: out_channels becomes new feature maps
spatial_change: stride/padding/pooling change height and width
classifier_bridge: conv features eventually become class logits
transfer_choice: freeze first, fine-tune only if validation improves
```

## Pass Check

You pass this roadmap when you can explain what changed between input image shape and feature map shape, and why pretrained CNN backbones are useful for small datasets.

<details>
<summary>Check reasoning and explanation</summary>

1. A passing answer connects tensors, model layers, loss, `backward()`, and optimizer updates into one training loop.
2. The evidence should include a runnable mini experiment, tensor-shape checks, and a loss or validation curve you can explain.
3. A good self-check names one failure mode such as shape mismatch, no loss decrease, overfitting, data leakage, or using Attention/Transformer words without explaining the data flow.

</details>
