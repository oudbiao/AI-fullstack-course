---
title: "3.2 Principles of Convolution Operations 🔧"
sidebar_position: 1
description: "From local image patterns, convolution kernels, stride, and padding to receptive fields — the first real step toward understanding what CNNs actually do."
keywords: [convolution, convolution kernel, CNN, stride, padding, receptive field, image features]
---

# Principles of Convolution Operations

![CNN convolution kernel sliding illustration](/img/course/cnn-convolution-kernel-en.png)

:::tip Where this section fits
If the earlier neurons and MLP sections taught you “what a neural network can compute,” then this convolution section answers an even more important question:

> **How do neural networks look at images efficiently?**

This section is the starting point of the entire CV storyline. Later, when you study classification, detection, and segmentation, you’ll keep coming back to the intuition here.
:::

## Learning Objectives

- Understand why image tasks cannot be solved directly with a fully connected layer in a naive way
- Build an intuitive understanding of convolution kernels, local connections, and parameter sharing
- Manually compute a minimal convolution example and truly see where each output value comes from
- Master `stride`, `padding`, and output size calculation
- Understand multi-channel convolution and receptive fields
- Be able to read the most basic `Conv2d` in PyTorch

---

## How this connects to the earlier MLP storyline

If you just finished learning MLPs, you can think of this section as:

- MLP already taught you “what the network can compute”
- This convolution section starts answering “how the network should look at images more appropriately”

In other words, this section is not overthrowing “linear layer + activation function.” It is improving how the input is organized:

- No longer flattening the image into a long vector in a rough, naive way
- Instead, letting the network look through local windows while preserving spatial structure

That is the most important structural change in vision tasks.

## 1. Why do image tasks need convolution?

### 1.1 First, let’s look at the problem with “just flattening”

Suppose you have a `32 x 32` grayscale image.

If you directly flatten it into a vector and feed it into a fully connected layer:

- The input dimension is `32 * 32 = 1024`
- If the next layer has 512 neurons, you need `1024 * 512 = 524288` weights

If the image is a bit larger, for example `224 x 224 x 3`:

- The input dimension becomes `150528`
- The number of parameters explodes instantly

Even worse, after flattening, the most important structure in the image gets destroyed:

- Relationships between nearby pixels
- Edges, textures, and local patterns
- Spatial structure

So:

> **An image is not ordinary tabular data.**

What matters most is not “how many numbers it has,” but “how those numbers sit next to each other in space.”

### 1.2 What exactly does convolution solve?

Convolution does two especially important things:

1. It looks only at local regions instead of the whole image at once
2. It reuses the same set of parameters as it slides across the whole image

These two design choices correspond to:

- **Local connection**
- **Parameter sharing**

You can think of convolution as:

> **Taking a small template and sliding it across the image to look for a certain local pattern.**

For example:

- Vertical lines
- Horizontal lines
- Edges
- Corner points
- Textures

### 1.3 What you should focus on first in this section is not the kernel itself

Focus first on these two “whys”:

1. Why can’t we just flatten the image directly?
2. Why must local neighborhood relationships be preserved?

Once these two points are clear, concepts like convolution kernels, stride, and padding will no longer feel like pure memorization.

---

## 2. What is a convolution kernel?

### 2.1 The easiest analogy to understand

A convolution kernel (kernel / filter) is like a tiny “transparent template.”

You place it over a small region of the image:

- Multiply corresponding values
- Then add them up

You get a score.

You can think of that score as:

> How much this region matches the pattern the kernel is looking for.

### 2.2 Minimal runnable example: doing a convolution by hand

```python
import numpy as np

# 4x4 image
image = np.array([
    [1, 2, 0, 0],
    [5, 3, 0, 4],
    [2, 1, 3, 1],
    [0, 2, 1, 2]
], dtype=np.float32)

# 2x2 convolution kernel
kernel = np.array([
    [1, 0],
    [0, -1]
], dtype=np.float32)

out = np.zeros((3, 3), dtype=np.float32)

for i in range(3):
    for j in range(3):
        patch = image[i:i + 2, j:j + 2]
        out[i, j] = np.sum(patch * kernel)

print("image =\n", image)
print("kernel =\n", kernel)
print("output =\n", out)
```

### 2.3 How is the first output value computed?

The top-left `2x2` patch is:

```text
[[1, 2],
 [5, 3]]
```

The convolution kernel is:

```text
[[ 1, 0],
 [ 0,-1]]
```

Element-wise multiplication:

```text
[[ 1*1, 2*0],
 [ 5*0, 3*(-1)]]
```

Summing them:

```text
1 + 0 + 0 - 3 = -2
```

So the top-left output value is `-2`.

That is the core computation of convolution.

### 2.4 The most important thing to remember about convolution kernels is not “they slide,” but “they search for patterns”

A better beginner-friendly way to say it is:

- A convolution kernel is a small pattern detector

Different kernels may respond more strongly to these patterns:

- Edges
- Direction changes
- Small textures
- Local corners

So what a convolution layer really does is not “multiply the image around,” but:

> **Extract low-level local patterns layer by layer, and hand them to later layers for further combination.**

---

## 3. Why can convolution detect edges?

### 3.1 Because it is essentially comparing local differences

If a convolution kernel is designed to do “left minus right” or “top minus bottom,” it becomes especially sensitive to boundaries.

For example, this kernel:

```text
[[ 1,  0],
 [ 0, -1]]
```

responds to local structures like “bright in the upper-left, dark in the lower-right.”

If a region of the image is smooth and the pixel values are similar, the convolution result is often close to 0.
If the local change is sharp, the convolution result becomes larger.

### 3.2 Let’s look at another edge kernel

```python
import numpy as np

image = np.array([
    [0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0]
], dtype=np.float32)

kernel = np.array([
    [-1, 1]
], dtype=np.float32)

out = np.zeros((5, 4), dtype=np.float32)
for i in range(5):
    for j in range(4):
        patch = image[i:i + 1, j:j + 2]
        out[i, j] = np.sum(patch * kernel)

print("output =\n", out)
```

You’ll see that the output is most obvious near the boundary where the values change from `0` to `1`.

---

## 4. What exactly are stride and padding?

### 4.1 Stride: how far to move each time

`stride` can be understood as how many steps the convolution kernel moves each time.

- `stride = 1`: move 1 step each time
- `stride = 2`: move 2 steps each time

The larger the stride:

- The smaller the output
- The faster the computation
- The more detail is lost

### 4.2 Padding: add a border around the image first

If you do not use padding, the convolution kernel stops when it reaches the edge, and the output size becomes smaller.

Padding is used to:

- Preserve edge information
- Control the output size

The most common approach is to pad with 0, also called zero padding.

### 4.3 When you first learn stride and padding, where do people usually get confused?

The most confusing part is usually not the formula itself, but:

- Not understanding that they control “how finely you look” and “how big the output is”

A more stable way to remember them is:

- `stride` is more like “how far to move each time”
- `padding` is more like “whether to add a border first”

So fundamentally, both affect two things:

- How much information is preserved
- How computation and output size change

![Convolution stride padding and output size change diagram](/img/course/ch06-conv-stride-padding-size-map-en.png)

:::tip Reading hint
When reading this diagram, think of `stride` as the sliding step and `padding` as the border you add around the image. The larger the step, the smaller the output; the more padding, the more edge information is preserved. The output size formula is just the result of these two actions.
:::

### 4.4 Output size formula

For 2D convolution:

> `output = floor((input + 2*padding - kernel_size) / stride) + 1`

For example:

- Input width/height: `6`
- Kernel size: `3`
- padding: `1`
- stride: `2`

Then the output size is:

> `floor((6 + 2*1 - 3) / 2) + 1 = floor(5/2) + 1 = 2 + 1 = 3`

### 4.5 Runnable example: verify the output size

```python
import torch
from torch import nn

x = torch.randn(1, 1, 6, 6)  # batch=1, channel=1, H=6, W=6

conv = nn.Conv2d(
    in_channels=1,
    out_channels=2,
    kernel_size=3,
    stride=2,
    padding=1
)

y = conv(x)

print("input shape :", x.shape)
print("output shape:", y.shape)
```

In the output, you’ll see that both height and width become `3`.

---

## 5. Multi-channel convolution: how do color images work?

### 5.1 The difference between grayscale and RGB images

A grayscale image usually has the shape:

- `H x W`

An RGB image is often written in deep learning as:

- `C x H x W`

where:

- `C = 3`
- corresponding to the R/G/B channels

### 5.2 A convolution kernel also “grows channels”

If the input is an RGB image, then a convolution kernel is no longer just `3 x 3`, but:

> `3 x 3 x 3`

That means:

- Look at a `3x3` area in the red channel
- Look at a `3x3` area in the green channel
- Look at a `3x3` area in the blue channel

Then add the results from the three channels together, plus a bias, to get one output value.

### 5.3 Multiple convolution kernels = multiple output channels

If you have 16 convolution kernels, you will get 16 feature maps.
That is why `Conv2d` uses:

- `in_channels`
- `out_channels`

```python
import torch
from torch import nn

x = torch.randn(2, 3, 32, 32)  # batch=2, RGB images
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
y = conv(x)

print("input shape :", x.shape)
print("output shape:", y.shape)
```

Here the output shape will be:

- `[2, 8, 32, 32]`

That means:

- 2 images
- Each image is transformed into 8-channel feature maps

---

## 6. Receptive field: why can deep networks see a larger area?

### 6.1 The intuition behind receptive field

The receptive field refers to:

> How large a region of the original image one position in the output can “see.”

A single `3x3` convolution layer can only see a local `3x3` region.

But if you stack multiple layers:

- The first layer sees `3x3`
- The second layer looks at the first layer’s output with another `3x3`

Then the second layer indirectly sees a larger area of the original image.

### 6.2 Why is this important?

Because image understanding is usually hierarchical:

- Early layers: edges, textures
- Middle layers: corners, local shapes
- Deep layers: object parts, overall semantics

The reason CNNs are powerful is not that “convolution itself is magical,” but that:

> **Small local features can be combined layer by layer into more abstract, larger patterns.**

![CNN receptive field grows layer by layer feature combination diagram](/img/course/ch06-cnn-receptive-field-growth-map-en.png)

:::tip Reading hint
Read this diagram from shallow to deep: the first layer sees only small edges, the second layer combines them into local shapes, and later layers gradually see larger object parts. The strength of CNNs is not a single convolution kernel, but that local patterns can be combined layer by layer into higher-level semantics.
:::

---

## 7. What exactly does a convolution layer do in PyTorch?

### 7.1 The most basic `Conv2d`

```python
import torch
from torch import nn

x = torch.randn(1, 1, 8, 8)

conv = nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1
)

y = conv(x)

print("input shape :", x.shape)
print("output shape :", y.shape)
print("weight shape :", conv.weight.shape)
print("bias shape :", conv.bias.shape)
```

Here:

- `out_channels=4` means there are 4 convolution kernels
- `conv.weight.shape = [4, 1, 3, 3]`
  - 4 output channels
  - each kernel looks at 1 input channel
  - kernel size `3x3`

### 7.2 Why is an activation function often added after a convolution layer?

Just like in MLPs:

- The convolution first performs a linear transformation
- Then the activation function introduces nonlinearity

A typical pattern is:

```python
nn.Conv2d(...)
nn.ReLU()
```

---

## 8. Common beginner mistakes

### 8.1 Treating convolution as a “magic feature extractor”

Convolution is not magic. In essence, it is just:

- A small window
- Element-wise multiplication
- Summation
- Sliding

### 8.2 Mixing up shapes

One of the most common mistakes is confusing:

- `H x W x C`
- `C x H x W`

In PyTorch, the usual format is:

- `N x C x H x W`

### 8.3 Not knowing how to calculate output size

Many errors are not because the model cannot learn, but because the dimensions do not match.
So you must be able to calculate the sizes for `kernel_size / stride / padding`.

---

## Summary

The most important thing in this section is not memorizing the word “convolution,” but building three stable intuitions:

1. Image tasks need to preserve spatial structure, so we cannot simply flatten and use a fully connected layer in a brute-force way
2. A convolution kernel repeatedly searches for local patterns across the whole image
3. Stacking multiple convolution layers allows the model to combine local features step by step into higher-level visual understanding

Once you understand these three points, you won’t treat convolution layers as a black box when you later study CNN architectures, classic models, and object detection.

---

## Exercises

1. Change the `2x2` convolution kernel in this section to other values and observe how the output changes.
2. Manually compute one output position, then compare it with the code result.
3. Rewrite a convolution layer with `kernel_size=5` and `stride=2` in PyTorch and verify the output size.
4. Think about this: if an object in an image shifts slightly as a whole, why is convolution usually more robust than a fully connected layer?
