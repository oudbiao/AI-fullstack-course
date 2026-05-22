---
title: "6.3.2 Convolution Basics"
description: "Learn convolution by hand and in PyTorch: kernels, local patterns, stride, padding, channels, output shapes, and receptive fields."
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "convolution, convolution kernel, CNN, stride, padding, receptive field, image features"
---
:::tip[Section Overview]
Convolution is how CNNs look at images without flattening away spatial structure. This page starts with a picture, then computes a convolution by hand, then verifies the same ideas with `nn.Conv2d`.
:::
## Learning Objectives

- Explain why flattening an image too early is wasteful.
- Compute one convolution output value by hand.
- Understand kernel, stride, padding, channel, and feature map.
- Verify output shapes with PyTorch.
- Explain why stacking convolutions grows the receptive field.

---

## Look at the Sliding Window First

![CNN convolution kernel sliding illustration](/img/course/cnn-convolution-kernel-en.webp)

Read the picture like this:

```text
small window -> multiply by kernel -> sum -> one output value -> slide and repeat
```

A convolution kernel is a small pattern detector. It does not look at the whole image at once. It scans local regions and writes a score into a feature map.

## Why Not Flatten the Image First?

A `32 x 32` grayscale image has `1024` pixels. A fully connected layer with `512` outputs would need:

```text
1024 * 512 = 524288 weights
```

A `224 x 224 x 3` color image has `150528` input values. A naive fully connected layer explodes in parameters and ignores where pixels are located.

Convolution fixes two problems:

| Problem with early flattening | Convolution idea |
|---|---|
| nearby pixels lose their spatial relationship | look at local windows |
| every position needs separate weights | reuse the same kernel everywhere |
| parameter count grows quickly | share parameters across the image |

The two core terms are:

- local connection: each output looks at a small area;
- parameter sharing: the same kernel scans many positions.

## Lab 1: Compute Convolution by Hand

```python
import numpy as np

image = np.array(
    [
        [1, 2, 0, 0],
        [5, 3, 0, 4],
        [2, 1, 3, 1],
        [0, 2, 1, 2],
    ],
    dtype=np.float32,
)

kernel = np.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=np.float32,
)

out = np.zeros((3, 3), dtype=np.float32)
for i in range(3):
    for j in range(3):
        patch = image[i : i + 2, j : j + 2]
        out[i, j] = np.sum(patch * kernel)

print("manual_conv_lab")
print(out)
```

Expected output:

```text
manual_conv_lab
[[-2.  2. -4.]
 [ 4.  0. -1.]
 [ 0.  0.  1.]]
```

Top-left output value:

```text
patch = [[1, 2],
         [5, 3]]

kernel = [[ 1,  0],
          [ 0, -1]]

score = 1*1 + 2*0 + 5*0 + 3*(-1) = -2
```

That is the whole core of convolution.

## Lab 2: Use a Kernel as an Edge Detector

This horizontal kernel compares neighboring pixels from left to right.

```python
import numpy as np

image = np.array(
    [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
    ],
    dtype=np.float32,
)

kernel = np.array([[-1, 1]], dtype=np.float32)

out = np.zeros((5, 4), dtype=np.float32)
for i in range(5):
    for j in range(4):
        patch = image[i : i + 1, j : j + 2]
        out[i, j] = np.sum(patch * kernel)

print("edge_lab")
print(out)
```

Expected output:

```text
edge_lab
[[0. 0. 0. 0.]
 [0. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 1. 0. 0.]
 [0. 0. 0. 0.]]
```

The `1` values appear where the image changes from `0` to `1`. That is why early CNN layers often learn edge-like filters.

## Stride, Padding, and Output Size

![Convolution stride padding and output size change diagram](/img/course/ch06-conv-stride-padding-size-map-en.webp)

| Term | Meaning | Effect |
|---|---|---|
| `kernel_size` | window size | larger kernel sees more local area |
| `stride` | how far the kernel moves each step | larger stride makes output smaller |
| `padding` | border added around input | preserves edge information and controls size |

Output size for one spatial dimension:

```text
output = floor((input + 2*padding - kernel_size) / stride) + 1
```

Example:

```text
input=6, kernel_size=3, padding=1, stride=2
output = floor((6 + 2*1 - 3) / 2) + 1 = 3
```

Verify in PyTorch:

```python
import torch
from torch import nn

x = torch.randn(1, 1, 6, 6)
conv = nn.Conv2d(
    in_channels=1,
    out_channels=2,
    kernel_size=3,
    stride=2,
    padding=1,
)
y = conv(x)

print("size_lab")
print("input:", tuple(x.shape))
print("output:", tuple(y.shape))
```

Expected output:

```text
size_lab
input: (1, 1, 6, 6)
output: (1, 2, 3, 3)
```

Read the shape as `[batch, channels, height, width]`.

## Multi-Channel Convolution

Color images have three input channels: red, green, and blue. In PyTorch, a batch of RGB images usually has shape:

```text
[batch, 3, height, width]
```

A `3 x 3` convolution over an RGB image actually has kernel shape:

```text
[out_channels, in_channels, kernel_height, kernel_width]
```

Run it:

```python
import torch
from torch import nn

x = torch.randn(2, 3, 32, 32)
conv = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
y = conv(x)

print("channel_lab")
print("input:", tuple(x.shape))
print("output:", tuple(y.shape))
print("weight:", tuple(conv.weight.shape))
print("bias:", tuple(conv.bias.shape))
```

Expected output:

```text
channel_lab
input: (2, 3, 32, 32)
output: (2, 8, 32, 32)
weight: (8, 3, 3, 3)
bias: (8,)
```

Interpretation:

- `2`: two images in the batch;
- `3`: RGB input channels;
- `8`: eight learned output feature maps;
- `(8, 3, 3, 3)`: eight kernels, each looking across three input channels.

## Receptive Field: How CNNs See More Over Depth

![CNN receptive field grows layer by layer feature combination diagram](/img/course/ch06-cnn-receptive-field-growth-map-en.webp)

One `3 x 3` convolution sees a small local region. If you stack layers, later features indirectly depend on larger regions of the original image.

Intuition:

| Layer depth | What it often learns |
|---|---|
| shallow | edges, color changes, textures |
| middle | corners, simple shapes, parts |
| deep | larger object parts and semantic patterns |

This hierarchy is why CNNs work well for images: small local clues can be composed into larger visual ideas.

## Basic `Conv2d` Checklist

```python
import torch
from torch import nn

x = torch.randn(1, 1, 8, 8)
conv = nn.Conv2d(
    in_channels=1,
    out_channels=4,
    kernel_size=3,
    stride=1,
    padding=1,
)
y = conv(x)

print("conv2d_lab")
print("input:", tuple(x.shape))
print("output:", tuple(y.shape))
print("weight:", tuple(conv.weight.shape))
print("bias:", tuple(conv.bias.shape))
```

Expected output:

```text
conv2d_lab
input: (1, 1, 8, 8)
output: (1, 4, 8, 8)
weight: (4, 1, 3, 3)
bias: (4,)
```

When you read any `Conv2d`, ask:

1. What is the input shape `[N, C, H, W]`?
2. Does `in_channels` equal the input `C`?
3. How many feature maps does `out_channels` create?
4. How do `kernel_size`, `stride`, and `padding` change `H` and `W`?

## Evidence to Keep

For every convolution lab, save one shape equation:

```text
input_shape: [N, C_in, H, W]
kernel: [C_out, C_in, kH, kW]
output_shape: [N, C_out, H_out, W_out]
meaning: C_out feature maps scan local regions
```

If this record is clear, convolution becomes a shape-and-pattern operation rather than a mysterious image layer.

## Common Mistakes

| Mistake | Why it hurts | Fix |
|---|---|---|
| using image shape `[H, W, C]` in PyTorch | PyTorch expects `[N, C, H, W]` | use `permute` when converting from image libraries |
| wrong `in_channels` | `Conv2d` cannot match the input | print `x.shape` before the layer |
| forgetting padding | feature maps shrink unexpectedly | calculate output size or print shapes |
| treating convolution as magic | hard to debug features | remember patch * kernel -> sum |
| flattening too early | spatial structure is lost | use conv blocks before classifier head |

## Exercises

1. Change the hand-written `2 x 2` kernel and observe how the output changes.
2. Manually compute `out[1, 0]` in Lab 1 and compare with the printed output.
3. Change `stride=1` in the size lab. What output shape do you get?
4. Change `out_channels=16` in the channel lab. Which shapes change?
5. Convert an image-like tensor from `[N, H, W, C]` to `[N, C, H, W]` with `permute`.

<details>
<summary>Reference implementation and walkthrough</summary>

1. Changing the kernel changes which local pattern is emphasized. Edge-like kernels, averaging kernels, and sharpening kernels produce visibly different output maps.
2. Manual computation should multiply the selected `2 x 2` patch element by element with the kernel and sum the results. If it differs, recheck row and column position.
3. Reducing stride from `2` to `1` makes the kernel move one pixel at a time, so the output becomes spatially larger.
4. Changing `out_channels` changes the number of produced feature maps. The batch size and spatial dimensions follow the input, kernel, stride, and padding settings.
5. Use `x = x.permute(0, 3, 1, 2)` for `[N, H, W, C] -> [N, C, H, W]`. PyTorch convolution layers expect channels before height and width.

</details>

## Key Takeaways

- Convolution preserves local spatial structure better than early flattening.
- A kernel is a small pattern detector shared across positions.
- `stride` and `padding` control how the kernel moves and how output size changes.
- Multi-channel convolution combines information across input channels.
- Stacked convolution layers grow receptive field and build visual hierarchy.
