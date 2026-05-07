---
title: "6.3.3 Basic CNN Architecture"
sidebar_position: 2
description: "From convolution blocks, activation, and pooling to the classification head, systematically understand how a CNN turns an image into a category decision layer by layer."
keywords: [CNN, convolution block, pooling, feature map, classification head, fully connected layer, Global Average Pooling]
---

# 6.3.3 Basic CNN Architecture

![CNN feature map pipeline](/img/course/cnn-feature-map-pipeline-en.png)

:::tip Where this section fits
In the previous section, we learned that convolution kernels “slide across the image to find local patterns.”
In this section, we will assemble those scattered pieces and answer a more complete question:

> **How does an entire CNN actually work?**

You will see that a CNN is not just made of convolution layers. It is a chain of modules that “extract features -> compress -> make decisions.”
:::

## Learning Objectives

- Understand which modules a typical CNN is made of
- Master the main path of `convolution -> activation -> pooling -> classification head`
- Understand why the number of channels keeps increasing and why spatial size keeps decreasing
- Read the forward pass of a minimal CNN in PyTorch
- Distinguish between the two classification head ideas: `Flatten` and `Global Average Pooling`

---

## First, get the whole map clear

### What does a typical CNN look like?

The classic CNN can be roughly drawn like this:

```mermaid
flowchart LR
    A["Input image"] --> B["Convolution layer"]
    B --> C["Activation function ReLU"]
    C --> D["Pooling layer"]
    D --> E["Convolution layer"]
    E --> F["Activation function ReLU"]
    F --> G["Pooling layer"]
    G --> H["Flatten / Global Pooling"]
    H --> I["Fully connected layer"]
    I --> J["Class output"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style B fill:#fff3e0,stroke:#e65100,color:#333
    style C fill:#fff3e0,stroke:#e65100,color:#333
    style D fill:#f3e5f5,stroke:#6a1b9a,color:#333
    style E fill:#fff3e0,stroke:#e65100,color:#333
    style F fill:#fff3e0,stroke:#e65100,color:#333
    style G fill:#f3e5f5,stroke:#6a1b9a,color:#333
    style H fill:#fffde7,stroke:#f9a825,color:#333
    style I fill:#e8f5e9,stroke:#2e7d32,color:#333
    style J fill:#ffebee,stroke:#c62828,color:#333
```

If we put that into plain language:

1. First use convolution to find local features
2. Then use an activation function to add nonlinearity
3. Then use pooling to shrink the size and keep key information
4. Repeat several rounds to get more and more abstract features
5. Finally, hand those features to the classification head for decision-making

### A memory aid

You can think of a CNN as a “multi-stage security screening system”:

- The first layer looks at edges and textures
- The second layer looks at local shapes
- The third layer looks at combinations of parts
- The last few layers decide whether it looks like a cat or a dog

In other words, a CNN does not directly understand “cat” at the beginning.
It first understands things like “fur edge, ear contour, eye region, body shape,” and then gradually combines them.

---

## Why does the number of channels in a CNN keep increasing?

### Channel count can be understood as “number of feature types”

In the input layer:

- Grayscale images usually have 1 channel
- RGB images usually have 3 channels

But once the image enters a CNN, the meaning of channels changes.
They are no longer just “color channels”; instead, they represent:

> **Different feature maps extracted by different convolution kernels.**

For example:

- The first kernel may be good at finding vertical edges
- The second kernel may be good at finding horizontal edges
- The third kernel may be good at finding diagonal edges

So when you see:

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
```

it means:

- There are 3 input channels
- There are 16 output feature maps

### Why do we often see 32, 64, 128 later?

Because deeper layers are expected to learn more and more abstract patterns.
Early layers only need to detect basic textures, while later layers need to combine them into more complex structures, so the channel count is usually increased gradually.

---

## Why does the spatial size keep shrinking?

### Because the model is moving from “details” to “abstraction”

Early layers focus more on local details:

- Where are the edges?
- Where are the textures?

Later layers focus more on overall abstraction:

- Are there ears?
- Are there wheels?
- Does it look like a cat?

So a common trend is:

- Height and width get smaller over time
- Channel count gets larger over time

You can think of it like this:

> Spatial resolution goes down, but semantic richness goes up.

![CNN channel count vs spatial size trade-off](/img/course/ch06-cnn-channel-spatial-tradeoff-map-en.png)

:::tip Reading guide
This figure helps you understand a common shape change in CNNs: as you go deeper, height and width usually get smaller because the model no longer needs to preserve every pixel detail; channels usually get larger because the model needs to store more and more abstract kinds of features.
:::

### What does a pooling layer do?

The most common pooling operation is `MaxPool`, which takes the maximum value within a small window.

For example:

```python
import numpy as np

feature_map = np.array([
    [1, 3, 2, 0],
    [4, 6, 1, 2],
    [0, 1, 5, 3],
    [2, 4, 1, 7]
], dtype=np.float32)

pooled = np.array([
    [feature_map[0:2, 0:2].max(), feature_map[0:2, 2:4].max()],
    [feature_map[2:4, 0:2].max(), feature_map[2:4, 2:4].max()]
])

print("feature_map =\n", feature_map)
print("pooled =\n", pooled)
```

The output will compress `4x4` into `2x2`.

### Doesn’t MaxPool “lose information”?

Yes, it does discard some details.
But it keeps the most prominent response in each local region, which is often very helpful for classification tasks.

You can think of it like this:

> Instead of remembering every pixel, it is better to first keep whether the strongest feature in this region has appeared.

---

## A convolution block is the basic building unit of a CNN

### What is a convolution block?

In modern deep learning, people usually do not look at a convolution layer alone. Instead, they often treat the following combination as one basic block:

```text
convolution -> activation -> (optional) pooling
```

or:

```text
convolution -> BN -> ReLU
```

### A minimal convolution block example

```python
import torch
from torch import nn

block = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2)
)

x = torch.randn(2, 3, 32, 32)
y = block(x)

print("input shape :", x.shape)
print("output shape:", y.shape)
```

This block does three things:

1. Maps a 3-channel image to 8-channel features
2. Adds nonlinearity through ReLU
3. Compresses `32x32` to `16x16` through pooling

---

## Forward pass of a complete small CNN

### Runnable example

```python
import torch
from torch import nn

class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # [B, 1, 28, 28] -> [B, 8, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(2),                             # -> [B, 8, 14, 14]

            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # -> [B, 16, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(2)                              # -> [B, 16, 7, 7]
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),                                # -> [B, 16*7*7]
            nn.Linear(16 * 7 * 7, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = TinyCNN(num_classes=10)
x = torch.randn(4, 1, 28, 28)
y = model(x)

print("output shape:", y.shape)
```

### Why is the final output `[4, 10]` here?

Because:

- There are 4 images in the batch
- Each image should output 10 class scores

In other words, this model is already a complete skeleton for an image classifier.

---

## Truly understanding this network structure

### The earlier part: `features`

This part is responsible for:

- Extracting local patterns
- Compressing spatial size
- Gradually obtaining more abstract features

### The later part: `classifier`

This part is responsible for:

- Turning high-dimensional feature maps into class scores

Remember this in one sentence:

> The front part “looks at the image and refines features,” while the back part “makes decisions based on those features.”

---

## What is the difference between Flatten and Global Average Pooling?

### Flatten: directly unroll the tensor

Using the example above:

- `16 x 7 x 7`
- Flattened into `784`

Advantages:

- Simple and direct

Disadvantages:

- The number of parameters may become large

### Global Average Pooling: keep only one average value per channel

For example:

- `16 x 7 x 7`
- Becomes `16`

This greatly reduces the number of parameters.

### A runnable mini example

```python
import torch

x = torch.randn(2, 16, 7, 7)

flat = torch.flatten(x, start_dim=1)
gap = x.mean(dim=(2, 3))

print("flatten shape:", flat.shape)
print("gap shape    :", gap.shape)
```

So in modern CNNs, we often prefer:

- A convolution backbone
- Global average pooling
- One final linear layer

---

## Why can CNNs gradually understand images from low-level to high-level features?

You can think of it like this:

- The 1st layer sees edges
- The 2nd layer sees corners and local textures
- The 3rd layer sees combinations of parts
- Deeper layers see object semantics

This is like when you look at a cat image:

1. First you see lines and color changes
2. Then you see ears, eyes, and whisker regions
3. Finally, you decide: this is a cat

The hierarchical structure of a CNN is essentially simulating this recognition process from local to global.

---

## How do you print intermediate shapes in PyTorch?

This is a very practical debugging skill.

```python
import torch
from torch import nn

class DebugCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

    def forward(self, x):
        print("input :", x.shape)
        x = self.conv1(x)
        print("conv1 :", x.shape)
        x = torch.relu(x)
        x = self.pool(x)
        print("pool1 :", x.shape)
        x = self.conv2(x)
        print("conv2 :", x.shape)
        return x

model = DebugCNN()
x = torch.randn(1, 1, 28, 28)
_ = model(x)
```

Most CNN errors are not really because “convolution does not work,” but because:

- The shapes were not calculated correctly
- The flattened dimension was written wrongly
- The input dimension for the linear layer does not match

---

## Common beginner mistakes

### Knowing only that “convolution is important,” but not that a CNN is actually a combination of many layers

The real power of CNNs comes from the structure, not from one convolution layer by itself.

### Not tracking shapes

This is one of the most common bug sources in image models.

### Thinking pooling just “shrinks things a bit”

Pooling is actually balancing feature retention and spatial compression.

---

## Summary

The most important thing in this section is not memorizing “CNN = Convolutional Neural Network,” but grasping its main workflow:

> **A CNN turns the original image into increasingly abstract features layer by layer, and then makes a classification decision based on those features.**

That is why a complete CNN usually looks like this:

- Convolution blocks stacked together
- Spatial size gradually decreasing
- Channel count gradually increasing
- A classification head at the end

Once you understand this, you will no longer just be memorizing diagrams when you look at LeNet, VGG, or ResNet.

---

## Exercises

1. Change the output channels of the second convolution in `TinyCNN` from 16 to 32 and see how the shape changes.
2. Change the classification head to the form `Global Average Pooling + Linear`.
3. Work out by hand why a `28x28` input becomes `7x7` after two `MaxPool2d(2)` operations.
4. Think about this: why do CNNs often use convolution blocks in the earlier part and a classification head only at the end?
