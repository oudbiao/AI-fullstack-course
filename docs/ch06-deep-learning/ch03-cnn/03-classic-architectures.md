---
title: "3.4 Classic CNN Architectures"
sidebar_position: 3
description: "From LeNet and AlexNet to VGG and ResNet, understand why classic CNN architectures evolved generation by generation, and what problem each generation actually solved."
keywords: [LeNet, AlexNet, VGG, ResNet, CNN, residual connection, classic architecture]
---

# Classic CNN Architectures

![Classic CNN architecture evolution](/img/course/imagenet-cnn-evolution-en.png)

:::tip Section overview
Learning classic architectures is not about memorizing model names. It is about seeing a very important evolution line clearly:

> **As image tasks become more complex, how did CNNs gradually become stronger?**

Once you understand this line, later when you encounter more modern vision models, you will not just see a pile of terms.
:::

## Learning objectives

- Understand what problems LeNet, AlexNet, VGG, and ResNet each solved
- Read the evolution logic of classic CNNs instead of only memorizing diagrams
- Understand why “stacking small convolution kernels” and “residual connections” matter
- Write a minimal residual block and truly understand the core idea of ResNet
- Judge the pros and cons of different architectures from an engineering perspective

---

## 1. Why learn “classic architectures”?

### 1.1 Classic models are not outdated knowledge, but the evolution history of visual modeling

Many beginners learn CNN architectures like this:

- LeNet: memorize one name
- AlexNet: memorize another name
- VGG: memorize yet another name
- ResNet: seems important

This kind of learning is easy to become fragmented.

A better way is to see them as an evolution chain:

1. LeNet: proved that convolutional networks can do image recognition
2. AlexNet: truly scaled up deep CNNs and achieved strong results on large data
3. VGG: turned “stacking many small convolution kernels” into a standard approach
4. ResNet: solved the problem of training very deep networks

So the real goal of learning classic architectures is not to “know the names,” but to know:

> What key weakness did each generation make up for?

### 1.1.1 These names actually feel like a “deep visual evolution history”

If you only treat them as model names, it will feel boring very quickly.
But if you see them as an evolution history, it becomes much easier to feel their significance:

- `LeNet` is like proving that “the convolution path is workable”
- `AlexNet` is like telling the world “this path is not only feasible, it can suddenly perform extremely well”
- `VGG` is like organizing engineering experience into a more stable recipe
- `ResNet` is like solving the real bottleneck of “once the network gets deeper, training becomes hard”

So this lesson is not just about models.
It is about:

> **Why the vision field has evolved into today’s structures step by step.**

### 1.2 A very easy-to-remember analogy

You can understand classic CNN architectures as “building a house”:

- LeNet: first prove the house can be built
- AlexNet: the house gets taller, and it starts to be truly commercialized
- VGG: a more unified and reproducible construction standard appears
- ResNet: solve the problem that very tall buildings are easy to collapse

That is their intuitive difference.

---

## 2. LeNet: the early prototype of convolutional networks

### 2.1 LeNet’s historical position

LeNet was originally used mainly for handwritten digit recognition.
Its importance is not that it is still very strong today, but that it established the main line very early:

> **Convolution layers extract features, pooling layers compress them, and fully connected layers do classification.**

### 2.2 The typical LeNet structure

Roughly speaking, LeNet is:

```text
Input -> Conv -> Pool -> Conv -> Pool -> FC -> Output
```

It already had the core skeleton of a CNN.

### 2.3 What did LeNet teach us?

LeNet truly taught later researchers:

- Images should not be flattened from the start
- Local feature extraction is feasible
- Hierarchical feature learning is effective

Today this seems very natural, but at the time it was a crucial breakthrough.

### 2.4 Why is LeNet, despite being old, still worth discussing?

Because it was like a real “prototype” moment:

- It did not solve every problem
- But it was the first to lay out the skeleton of the entire later CNN main line

That is why in many classic architecture lessons,
LeNet’s meaning is often not “how strong it is today,”
but rather:

> **It finally showed later people how this building should probably be constructed.**

---

## 3. AlexNet: the real start of deep CNN breakthroughs

### 3.1 Why is AlexNet a milestone?

AlexNet is best known for truly demonstrating the power of deep CNNs on ImageNet.

Its historical significance can be summarized as:

- Deeper model
- Larger data
- GPU truly made a difference
- Techniques like ReLU and Dropout began to be widely adopted

### 3.1.1 Why do many people call AlexNet the “starting gun” of the deep learning revival?

Because before it, many people did not really believe that:

- deeper networks
- larger datasets
- stronger computing power

would together lead to such a dramatic performance jump.

What was most shocking about AlexNet was that it did not feel like “a little improvement step by step.”
Instead, it was more like:

- instantly pushing the whole route from “one research direction” to “the mainstream direction”

So its place in history is not only winning ImageNet,
but also making many people believe for the first time:

> **Deep learning is not just exciting in concept; it can already deliver decisive results.**

### 3.2 What problem did AlexNet solve?

Compared with earlier small models, it proved:

> **As long as data, compute, and training techniques keep up, deep convolutional networks can significantly improve image recognition.**

### 3.3 What did AlexNet inspire?

AlexNet did not just “get deeper.” It told everyone that:

- Deep networks are worth doing
- GPU training is the future direction
- Activation functions and regularization techniques are critical

It was more like the real starting gun for the deep vision era.

---

## 4. VGG: Why is “stacking small convolution kernels” so important?

### 4.1 One core idea of VGG

The easiest thing to remember about VGG is:

> **Use many stacked `3x3` small convolution layers instead of directly using large convolution kernels.**

### 4.2 Why not use large kernels, and instead use many small ones?

Because stacking several small convolution kernels has several advantages:

1. The receptive field can still become larger
2. The number of parameters is more controllable
3. More nonlinearities can be inserted in between

An intuitive example:

- One `7x7` convolution kernel: sees a large region in one step
- Three consecutive `3x3` convolutions: can also see a large range in the end, but each step can add nonlinearity

This is usually more flexible.

### 4.3 A rough parameter-count intuition

Assume the input and output channel numbers are the same. Roughly compare:

- One `7x7` convolution: parameters are proportional to `49`
- One `3x3` convolution: parameters are proportional to `9`

Although stacking multiple layers does not always mean fewer parameters, “small kernels + many layers” often leads to more refined representations.

### 4.4 Runnable example: a VGG-style small block

```python
import torch
from torch import nn

vgg_block = nn.Sequential(
    nn.Conv2d(3, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2)
)

x = torch.randn(2, 3, 32, 32)
y = vgg_block(x)

print("input shape :", x.shape)
print("output shape:", y.shape)
```

This block is very much like the classic VGG idea:

- First apply convolution twice in a row
- Then pool

---

## 5. ResNet: Why does training get harder when the network gets deeper?

### 5.1 A counterintuitive question

In theory, deeper networks have stronger expressive power, so the results should be better.
But in practice, people found that:

> Once the network becomes deep enough, training becomes harder, and the results are not necessarily better.

This is not as simple as “the model is too strong and overfits.” Instead, it is because:

- Gradient flow becomes harder
- Optimization becomes more difficult
- Very deep networks are not always easy to learn an “at least not worse” mapping

### 5.2 The core idea of ResNet

The key idea introduced by ResNet is the residual connection:

> Instead of directly learning `H(x)`, learn `F(x) = H(x) - x`

Then the output becomes:

> `y = F(x) + x`

### 5.2.1 What is truly impressive about ResNet?

What is impressive is not just the word “deeper,”
but that it answers a very practical question:

- If deep networks have such potential, why do they become harder to train as they go deeper?

The value of ResNet is that it does not avoid this problem.
Instead, it directly gives a very engineering-oriented answer:

- Provide a side path that makes it easier for deep networks to preserve the original information

That is why many people, after learning ResNet, clearly feel for the first time that:

> **Neural network architecture design is not just stacking layers. It is seriously dealing with optimization difficulties.**

### 5.3 Why does this help?

Because the model can now more easily learn:

- “This layer is useful, so learn a little new thing”
- “This layer does not need major changes, so keep the input information as much as possible”

In other words:

> Residual connections create a side path for deep networks so they do not completely lose the original information.

---

## 6. A minimal residual block that is really worth typing by hand

### 6.1 First look at the code

```python
import torch
from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + identity
        out = self.relu(out)
        return out

block = ResidualBlock(8)
x = torch.randn(2, 8, 16, 16)
y = block(x)

print("input shape :", x.shape)
print("output shape:", y.shape)
```

### 6.2 Which line is the most important?

The most important line is:

```python
out = out + identity
```

This is the residual connection itself.

It allows the network to learn new features without completely losing the original input information.

### 6.3 Why must the shapes match?

Because the addition is element-wise.
If the input and output shapes do not match, residual addition cannot be done directly.

That is also why in real ResNet models, people sometimes use:

- `1x1 conv`

to align dimensions.

---

## 7. A clear evolution line from classic architectures

### 7.1 Summary of the evolution logic

| Architecture | Core contribution | Problem it solved |
|---|---|---|
| LeNet | Early CNN skeleton | Proved that convolution can be used for image recognition |
| AlexNet | Deeper, larger, GPU training | Made deep CNNs take off for real |
| VGG | Stacking many small convolution kernels | Improved expressive power and made the structure more unified |
| ResNet | Residual connections | Solved the difficulty of training very deep networks |

### 7.2 What should you really remember?

Not “which model had how many layers and convolutions,” but:

> **Each generation of classic architecture is paving the way for deeper, more stable, and stronger visual representation learning.**

---

## 8. Do we still need to learn these classic architectures today?

### 8.1 Yes, but not to copy them exactly

In real projects today, you may not necessarily start from LeNet or AlexNet directly.
But these architectures are still important because they teach you:

- Why CNNs are shaped this way
- Why depth matters
- Why small convolution kernels became popular
- Why residual connections have almost become standard

### 8.2 Many modern models still inherit these ideas

Even though more modern models have appeared today, many of their core ideas can still be traced back to classic CNNs:

- Hierarchical features
- Channel expansion
- Deep stacking
- Residual paths

---

## 9. Common mistakes beginners make

### 9.1 Treating classic architectures as just “memorizing model names”

This is the easiest way to forget them, and it is almost impossible to transfer the knowledge.

### 9.2 Only looking at architecture diagrams without understanding why they were designed that way

Once you do not understand the design motivation, it becomes very hard to judge whether changes in a new architecture are meaningful.

### 9.3 Thinking the key point of ResNet is simply “deeper”

The key of ResNet is not depth itself, but:

> **Making depth trainable.**

---

## Summary

The most important thing in this section is not memorizing the names LeNet, AlexNet, VGG, and ResNet, but grasping this main line:

> **The evolution of classic CNN architectures is essentially about how to make image networks learn deeper, more stably, and more effectively.**

Once you understand this line, when you look at more modern vision models later, you will know which ideas they are continuing and what they are improving.

---

## Exercises

1. Summarize the most important point of LeNet, AlexNet, VGG, and ResNet in your own words.
2. Change `channels=8` in the minimal residual block to 16, and check whether the shapes still match.
3. Think about this: why are several consecutive `3x3` convolutions often more popular than using one large convolution kernel directly?
4. If you had to use just one sentence to distinguish VGG from ResNet, what would you say?
