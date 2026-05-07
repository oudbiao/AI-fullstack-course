---
title: "6.1.8 Optional Background: Deep Learning Breakthroughs"
sidebar_position: 1
description: "Understand the major breakthroughs in deep learning through their historical progression: perceptron, the XOR setback, backpropagation, gradient vanishing, LSTM, RBM/DBN, AlexNet, ResNet, Attention, and Transformer, and what problems each one solved."
keywords: [deep learning history, three waves of neural networks, perceptron, XOR, backpropagation, LSTM, AlexNet, ResNet, Transformer]
---

# 6.1.8 Optional Background: Deep Learning Breakthroughs

![Deep Learning History Breakthrough Map](/img/course/ch06-dl-history-breakthrough-map-en.png)

:::tip Section Overview
This section helps you place the model names in Chapter 6 back into their historical context.

You do not need to memorize every year, but you should understand what question each breakthrough was answering:

> **Why was the previous method not enough? What exactly did the new method add?**
:::

## First, grasp the three major shifts in deep learning history

Deep learning did not develop smoothly all the way through. It is more like several cycles of “hope rises, bottlenecks appear, conditions mature, and then it takes off again.”

| Stage | Expectations at the time | Main bottleneck | Later breakthrough |
|---|---|---|---|
| First wave of neural networks | The perceptron showed that machines could learn rules from data | Single-layer models had limited expressiveness; XOR could not be solved | Multi-layer networks and backpropagation |
| Second wave of neural networks | Backpropagation made multi-layer networks trainable | Gradient vanishing, little data, weak compute | LSTM, initialization, and pretraining ideas |
| Deep learning revival | Data, GPUs, and network architectures all matured | Deep networks were hard to train; sequence modeling was a bottleneck | AlexNet, ResNet, Attention, Transformer |

This historical line closely matches the learning order in Chapter 6:

| Historical problem | Related learning in this chapter |
|---|---|
| What can a single neuron do | 1.4 From Neurons to MLP |
| How to train multi-layer networks | 1.5 Forward Propagation and Backpropagation |
| Why training is unstable | 1.6 Optimizers, 1.7 Regularization, 1.8 Initialization |
| Why CNNs are suitable for images | 3.2 How Convolution Works, 3.4 Classic CNNs |
| Why images fit CNNs | Chapter 3 CNNs |
| Why sequences need memory mechanisms | Chapter 4 RNN / LSTM |
| How to handle long-range dependencies and parallel training | Chapter 5 Attention / Transformer |

## 1943–1958: From artificial neurons to the perceptron

In 1943, McCulloch and Pitts proposed an early abstraction of the artificial neuron: a neuron could receive inputs and produce an output after simple computation. This idea was very rough, but it translated “brain-like computation” into a computable model for the first time.

In 1958, Rosenblatt proposed the perceptron. What made the perceptron exciting was this:

> **The machine was not just executing hand-written rules; it could adjust parameters from samples.**

For beginners, you can think of the perceptron as the smallest neural network model:

```text
input features -> weighted sum -> activation decision -> output class
```

It is very similar to the linear models in Chapter 5, but it opened the door to neural networks: if one neuron can learn some patterns, could many neurons and multi-layer structures learn more complex ones?

Recommended related learning:

| Related section | What you should understand |
|---|---|
| 1.4 From Neurons to MLP | Neurons, weights, bias, activation functions |
| Chapter 5 Logistic Regression | How linear scoring connects to class probabilities |

## 1969: The XOR problem cooled down the first neural network wave

The limitations of the perceptron quickly became clear. Minsky and Papert pointed out that a single-layer perceptron cannot solve nonlinear separable problems like XOR.

The key point about XOR is not that it is very complex, but that it reminds us:

> **If the data cannot be separated by a single straight line, a single-layer model cannot learn it.**

This had a huge historical impact because it showed that early neural networks were far too limited in expressiveness. As a result, the first wave of neural network enthusiasm cooled off noticeably.

But from today’s learning perspective, XOR is actually a particularly good teaching example:

| Problem | Why it matters |
|---|---|
| A single-layer model cannot separate XOR | Shows the limitation of linear decision boundaries |
| Adding a hidden layer can solve it | Shows that multi-layer networks can combine nonlinearities |
| Nonlinear activation is needed | Shows that activation functions are not decoration; they are the source of expressiveness |

Recommended related learning:

| Related section | What you should understand |
|---|---|
| 1.4 From Neurons to MLP | Why multi-layer structures and activation functions matter |
| Chapter 5 Task Types and Decision Boundaries | Why linear models are not universal |

## 1980: Neocognitron quietly planted the key ideas behind CNNs

Long before AlexNet, Fukushima’s Neocognitron, proposed in 1980, already contained the spirit of many core ideas in modern CNNs:

- Local receptive fields: not every pixel connects to every position; first look at local regions
- Hierarchical features: first detect simple edges, then gradually combine them into more complex shapes
- Intuition of spatial invariance: the same feature appearing in different positions should still be recognizable

For beginners, you can understand it this way:

> **An image is not a collection of isolated pixels, but something built layer by layer from local textures, edges, and shapes.**

The Neocognitron did not directly become today’s mainstream engineering framework, but it introduced the core CNN intuition very early. Later, LeNet, AlexNet, and ResNet continued pushing this path toward something trainable, scalable, and practical.

Recommended related learning:

| Related section | What you should understand |
|---|---|
| 3.2 How Convolution Works | Local receptive fields, convolution kernels, feature maps |
| 3.4 Classic CNN Architectures | How LeNet, AlexNet, and ResNet inherit and amplify CNN ideas |

## 1986: Backpropagation finally made multi-layer networks trainable

If a network has only one layer, adjusting parameters is still relatively intuitive. But the problem with multi-layer networks is that the influence of early-layer parameters on the final loss is indirect and complicated.

Backpropagation solves this core problem:

> **It sends the final error back through the computation graph layer by layer, telling each parameter which direction to change.**

It relies on the chain rule from Chapter 4 and also powers the training loop in Chapter 6.

You can think of backpropagation as a project postmortem:

| Training action | Analogy |
|---|---|
| Forward propagation | The project makes one prediction first |
| Compute loss | Check how far the result is from the target |
| Backpropagation | Trace how much each step contributed to the error |
| Optimizer update | Adjust parameters based on responsibility assignment |

Recommended related learning:

| Related section | What you should understand |
|---|---|
| 1.5 Forward Propagation and Backpropagation | Computation graph, loss, gradient |
| PyTorch automatic differentiation | What `loss.backward()` means |
| Chapter 4 Chain Rule | Why backpropagation works mathematically |

## 1989–1997: Expressiveness, gradient vanishing, and LSTM

In 1989, Cybenko’s universal approximation theorem theoretically showed that feedforward networks with nonlinearities have very strong function approximation ability. It sent an important signal for the neural network path: if the structure and training are appropriate, neural networks can indeed represent complex functions.

But being theoretically expressive does not mean being easy to train in practice. In 1994, Bengio and others systematically pointed out the gradient vanishing problem in long-sequence training. Ordinary RNNs, when processing long text or long time series, easily “forget” early information, and gradients also struggle to flow stably back to very early time steps.

In 1997, LSTM used gating mechanisms to alleviate this problem. You can think of LSTM as giving the RNN a more reliable memory notebook:

| Model | Problem it solves |
|---|---|
| Ordinary RNN | Can handle sequences, but easily forgets distant information |
| LSTM | Uses gates to control what to remember, what to forget, and what to output |
| GRU | Achieves similar capability with a simpler gating structure |

Recommended related learning:

| Related section | What you should understand |
|---|---|
| 1.8 Weight Initialization | Why stable signals and gradients matter |
| 4.2 RNN Basics | Sequence modeling and hidden states |
| 4.3 LSTM and GRU | How gating alleviates long-term dependency issues |

## 2006: RBM / DBN brought deep networks back into focus

Around 2006, Hinton’s Deep Belief Nets and RBM pretraining work brought deep networks back into the spotlight. At that time, directly training deep networks was not easy, and pretraining offered a strategy of “learn representations layer by layer first, then fine-tune for the task.”

You may not need to hand-code RBMs in a project today, but their historical significance is that:

> **They made people believe again that multi-layer representation learning could be feasible.**

This was an important prelude to the “deep learning revival.” Later, data scale, GPUs, initialization, regularization, optimizers, and network architectures all matured together, and deep learning truly took off.

Recommended related learning:

| Related section | What you should understand |
|---|---|
| Training techniques | Why deep networks need initialization, regularization, and diagnostics |
| Generative model electives | RBM, VAE, and GAN are different ways of learning data distributions |
| Chapter 7 Pretraining | The idea of “learn general representations first, then transfer to tasks” |

## 2012–2015: AlexNet, ImageNet, and ResNet made deep learning truly break through in vision

In 2012, AlexNet achieved a breakthrough result in the ImageNet image classification competition. This breakthrough was not just about the model architecture itself, but about several conditions maturing at the same time:

- Large-scale labeled dataset ImageNet
- GPU-accelerated training
- Deeper CNNs
- Training techniques such as ReLU, Dropout, and data augmentation

AlexNet made many people realize that deep learning could clearly outperform traditional approaches in vision tasks.

In 2015, ResNet solved the problem of training very deep networks by introducing residual connections. The intuition behind residual connections is: do not force every layer to learn a complete transformation from scratch; instead, let it learn “how much to change relative to the input.”

Recommended related learning:

| Related section | What you should understand |
|---|---|
| 3.2 How Convolution Works | Why CNNs are suitable for images |
| 3.4 Classic CNN Architectures | The evolution of LeNet, AlexNet, VGG, and ResNet |
| 3.5 Transfer Learning | Why pretrained vision models can transfer to new tasks |
| Chapter 10 Computer Vision | How image classification, detection, and segmentation continue to develop |

## 2017: Attention and Transformer rewrote the main line of sequence modeling

RNNs and LSTMs process sequences in order, which naturally creates a problem: they are hard to parallelize, and the path for long-range information is also long. The breakthrough of the Transformer is:

> **It uses self-attention to allow any position in the sequence to directly relate to any other position.**

The paper *Attention Is All You Need* did not just propose a new model; it changed the foundation of NLP, large models, and multimodal systems that came after it.

Transformer solved several key problems:

| Old problem | What Transformer changed |
|---|---|
| RNNs compute in order and are not easy to parallelize | Self-Attention can process tokens in parallel |
| The path for long-range dependencies is too long | Attention lets distant tokens directly attend to each other |
| Task structures are scattered across different models | Encoder, Decoder, and the pretraining paradigm unify many tasks |

Recommended related learning:

| Related section | What you should understand |
|---|---|
| 5.2 Attention Mechanism | Q/K/V and self-attention |
| 5.3 Transformer Architecture | Blocks, residuals, LayerNorm, FFN |
| Chapter 7 Principles of Large Models | How Transformer became the foundation of LLMs |
| Chapters 8–9 RAG / Agent | How large models connect with knowledge and tools |

## Map the breakthroughs in deep learning to the learning path of Chapter 6

| Historical breakthrough | Problem it solved | Corresponding chapter in this course |
|---|---|---|
| McCulloch-Pitts / Perceptron | Neurons are computable, parameters are learnable | 1.4 From Neurons to MLP |
| XOR limitation | Single-layer linear models lack expressiveness | 1.4 MLP, activation functions |
| Neocognitron | Local receptive fields and hierarchical visual features | 3.2 Convolution Operations, 3.4 Classic CNNs |
| Backpropagation | How multi-layer networks assign errors and update parameters | 1.5 Forward Propagation and Backpropagation, PyTorch autograd |
| Cybenko universal approximation | Multi-layer nonlinear networks have strong expressiveness | 1.4 MLP background |
| Gradient vanishing | Deep and long-sequence training is unstable | 1.8 Initialization, 4.3 LSTM |
| LSTM / GRU | Long-sequence memory and gated control | Chapter 4 RNNs and Sequence Models |
| RBM / DBN | Historical prelude to trainable deep networks | Generative models, pretraining background |
| AlexNet / ImageNet | Data + GPU + CNNs broke through vision tasks | Chapter 3 CNNs, Chapter 10 Vision |
| ResNet | Training very deep CNNs is difficult | 3.4 Classic CNN Architectures |
| Attention / Transformer | Long dependencies, parallel training, and unified sequence modeling | Chapter 5 Transformer, Chapter 7 LLMs |

## The intuition you should have after finishing this section

The history of deep learning is not just a pile of model names, but a continuous chain of problems:

| Old problem | New breakthrough | Capability you should practice now |
|---|---|---|
| Rules are impossible to write out completely | Perceptron and neurons | Understand parameter learning |
| Single-layer models are too weak | Multi-layer networks and activation functions | Understand expressiveness |
| Multi-layer networks are hard to train | Backpropagation | Understand the training loop |
| Long sequences are hard to remember | LSTM / GRU | Understand gated memory |
| Image tasks are hard | CNN / AlexNet / ResNet | Understand local features and deep structures |
| Sequences are hard to parallelize; long dependencies are hard | Attention / Transformer | Understand the foundation of large models |

If you can answer every model name with “what problem did it solve from the previous generation?”, then Chapter 6 will no longer feel like a list of architectures. It will become a clear path of technical evolution.
