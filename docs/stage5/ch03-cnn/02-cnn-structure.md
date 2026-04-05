---
title: "3.3 CNN 基本结构"
sidebar_position: 2
description: "从卷积块、激活、池化到分类头，系统理解一个 CNN 是如何一层层把图像变成类别判断的。"
keywords: [CNN, 卷积块, 池化, 特征图, 分类头, 全连接层, Global Average Pooling]
---

# CNN 基本结构

:::tip 本节定位
上一节我们已经知道卷积核是在图像上“滑动找局部模式”。  
这一节要把这些零散部件组装起来，回答一个更完整的问题：

> **一整个 CNN 到底是怎么工作的？**

你会看到，CNN 不是只有卷积层，而是由一串“提特征 -> 压缩 -> 决策”的模块组成。
:::

## 学习目标

- 理解一个典型 CNN 由哪些模块组成
- 掌握 `卷积 -> 激活 -> 池化 -> 分类头` 这条主线
- 理解通道数为什么会越来越多、空间尺寸为什么会越来越小
- 看懂一个最小 CNN 在 PyTorch 中的前向传播
- 分清 `Flatten` 和 `Global Average Pooling` 两种分类头思路

---

## 一、先把整张地图看清楚

### 1.1 一个最典型的 CNN 长什么样？

最经典的 CNN 可以先粗略画成这样：

```mermaid
flowchart LR
    A["输入图像"] --> B["卷积层"]
    B --> C["激活函数 ReLU"]
    C --> D["池化层"]
    D --> E["卷积层"]
    E --> F["激活函数 ReLU"]
    F --> G["池化层"]
    G --> H["展平 / 全局池化"]
    H --> I["全连接层"]
    I --> J["类别输出"]

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

如果把它翻译成人话，就是：

1. 先用卷积找局部特征
2. 再用激活函数加入非线性
3. 再用池化压缩尺寸、保留关键信息
4. 重复几轮，得到越来越抽象的特征
5. 最后把这些特征交给分类头做决策

### 1.2 一个帮助记忆的类比

你可以把 CNN 理解成一个“多层安检系统”：

- 第一层看边缘、纹理
- 第二层看局部形状
- 第三层看部件组合
- 最后几层才判断“这到底像猫还是狗”

也就是说，CNN 不会一上来就直接理解“猫”，  
它是先理解“毛边、耳朵轮廓、眼睛区域、身体形状”，再一点点组合起来。

---

## 二、为什么 CNN 的通道数越来越多？

### 2.1 通道数可以理解成“特征种类数”

在输入层里：

- 灰度图通常是 1 个通道
- RGB 图通常是 3 个通道

但进入 CNN 以后，通道的意义变了。  
它不再只是“颜色通道”，而是：

> **不同卷积核提取出来的不同特征图。**

例如：

- 第 1 个核可能擅长找竖边
- 第 2 个核可能擅长找横边
- 第 3 个核可能擅长找斜角

所以当你看到：

```python
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3)
```

它的意思是：

- 输入有 3 个通道
- 输出 16 种特征图

### 2.2 为什么后面常常是 32、64、128？

因为越往后，模型希望学到更多更抽象的模式。  
前面层只需要发现基础纹理，后面层要组合出更复杂的结构，所以通常会逐渐增加通道数。

---

## 三、为什么空间尺寸会越来越小？

### 3.1 因为模型要从“细节”走向“概括”

前几层更关注局部细节：

- 哪里有边缘
- 哪里有纹理

后几层更关注整体概括：

- 有没有耳朵
- 有没有轮子
- 是不是像一只猫

所以一个常见趋势是：

- 高宽逐渐变小
- 通道逐渐变多

这可以理解成：

> 空间分辨率下降，但语义浓度上升。

### 3.2 池化层在做什么？

池化最常见的是 `MaxPool`，它会在一个小窗口里取最大值。

比如：

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

输出会把 `4x4` 压成 `2x2`。

### 3.3 MaxPool 为什么不是“丢信息”吗？

是的，它确实会丢掉一部分细节。  
但它保留了每个局部区域里最显著的响应，这对分类任务通常很有帮助。

你可以把它理解成：

> 与其记住每一个像素，不如先保留“这一块里最强的特征有没有出现”。 

---

## 四、卷积块（Conv Block）才是 CNN 的基本单元

### 4.1 什么是卷积块？

在现代深度学习里，人们通常不会孤立地看一层卷积，而更常把下面这一组合当成一个基本块：

```text
卷积 -> 激活 -> （可选）池化
```

或者：

```text
卷积 -> BN -> ReLU
```

### 4.2 一个最小卷积块示例

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

这个块做了三件事：

1. 把 3 通道图像映射成 8 通道特征
2. 通过 ReLU 引入非线性
3. 通过池化把 `32x32` 压成 `16x16`

---

## 五、一个完整小 CNN 的前向传播

### 5.1 可运行示例

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

### 5.2 为什么这里最后输出是 `[4, 10]`？

因为：

- batch 里有 4 张图
- 每张图要输出 10 个类别分数

也就是说，这个模型已经是一个完整的图像分类器骨架了。

---

## 六、真正理解这段网络结构

### 6.1 前半段 `features`

这部分负责：

- 提取局部模式
- 压缩空间尺寸
- 逐步得到更抽象的特征

### 6.2 后半段 `classifier`

这部分负责：

- 把高维特征图变成类别分数

一句话记住：

> 前面在“看图并做特征提炼”，后面在“根据特征做决策”。

---

## 七、Flatten 和 Global Average Pooling 有什么区别？

### 7.1 Flatten：直接摊平

像上面的例子：

- `16 x 7 x 7`
- 展平成 `784`

优点：

- 简单直接

缺点：

- 参数量可能变大

### 7.2 Global Average Pooling：每个通道只保留一个平均值

例如：

- `16 x 7 x 7`
- 变成 `16`

这样参数会少很多。

### 7.3 一个可运行小例子

```python
import torch

x = torch.randn(2, 16, 7, 7)

flat = torch.flatten(x, start_dim=1)
gap = x.mean(dim=(2, 3))

print("flatten shape:", flat.shape)
print("gap shape    :", gap.shape)
```

所以现代 CNN 很多时候会更偏向：

- 卷积主干
- 全局平均池化
- 最后一个线性层

---

## 八、为什么 CNN 能从低层到高层逐步理解图像？

可以这样理解：

- 第 1 层看到：边缘
- 第 2 层看到：角点、局部纹理
- 第 3 层看到：部件组合
- 更深层看到：物体语义

这就像你看一张猫图：

1. 先看到线条和颜色变化
2. 再看到耳朵、眼睛、胡须区域
3. 最后综合判断：这是一只猫

CNN 的层级结构，本质上就在模拟这种“从局部到整体”的识别过程。

---

## 九、PyTorch 中怎么打印中间 shape？

这是非常实用的调试技巧。

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

很多 CNN 报错，本质上都不是卷积不会，而是：

- shape 没算清楚
- 展平尺寸写错
- 线性层输入维度不匹配

---

## 十、初学者最常踩的坑

### 10.1 只知道“卷积很重要”，但不知道一个 CNN 其实是很多层组合

CNN 真正的力量来自结构，而不是一层卷积本身。

### 10.2 不会跟踪 shape

这是图像模型最常见的 bug 来源之一。

### 10.3 以为池化只是“随便压小一点”

池化其实是在做特征保留和空间压缩的平衡。

---

## 小结

这一节最重要的不是背“CNN = 卷积神经网络”，而是抓住它的工作主线：

> **CNN 会一层层把原始图像变成越来越抽象的特征，最后再基于这些特征做分类决策。**

这就是为什么一个完整 CNN 看起来总像：

- 卷积块堆叠
- 空间逐步缩小
- 通道逐步增加
- 最后接分类头

理解了这一点，后面看 LeNet、VGG、ResNet 就不会只是记结构图了。

---

## 练习

1. 把 `TinyCNN` 的第二层卷积输出通道从 16 改成 32，看看 shape 怎么变化。
2. 把分类头改成 `Global Average Pooling + Linear` 的形式。
3. 手工算一遍 `28x28` 输入经过两次 `MaxPool2d(2)` 后为什么会变成 `7x7`。
4. 想一想：为什么 CNN 前半段常常用卷积块，后半段才接分类头？
