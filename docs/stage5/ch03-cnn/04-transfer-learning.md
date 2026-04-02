---
title: "3.4 迁移学习 🔧"
sidebar_position: 4
description: "从为什么不从零训练开始，到冻结骨干、替换分类头和逐步微调，真正理解视觉里的迁移学习。"
keywords: [迁移学习, fine-tuning, feature extractor, freeze backbone, transfer learning, CNN]
---

# 迁移学习 🔧

:::tip 本节定位
如果你现在已经知道 CNN 会提特征、经典架构怎么演进，那接下来非常自然的一个工程问题就是：

> **我做自己的图像任务时，真的需要从零训练一整个 CNN 吗？**

大多数时候，答案是否定的。  
迁移学习就是在回答：怎样把别的任务上学到的视觉知识借过来。
:::

## 学习目标

- 理解为什么图像任务里迁移学习往往比从零训练更现实
- 分清“固定特征提取器”和“微调”两种常见方式
- 学会替换分类头、冻结骨干网络参数
- 看懂一个真正能运行的小型迁移学习示例
- 理解什么时候该只训头部，什么时候该解冻更多层

---

## 一、为什么迁移学习几乎成了视觉任务默认选项？

### 1.1 从零训练有多贵？

如果你要从零训练一个像样的视觉模型，通常至少会遇到这些问题：

- 数据不够多
- 标注成本高
- 训练时间长
- 容易过拟合

例如，你手里只有 2000 张图片，要分 5 类。  
这在真实项目里已经不算特别少，但对从零训练一个深 CNN 来说，仍然很可能不够稳。

### 1.2 预训练模型到底“预训练”了什么？

一个在大规模图像数据上训练过的模型，通常已经学会了很多通用视觉特征：

- 边缘
- 纹理
- 颜色组合
- 部件形状
- 常见物体模式

这些能力不是“猫狗任务专属”的，而是很多图像任务都用得到的基础视觉知识。

所以迁移学习的核心直觉是：

> **先复用已经学到的底层视觉能力，再把最后几层调成适合你自己的任务。**

### 1.3 一个帮助记忆的类比

迁移学习很像请一个已经学过通用绘画技巧的人来帮你画专业插图：

- 他不用从“怎么握笔”重新学起
- 你只需要让他适应你的具体风格和题材

这就是为什么视觉任务里迁移学习通常非常划算。

---

## 二、迁移学习最常见的两种方式

### 2.1 方式一：固定特征提取器（feature extractor）

做法：

- 预训练骨干网络参数不动
- 只训练最后的分类头

优点：

- 快
- 不容易把预训练能力训坏
- 适合数据特别少的场景

缺点：

- 适应新任务能力有限

### 2.2 方式二：微调（fine-tuning）

做法：

- 替换掉最后的分类头
- 除了头部，逐步解冻一部分甚至全部骨干网络

优点：

- 更能适应目标任务

缺点：

- 更容易过拟合
- 训练更慢
- 学习率更需要小心

### 2.3 一句话记忆

- 数据少：先倾向于固定特征提取器
- 数据多 / 任务差异大：再考虑逐步微调

---

## 三、一个“可直接运行”的迁移学习玩具示例

为了保证代码在没有外部模型下载的情况下也能跑通，我们自己模拟一个“已经预训练好的 backbone”。

### 3.1 先定义一个小型 backbone

```python
import torch
from torch import nn

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.features(x)
        return x.flatten(1)
```

这个 backbone 的输出是一个固定长度的特征向量。  
这就和很多真实预训练模型的“骨干网络输出特征”很像。

---

## 四、先做“固定特征提取器”版本

### 4.1 替换分类头并冻结 backbone

```python
import torch
from torch import nn

class TransferClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

model = TransferClassifier(num_classes=3)

# 冻结 backbone
for param in model.backbone.parameters():
    param.requires_grad = False

for name, param in model.named_parameters():
    print(name, "trainable =", param.requires_grad)
```

### 4.2 你应该从输出里看到什么？

你会发现：

- `backbone` 里的参数都不可训练
- 只有 `head` 的参数在训练

这就是最标准的“只训头部”迁移学习。

---

## 五、做一个真正能训练的小型图像分类任务

### 5.1 用合成数据模拟一个小任务

我们造 3 类简单图像：

- 竖线
- 横线
- 对角线

这样可以不用外部数据集，也能把训练闭环跑通。

```python
import numpy as np
import torch

def make_image(label, size=12):
    img = np.zeros((size, size), dtype=np.float32)

    if label == 0:  # 竖线
        img[:, size // 2] = 1.0
    elif label == 1:  # 横线
        img[size // 2, :] = 1.0
    else:  # 对角线
        for i in range(size):
            img[i, i] = 1.0

    img += np.random.randn(size, size).astype(np.float32) * 0.05
    return np.clip(img, 0.0, 1.0)

X, y = [], []
for label in range(3):
    for _ in range(80):
        X.append(make_image(label))
        y.append(label)

X = torch.tensor(np.array(X)).unsqueeze(1)
y = torch.tensor(np.array(y))

print(X.shape, y.shape)
```

---

## 六、完整训练：只训练头部

```python
import torch
from torch import nn

torch.manual_seed(42)

class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        return self.features(x).flatten(1)

class TransferClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.backbone = TinyBackbone()
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        feat = self.backbone(x)
        return self.head(feat)

model = TransferClassifier(num_classes=3)

# 冻结 backbone
for param in model.backbone.parameters():
    param.requires_grad = False

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.head.parameters(), lr=0.05)

for epoch in range(80):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")
```

### 6.2 这段代码真正想让你学会什么？

不是“冻结参数”这件语法本身，而是：

> 迁移学习的第一步，常常不是重训整个模型，而是先看已有特征能不能已经支撑你的任务。 

---

## 七、什么时候要进一步微调？

### 7.1 一个很常见的下一步

如果只训头部效果不够好，可以考虑：

- 解冻最后一个卷积块
- 用更小学习率继续训练

### 7.2 一个最小微调示例

```python
# 解冻最后一个卷积层
for param in model.backbone.features[3].parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.005
)

for epoch in range(40):
    pred = model(X)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        acc = (pred.argmax(dim=1) == y).float().mean().item()
        print(f"finetune epoch={epoch:3d}, loss={loss.item():.4f}, acc={acc:.3f}")
```

### 7.3 为什么微调通常要更小学习率？

因为 backbone 已经有一套“原来学到的特征”。  
如果学习率太大，很容易把这些已经不错的表示一下子冲坏。

所以常见经验是：

- 头部学习率大一点
- backbone 学习率小一点

---

## 八、真实项目里迁移学习通常怎么做？

### 8.1 最常见套路

1. 选一个预训练 backbone
2. 替换最后分类头
3. 先只训头部
4. 如果效果不够，再逐步解冻
5. 持续看验证集表现

### 8.2 为什么这套流程很流行？

因为它兼顾了：

- 训练速度
- 稳定性
- 最终效果

这比“一上来全训”通常更稳。

---

## 九、初学者最常踩的坑

### 9.1 以为迁移学习就是“复制一个大模型”

真正关键的是：

- 哪些层冻结
- 哪些层解冻
- 学习率怎么配

### 9.2 一上来就全量微调

这通常既慢又不稳，特别是在小数据任务上。

### 9.3 忘记检查哪些参数在训练

这是非常常见的错误。  
训练前最好打印一遍 `requires_grad` 状态。

---

## 小结

这一节最重要的不是背“迁移学习”四个字，而是建立一个稳定工程直觉：

> **先复用预训练模型已经学到的通用特征，再根据你的任务决定训头部、训局部还是训全部。**

这也是为什么在很多现实视觉项目里，迁移学习不是技巧，而几乎是默认起点。

---

## 练习

1. 把示例中的类别从 3 类扩展到 4 类，再设计一种新的图像模式。
2. 比较“只训头部”和“再解冻一层”时的训练曲线。
3. 打印模型里所有参数的 `requires_grad`，确认你真的知道哪些层在训练。
4. 想一想：如果你的目标任务和原预训练任务差别非常大，为什么可能需要解冻更多层？
