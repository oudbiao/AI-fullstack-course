---
title: "3.6 CNN 实战：图像分类"
sidebar_position: 5
description: "从造数据、搭网络、训练、验证到预测，完整走通一个小型 CNN 图像分类项目。"
keywords: [image classification, CNN, PyTorch, train loop, validation, synthetic dataset]
---

# CNN 实战：图像分类

:::tip 本节定位
卷积、CNN 结构、经典架构、迁移学习都讲完以后，最重要的一件事就是：

> **把这些概念真正串成一个完整训练闭环。**

这一节不会追求“大模型效果”，而是追求一件更重要的事：

> 让你完整走通一次图像分类项目。
:::

## 学习目标

- 构造一个最小可训练的图像分类任务
- 用 CNN 完整跑通训练、验证和预测
- 理解图像分类项目里数据、模型、损失函数和指标怎样配合
- 学会从结果判断模型到底有没有学到东西

---

## 一、图像分类项目最小闭环是什么？

一个图像分类项目，最少要有这几个部分：

1. 数据
2. 类别标签
3. 模型
4. 损失函数
5. 训练循环
6. 验证 / 测试

很多初学者之所以学得发虚，就是因为只看到了“模型结构”，却没有把整个闭环串起来。

这节课的重点就是把这条链完整走一遍。

---

## 二、先准备一份能直接跑的数据

### 2.1 为什么继续用合成图像？

因为这样：

- 不依赖外部下载
- 类别规律非常清楚
- 最适合教学

### 2.2 造三类小图像

我们造 3 类：

- 竖线
- 横线
- 对角线

```python
import numpy as np
import matplotlib.pyplot as plt

def make_image(label, size=12):
    img = np.zeros((size, size), dtype=np.float32)

    if label == 0:
        img[:, size // 2] = 1.0
    elif label == 1:
        img[size // 2, :] = 1.0
    else:
        for i in range(size):
            img[i, i] = 1.0

    img += np.random.randn(size, size).astype(np.float32) * 0.05
    return np.clip(img, 0.0, 1.0)

fig, axes = plt.subplots(1, 3, figsize=(9, 3))
for label in range(3):
    axes[label].imshow(make_image(label), cmap="gray")
    axes[label].set_title(f"class {label}")
    axes[label].axis("off")
plt.tight_layout()
plt.show()
```

### 2.3 这个数据集虽然简单，但足够教会你什么？

它足够教会你：

- 图像张量该怎么组织
- 分类标签怎样对齐
- CNN 怎么学习局部模式

这比直接扔给你一个大数据集更适合入门。

---

## 三、把数据做成训练集和验证集

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

X, y = [], []
for label in range(3):
    for _ in range(100):
        X.append(make_image(label))
        y.append(label)

X = np.array(X, dtype=np.float32)
y = np.array(y)

# 打乱
indices = np.random.permutation(len(X))
X = X[indices]
y = y[indices]

# 转成张量
X = torch.tensor(X).unsqueeze(1)  # [N, 1, H, W]
y = torch.tensor(y)

split = int(len(X) * 0.8)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

print("train:", X_train.shape, y_train.shape)
print("val  :", X_val.shape, y_val.shape)
```

### 3.2 为什么要 `unsqueeze(1)`？

因为 PyTorch 的卷积输入需要：

- `[batch, channel, height, width]`

这里是灰度图，所以通道数是 1。

---

## 四、定义一个最小 CNN 分类器

```python
import torch
from torch import nn

class TinyCNNClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 3 * 3, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

model = TinyCNNClassifier(num_classes=3)
sample_out = model(X_train[:4])
print("sample output shape:", sample_out.shape)
```

### 4.2 为什么这里 `16 * 3 * 3`？

因为原图大小是 `12x12`：

- 第一次 `MaxPool2d(2)` 后变 `6x6`
- 第二次 `MaxPool2d(2)` 后变 `3x3`

最后输出通道是 16，所以展平后就是：

> `16 * 3 * 3`

这就是 CNN 实战里最常见的 shape 计算题。

---

## 五、完整训练循环

```python
import torch
from torch import nn

torch.manual_seed(42)

model = TinyCNNClassifier(num_classes=3)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(100):
    model.train()
    train_logits = model(X_train)
    train_loss = loss_fn(train_logits, y_train)

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val)
            val_loss = loss_fn(val_logits, y_val)
            train_acc = (train_logits.argmax(dim=1) == y_train).float().mean().item()
            val_acc = (val_logits.argmax(dim=1) == y_val).float().mean().item()

        print(
            f"epoch={epoch:3d}, "
            f"train_loss={train_loss.item():.4f}, "
            f"val_loss={val_loss.item():.4f}, "
            f"train_acc={train_acc:.3f}, "
            f"val_acc={val_acc:.3f}"
        )
```

### 5.2 这段代码里最该盯住什么？

初学图像分类时，最重要的是看这四样：

- `train_loss`
- `val_loss`
- `train_acc`
- `val_acc`

因为它们会告诉你：

- 模型有没有学到
- 有没有过拟合
- 是否还在稳定收敛

---

## 六、真正做一次预测

### 6.1 看单个样本

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    sample = X_val[0:1]
    pred = model(sample).argmax(dim=1).item()
    true = y_val[0].item()

plt.imshow(sample[0, 0].numpy(), cmap="gray")
plt.title(f"pred={pred}, true={true}")
plt.axis("off")
plt.show()
```

### 6.2 为什么这一步很重要？

因为很多时候：

- 指标看起来不错
- 但你并不知道模型到底在“看什么”

亲自看几张预测样本，会帮助你更快建立直觉。

---

## 七、怎样判断模型有没有真的学会？

### 7.1 几个典型信号

如果模型真的学到了：

- train loss 会下降
- val loss 通常也会下降
- train / val acc 会提高
- 对单样本预测越来越稳定

### 7.2 几个典型异常

#### 训练集和验证集都很差

可能是：

- 模型太弱
- 学习率不合适
- 数据构造有问题

#### 训练集很好，验证集很差

可能是：

- 过拟合
- 数据量太少
- 噪声过大

#### loss 不动

可能是：

- shape 错
- 标签错
- 学习率太小

---

## 八、真实图像分类项目还要补什么？

我们这个教学例子故意压得很小。  
真实项目里通常还要继续补：

- DataLoader
- 数据增强
- 更真实的数据集
- 更强 backbone
- 更系统的验证指标
- 模型保存与恢复

也就是说：

> 本节教的是“完整闭环”，不是“工业级最终方案”。 

---

## 九、初学者最常踩的坑

### 9.1 只会抄模型，不会检查数据 shape

图像任务里，shape 几乎永远是第一检查项。

### 9.2 只盯 train loss

验证集指标同样重要。

### 9.3 模型能跑起来就以为任务做完了

真正的项目，不只是跑通，而是要能解释结果。

---

## 小结

这一节最重要的不是把 CNN 跑通，而是把图像分类项目的完整闭环走通：

> **造数据 / 整理数据 / 定义模型 / 训练 / 验证 / 单样本预测。**

只有这条链真的完整了，后面你换更复杂的数据集和更强模型时，才不会一直发虚。

---

## 练习

1. 给当前数据再加一个第 4 类图像模式，比如“反对角线”。
2. 把 `TinyCNNClassifier` 的通道数改大，看收敛速度是否变化。
3. 尝试加一个 `Dropout`，观察验证集表现是否更稳。
4. 想一想：为什么图像分类项目里，数据构造和验证方式往往比多加几层网络更重要？
