---
title: "6.3.5 迁移学习"
sidebar_position: 4
description: "一步步实操迁移学习：预训练小 backbone、替换 head、冻结参数、微调最后一个 block，并读懂取舍。"
keywords: [迁移学习, fine-tuning, feature extractor, freeze backbone, transfer learning, CNN]
---

# 6.3.5 迁移学习

:::tip 本节定位
迁移学习是很多视觉项目的默认起点：复用一个已经学过通用视觉模式的 backbone，替换任务相关的 head，只有当验证结果说明需要时，才继续微调更多层。
:::

## 学习目标

- 解释为什么从零训练 CNN 往往很浪费。
- 区分 pretrained backbone 和 classification head。
- 冻结 backbone，只训练新的 head。
- 用更小学习率解冻最后一个卷积 block 做微调。
- 避免数据泄漏、破坏性微调等常见迁移学习错误。

---

## 先看决策流程

![迁移学习冻结 backbone 与逐步微调决策图](/img/course/ch06-transfer-learning-freeze-finetune-map.webp)

按这条线读图：

```text
pretrained backbone -> replace head -> train head -> validate -> 必要时解冻后面层
```

两个问题决定选择：

| 问题 | 数据少 / 任务相近 | 数据多 / 任务差异大 |
|---|---|---|
| 你有多少标注数据？ | 先冻结大部分层 | 谨慎微调更多层 |
| 新任务和原任务像不像？ | 预训练特征可能迁移得很好 | 后面层可能需要适配 |

本节用纯 PyTorch 和合成图像，所以不需要下载 `torchvision` 权重也能运行。真实项目中，backbone 通常来自预训练的 `torchvision` 或 `timm` 模型。

## 核心术语

| 术语 | 含义 |
|---|---|
| backbone | 特征提取器，通常是最终分类器之前的所有层 |
| head | 接在 backbone 后面的任务相关分类器或回归器 |
| freeze | 设置 `requires_grad=False`，让参数不更新 |
| fine-tune | 解冻一部分预训练层并继续训练 |
| logits | `softmax` 之前的原始类别分数 |

实践规则：

```text
数据少 -> 先训练 head
效果不够 -> 用更小学习率解冻后面的 backbone 层
```

## 完整实验：离线模拟迁移学习

这个实验有三个阶段：

1. 在简单线条图案上预训练一个 tiny backbone。
2. 在新的目标任务上复用 backbone，只训练 head。
3. 解冻最后一个卷积层，用更小学习率轻微微调。

运行完整脚本：

```python
import copy
import numpy as np
import torch
from torch import nn

SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)


def make_image(label, task, size=16, noise=0.05):
    img = np.zeros((size, size), dtype=np.float32)
    c = size // 2

    if task == "source":
        if label == 0:
            img[:, c] = 1.0
        elif label == 1:
            img[c, :] = 1.0
        else:
            for i in range(size):
                img[i, i] = 1.0
    elif task == "target":
        if label == 0:
            img[:, c] = 1.0
            img[c, :] = 1.0
        elif label == 1:
            for i in range(size):
                img[i, size - 1 - i] = 1.0
        else:
            img[3:-3, 3] = 1.0
            img[3:-3, -4] = 1.0
            img[3, 3:-3] = 1.0
            img[-4, 3:-3] = 1.0

    img += np.random.randn(size, size).astype(np.float32) * noise
    return np.clip(img, 0.0, 1.0)


def make_dataset(task, per_class, size=16):
    X, y = [], []
    for label in range(3):
        for _ in range(per_class):
            X.append(make_image(label, task, size=size))
            y.append(label)
    X = torch.tensor(np.array(X)).unsqueeze(1)
    y = torch.tensor(np.array(y), dtype=torch.long)
    return X, y


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

    def forward(self, x):
        return self.features(x).flatten(1)


class ImageClassifier(nn.Module):
    def __init__(self, backbone=None, num_classes=3):
        super().__init__()
        self.backbone = backbone if backbone is not None else TinyBackbone()
        self.head = nn.Linear(16, num_classes)

    def forward(self, x):
        return self.head(self.backbone(x))


def accuracy(model, X, y):
    with torch.no_grad():
        return (model(X).argmax(dim=1) == y).float().mean().item()


def train(model, X, y, optimizer, epochs, label, print_every):
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(1, epochs + 1):
        logits = model(X)
        loss = loss_fn(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % print_every == 0:
            acc = accuracy(model, X, y)
            print(f"{label} epoch={epoch:02d} loss={loss.item():.4f} acc={acc:.3f}")


source_X, source_y = make_dataset("source", per_class=80)
target_train_X, target_train_y = make_dataset("target", per_class=12)
target_val_X, target_val_y = make_dataset("target", per_class=40)

# Stage 1: pretrain a source model.
source_model = ImageClassifier(num_classes=3)
train(
    source_model,
    source_X,
    source_y,
    torch.optim.Adam(source_model.parameters(), lr=0.03),
    epochs=60,
    label="pretrain",
    print_every=20,
)

# Stage 2: transfer the backbone and train only a new head.
frozen_backbone = copy.deepcopy(source_model.backbone)
transfer_model = ImageClassifier(backbone=frozen_backbone, num_classes=3)
for p in transfer_model.backbone.parameters():
    p.requires_grad = False

print("trainable_after_freeze")
for name, p in transfer_model.named_parameters():
    print(f"{name:<28} {p.requires_grad}")

train(
    transfer_model,
    target_train_X,
    target_train_y,
    torch.optim.Adam(transfer_model.head.parameters(), lr=0.05),
    epochs=20,
    label="head",
    print_every=10,
)
print("head_only_val_acc", round(accuracy(transfer_model, target_val_X, target_val_y), 3))

# Stage 3: unfreeze the last conv layer and fine-tune gently.
for p in transfer_model.backbone.features[3].parameters():
    p.requires_grad = True

optimizer = torch.optim.Adam(
    [
        {"params": transfer_model.backbone.features[3].parameters(), "lr": 0.0005},
        {"params": transfer_model.head.parameters(), "lr": 0.005},
    ]
)
train(
    transfer_model,
    target_train_X,
    target_train_y,
    optimizer,
    epochs=20,
    label="finetune",
    print_every=10,
)
print("finetune_val_acc", round(accuracy(transfer_model, target_val_X, target_val_y), 3))
```

预期输出：

```text
pretrain epoch=01 loss=1.0995 acc=0.667
pretrain epoch=20 loss=0.0000 acc=1.000
pretrain epoch=40 loss=0.0000 acc=1.000
pretrain epoch=60 loss=0.0000 acc=1.000
trainable_after_freeze
backbone.features.0.weight   False
backbone.features.0.bias     False
backbone.features.3.weight   False
backbone.features.3.bias     False
head.weight                  True
head.bias                    True
head epoch=01 loss=2.4749 acc=0.361
head epoch=10 loss=0.7364 acc=0.667
head epoch=20 loss=0.4991 acc=0.944
head_only_val_acc 0.875
finetune epoch=01 loss=0.4759 acc=0.667
finetune epoch=10 loss=0.4367 acc=1.000
finetune epoch=20 loss=0.4096 acc=1.000
finetune_val_acc 1.0
```

![迁移学习实验结果图](/img/course/ch06-transfer-learning-lab-result-map.webp)

这张图分三步读：

- `pretrain` 说明 tiny backbone 已经能提取可复用的线条特征。
- `trainable_after_freeze` 是安全检查：backbone 保持冻结，只有新的 head 更新。
- `head_only_val_acc=0.875` 已经可用，而 `finetune_val_acc=1.0` 说明只轻微解冻最后卷积层，在这个验证集上确实有帮助。

## 这个实验说明了什么

| 阶段 | 发生了什么 | 实践含义 |
|---|---|---|
| pretrain | backbone 学到类似线条的视觉特征 | 这里模拟真实预训练模型 |
| freeze | 只有新 head 可训练 | 对小目标数据更快、更安全 |
| train head | 目标验证准确率已经可用 | 复用特征已经有帮助 |
| fine-tune | 最后一个卷积层轻微适配 | 小学习率能降低破坏旧特征的风险 |

微调并不自动更好。如果目标数据很少，或者学习率太大，它可能过拟合，也可能破坏预训练特征。判断标准永远是验证集，而不是训练 loss。

## 真实项目流程

1. 在碰模型之前，先切好 train/validation/test。
2. 加载预训练 backbone。
3. 替换 head，让输出类别数匹配你的任务。
4. 冻结 backbone，只训练 head。
5. 查看验证集错误。
6. 如有必要，解冻后面的 block，并给 backbone 更小学习率。
7. 验证集不再提升时停止。

对真实图片，还要匹配预训练权重期待的预处理：输入尺寸、归一化 mean/std、颜色通道顺序。

## 冻结还是微调？

| 情况 | 起步选择 |
|---|---|
| 数据很少，任务相近 | 冻结 backbone，只训练 head |
| 中等数据，任务相近 | 先冻结，再解冻最后一个 block |
| 数据较多，视觉领域不同 | 谨慎微调更多 block |
| 医疗、遥感、工业图像 | 认真验证；自然图像预训练特征可能只部分迁移 |
| 部署设备受限 | 先用更小 backbone 或 freeze-and-head baseline |

## 常见错误

| 错误 | 为什么有问题 | 修复 |
|---|---|---|
| 一开始就微调所有层 | 小数据上不稳定 | 先训练 head |
| 所有层用同一个学习率 | backbone 更新太猛 | 给预训练层更小 LR |
| 忘记检查 `requires_grad` | 错误层静悄悄地训练 | 打印可训练参数 |
| 只在训练集评估 | 看不出过拟合 | 保留验证集 |
| 预处理不匹配 | 预训练特征收到陌生输入尺度 | 使用权重要求的 transform |
| split 泄漏 | 验证集失去意义 | 必要时按图片来源、用户或物体分组切分 |

## 练习

1. 增加第 4 个目标类别，并设计一个新的合成图案。
2. 把目标训练数据从每类 `12` 张增加到 `40` 张，只训 head 会不会更好？
3. 把 backbone 微调学习率从 `0.0005` 改成 `0.05`，观察发生什么。
4. 解冻最后一个卷积后，只打印可训练参数名。
5. 解释什么时候 GAP 加小 head 比大型 `Flatten` head 更合适。

## 小结

- 迁移学习复用视觉特征，而不是从零重新学一切。
- 最安全的第一个 baseline 通常是：替换 head、冻结 backbone、训练 head。
- 只有验证结果说明有必要时，才微调后面层。
- 预训练层要用更小学习率。
- 好的迁移学习是一套工程流程，不只是复制一个大模型。
