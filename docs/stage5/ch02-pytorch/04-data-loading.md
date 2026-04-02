---
title: "2.4 数据加载"
sidebar_position: 4
description: "理解 Dataset、DataLoader、batch、shuffle 和训练集划分，让模型能稳定地一批批吃数据。"
keywords: [Dataset, DataLoader, batch, shuffle, random_split, PyTorch]
---

# 数据加载

## 学习目标

- 理解为什么训练时几乎不会一次性把所有数据直接塞进模型
- 掌握 `Dataset` 和 `DataLoader` 的分工
- 能自己写一个最简单的自定义数据集
- 理解 `batch_size`、`shuffle`、训练集 / 验证集划分

---

## 一、为什么需要数据加载器？

想象你在喂模型吃饭。

- 一次把全部数据倒进去：太撑，内存可能扛不住
- 一口一口喂：更稳定，也更适合反复训练

在深度学习里，这“一口”就叫一个 **batch**。

所以我们通常不会这样做：

```python
pred = model(all_data_once)
```

而是会这样做：

```python
for batch_x, batch_y in dataloader:
    pred = model(batch_x)
```

---

## 二、`Dataset` 和 `DataLoader` 各负责什么？

可以把它们理解成：

| 组件 | 类比 | 作用 |
|---|---|---|
| `Dataset` | 仓库 | 告诉 PyTorch “第 i 条数据是什么” |
| `DataLoader` | 搬运车 | 负责分批、打乱、并行加载 |

一句话记忆：

- `Dataset` 负责“单条数据怎么取”
- `DataLoader` 负责“怎么把单条数据凑成一批”

---

## 三、先看一个最小 `Dataset`

```python
import torch
from torch.utils.data import Dataset

class StudentDataset(Dataset):
    def __init__(self):
        # 两个特征：学习时长、完成练习数
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0]
        ])

        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()

print("数据集大小:", len(dataset))
print("第 0 条样本:", dataset[0])
print("第 3 条样本:", dataset[3])
```

### 自定义数据集必须实现什么？

通常最基本只需要两个方法：

- `__len__()`：返回总样本数
- `__getitem__(idx)`：返回第 `idx` 条数据

---

## 四、把数据集交给 `DataLoader`

```python
import torch
from torch.utils.data import Dataset, DataLoader

class StudentDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0]
        ])
        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()
loader = DataLoader(dataset, batch_size=2, shuffle=True)

for batch_idx, (batch_x, batch_y) in enumerate(loader):
    print(f"batch {batch_idx}")
    print("batch_x:\n", batch_x)
    print("batch_y:\n", batch_y)
```

### 这里最关键的两个参数

| 参数 | 作用 |
|---|---|
| `batch_size=2` | 每次取 2 条样本 |
| `shuffle=True` | 每个 epoch 开头打乱顺序 |

---

## 五、为什么要打乱数据？

如果你的数据原本是按某种顺序排好的，比如：

- 前 100 条都是低分样本
- 后 100 条都是高分样本

那模型前期会连续看到一大段同类样本，训练容易偏。  
所以训练集一般都建议 `shuffle=True`。

但验证集 / 测试集通常不需要打乱：

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
```

---

## 六、训练集和验证集怎么拆？

PyTorch 提供了 `random_split`：

```python
import torch
from torch.utils.data import Dataset, random_split

class StudentDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0]
        ])
        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()

train_dataset, val_dataset = random_split(
    dataset,
    [4, 2],
    generator=torch.Generator().manual_seed(42)
)

print("训练集大小:", len(train_dataset))
print("验证集大小:", len(val_dataset))
```

### 为什么这里要设随机种子？

因为不设的话，每次切分结果都可能不同。  
学习和调试阶段，固定随机种子更方便复现。

---

## 七、一个完整可运行的小流程

```python
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class StudentDataset(Dataset):
    def __init__(self):
        self.features = torch.tensor([
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 3.0],
            [5.0, 5.0],
            [6.0, 6.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0]
        ])
        self.labels = torch.tensor([
            [55.0],
            [60.0],
            [68.0],
            [78.0],
            [85.0],
            [92.0],
            [96.0],
            [99.0]
        ])

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

dataset = StudentDataset()

train_dataset, val_dataset = random_split(
    dataset,
    [6, 2],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)

print("训练集批次:")
for x, y in train_loader:
    print(x.shape, y.shape)

print("\n验证集批次:")
for x, y in val_loader:
    print(x.shape, y.shape)
```

---

## 八、`batch_size` 应该怎么选？

初学时先记住直觉版：

- 小 batch：更新更频繁，噪声更大
- 大 batch：更稳定，但更吃显存

如果你现在只是跑教学示例，通常选：

- `8`
- `16`
- `32`

就够用了。

等你开始训练更大的模型，再考虑显存和吞吐量平衡。

---

## 九、初学者常见误区

### 1. 以为 `Dataset` 就是把所有数据都读进内存

不一定。  
教学示例里我们确实这样写，但真实工程里，`__getitem__()` 常常会在访问时再去读磁盘文件。

### 2. 训练集也不打乱

可能能跑，但通常不是好习惯。

### 3. 只会写数组，不会写数据集类

小实验可以偷懒，稍微正规一点的项目都建议写成 `Dataset`。

---

## 十、小结

这节课最重要的不是背类名，而是建立“数据流”意识：

1. 数据先按样本组织在 `Dataset`
2. 再由 `DataLoader` 凑成 batch
3. 然后一批一批送进模型

下一节我们就把模型、损失、优化器和数据加载器连起来，写出完整训练流程。

---

## 练习

1. 把 `StudentDataset` 里的样本量扩展到 12 条，重新划分训练集和验证集。
2. 把 `batch_size` 改成 `1`、`2`、`4`，观察每轮迭代的 batch 数量。
3. 打印 `shuffle=True` 时连续两轮加载的第一批数据，看看顺序是否变化。
