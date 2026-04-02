---
title: "2.6 实用技巧"
sidebar_position: 6
description: "从设备切换、随机种子、AMP、梯度裁剪到 checkpoint，掌握 PyTorch 训练中最常见也最实用的工程技巧。"
keywords: [PyTorch, AMP, 混合精度, 梯度裁剪, checkpoint, device, reproducibility]
---

# 实用技巧

## 学习目标

完成本节后，你将能够：

- 正确处理 CPU / GPU 设备切换
- 使用随机种子提升实验可复现性
- 理解混合精度训练和梯度裁剪的作用
- 会保存和恢复模型 checkpoint
- 建立一份 PyTorch 调试检查清单

---

## 一、先解决最常见的工程问题

### 1.1 设备切换：先别假设你一定有 GPU

很多初学者会直接把代码写死成 `cuda()`，结果在没有 GPU 的机器上直接报错。

更稳妥的写法是：

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("当前设备:", device)

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]]).to(device)
print(x)
print("张量所在设备:", x.device)
```

你可以把 `device` 理解成“训练发生在哪张工作台上”：

- CPU：普通桌面
- GPU：并行运算的大工作台

### 1.2 固定随机种子：让实验尽量可复现

训练不稳定时，第一件事往往不是改模型，而是先固定随机性。

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

print(torch.randn(3))
set_seed(42)
print(torch.randn(3))
```

如果两次打印结果一样，说明这部分随机性被固定住了。

:::info 为什么“尽量”而不是“绝对”？
有些 GPU 算子和并行细节仍然可能引入微小差异，所以可复现通常是“更接近”，不是“绝对一模一样”。
:::

---

## 二、让训练过程更稳

### 2.1 `train()`、`eval()` 和 `no_grad()` 要形成肌肉记忆

训练与验证最容易写乱的地方，不是模型结构，而是模式切换。

标准习惯：

```python
model.train()   # 训练前
...
model.eval()    # 验证 / 推理前
with torch.no_grad():
    ...
```

你可以把它理解成：

- `train()`：模型进入“练习模式”
- `eval()`：模型进入“考试模式”
- `no_grad()`：考试时不做反向传播草稿，节省内存

### 2.2 梯度裁剪：防止梯度突然爆掉

在 RNN、Transformer 或较深网络里，梯度有时会变得很大，导致训练不稳定。  
梯度裁剪就是“给梯度设一个上限”。

```python
import torch
from torch import nn

torch.manual_seed(42)

model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 1)
)

x = torch.randn(32, 10)
y = torch.randn(32, 1) * 50

loss_fn = nn.MSELoss()
pred = model(x)
loss = loss_fn(pred, y)
loss.backward()

def grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.norm(2).item() ** 2
    return total ** 0.5

before = grad_norm(model)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
after = grad_norm(model)

print("裁剪前梯度范数:", round(before, 4))
print("裁剪后梯度范数:", round(after, 4))
```

这就像给下坡的自行车加个限速器，避免冲得太猛。

---

## 三、让训练更快

### 3.1 混合精度训练（AMP）：更省显存、更快

AMP 的核心思想是：

> 在合适的地方用更低精度计算，以换取更快速度和更低显存占用。

它尤其适合 GPU 训练。  
为了保证下面代码在没有 GPU 的机器上也能直接运行，我们写成“有 GPU 就启用，没有就正常训练”。

```python
import torch
from torch import nn

device = "cuda" if torch.cuda.is_available() else "cpu"
model = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 1)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

x = torch.randn(64, 16).to(device)
y = torch.randn(64, 1).to(device)

if device == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    for _ in range(3):
        optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            pred = model(x)
            loss = loss_fn(pred, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print("已使用 AMP 在 GPU 上完成 3 步训练")
else:
    for _ in range(3):
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
    print("当前无 GPU，使用普通精度完成 3 步训练")
```

### 3.2 Batch 太大怎么办？

如果你经常遇到显存不够：

1. 先减小 `batch_size`
2. 再考虑 AMP
3. 再考虑梯度累积

梯度累积的直觉是：

> 虽然一次吃不下大 batch，但可以分几口吃完，再统一更新一次。

---

## 四、保存和恢复训练进度

### 4.1 为什么 checkpoint 很重要？

训练随时可能因为这些原因中断：

- 断电
- Notebook 超时
- GPU 被回收
- 程序报错

checkpoint 就像“游戏存档”。

### 4.2 一个最小可运行示例

```python
import torch
from torch import nn

model = nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

checkpoint_path = "demo_checkpoint.pt"

# 保存
torch.save({
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "epoch": 5
}, checkpoint_path)

print("checkpoint 已保存:", checkpoint_path)

# 恢复
new_model = nn.Linear(2, 1)
new_optimizer = torch.optim.SGD(new_model.parameters(), lr=0.1)

ckpt = torch.load(checkpoint_path, map_location="cpu")
new_model.load_state_dict(ckpt["model_state_dict"])
new_optimizer.load_state_dict(ckpt["optimizer_state_dict"])

print("恢复的 epoch:", ckpt["epoch"])
```

真实项目里，通常会额外保存：

- 最优验证集指标
- 训练配置
- tokenizer / label mapping

---

## 五、调试时先看哪里？

### 5.1 形状（shape）永远排第一

PyTorch 里很多 bug，本质上都不是“模型太难”，而是：

- shape 不对
- dtype 不对
- device 不一致

训练前建议多打几行：

```python
print("x shape:", x.shape)
print("y shape:", y.shape)
print("x dtype:", x.dtype)
print("x device:", x.device)
```

### 5.2 训练不下降时的检查顺序

可以按这个顺序查：

1. 数据有没有读对
2. 标签有没有对齐
3. loss 有没有算对
4. `optimizer.zero_grad()` 有没有写
5. `backward()` 和 `step()` 顺序对不对
6. 学习率是不是太大或太小

### 5.3 看到 `nan` 怎么办？

常见原因有：

- 学习率太大
- 输入尺度过大
- 梯度爆炸
- 除零或 `log(0)` 等数值问题

最实用的第一反应：

1. 降低学习率
2. 打印 loss 和参数范围
3. 打开梯度裁剪

---

## 六、一份适合保存的训练模板

```python
model.train()
for batch_x, batch_y in train_loader:
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)

    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

model.eval()
with torch.no_grad():
    for batch_x, batch_y in val_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        pred = model(batch_x)
        val_loss = loss_fn(pred, batch_y)
```

这个模板不花哨，但非常实用。

---

## 小结

这节课最重要的不是新 API，而是训练工程直觉：

- 设备别写死
- 随机种子先固定
- `train / eval / no_grad` 要分清
- 大梯度要会裁
- 训练进度要会存

很多模型训练卡住，不是因为算法不会，而是这些“小工程细节”没处理好。

---

## 练习

1. 给你自己的 PyTorch 训练代码加上 `device` 处理，确保 CPU 和 GPU 都能跑。
2. 在现有训练循环里加上梯度裁剪，打印裁剪前后的梯度范数。
3. 加一个 checkpoint 存档逻辑，并在中断后尝试恢复。
