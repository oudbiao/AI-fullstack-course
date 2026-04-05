---
title: "4.4 序列建模实战"
sidebar_position: 3
description: "用一个真正能训练的小型时间序列任务，把窗口构造、RNN/LSTM 训练、验证和预测串起来。"
keywords: [sequence modeling, time series, RNN, LSTM, sliding window, forecast]
---

# 序列建模实战

:::tip 本节定位
前两节你已经理解了：

- RNN 在“边读边记”
- LSTM / GRU 在“更聪明地控制记忆”

这一节要把这些概念真正落到一个小项目上：

> **给一段序列，预测后面的值。**
:::

## 学习目标

- 学会把连续序列切成训练样本
- 用 LSTM 搭一个最小时间序列预测器
- 理解训练集、验证集和预测流程
- 学会判断模型是在学规律还是在瞎记
- 知道序列实战里最常见的坑是什么

---

## 一、为什么选“时间序列预测”来做实战？

### 1.1 因为它最适合练序列建模的基本功

很多序列任务都可以抽象成：

- 前面一段输入
- 后面一个输出

时间序列预测就是最典型的例子。

比如：

- 根据过去 7 天销量，预测第 8 天销量
- 根据过去 12 个温度值，预测下一个温度

### 1.2 一个非常重要的直觉

做这类任务时，模型不是在记单个数字，而是在学：

> **变化模式。**

例如：

- 周期
- 趋势
- 波动

这和普通分类任务很不一样。

---

## 二、先造一份可以直接运行的数据

### 2.1 用正弦波 + 噪声造一个最小序列

这样做的好处是：

- 不依赖外部数据集
- 模式清晰
- 非常适合教学

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05

plt.figure(figsize=(10, 4))
plt.plot(t, series)
plt.title("Toy Time Series")
plt.grid(True, alpha=0.3)
plt.show()
```

### 2.2 这串数据长什么样？

它有两个特征：

- 整体是波动的
- 带一点随机噪声

这就比完全规则的序列更接近真实任务一点。

---

## 三、滑动窗口：怎么把一整段序列切成样本？

### 3.1 核心思想

模型不能直接吃“一整条无限序列”。  
我们通常会把它切成很多小片段：

- 前 `window_size` 个点作为输入
- 第 `window_size + 1` 个点作为标签

这就叫滑动窗口。

### 3.2 可运行示例

```python
import numpy as np

series = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.float32)
window_size = 3

X, y = [], []
for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = np.array(X)
y = np.array(y)

print("X =\n", X)
print("y =", y)
```

### 3.3 这一步为什么这么关键？

因为它决定了序列任务的样本定义。  
如果窗口构造错了，后面的训练、验证和预测都会跟着错。

---

## 四、把数据整理成 PyTorch 可训练格式

### 4.1 完整数据准备

```python
import numpy as np
import torch

np.random.seed(42)
torch.manual_seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05
series = series.astype(np.float32)

window_size = 12
X, y = [], []

for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = np.array(X)
y = np.array(y)

# 转成 [batch, seq_len, input_size]
X = torch.tensor(X).unsqueeze(-1)
y = torch.tensor(y).unsqueeze(-1)

print("X shape:", X.shape)
print("y shape:", y.shape)
```

### 4.2 为什么要 `unsqueeze(-1)`？

因为 LSTM 期望的输入通常是：

- `[batch, seq_len, input_size]`

这里每个时间步只有 1 个特征值，所以：

- `input_size = 1`

---

## 五、一个真正能训练的小型 LSTM 预测器

### 5.1 定义模型

```python
import torch
from torch import nn

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.fc(last_hidden)
```

### 5.2 为什么只取最后一个时间步？

因为当前这个任务是：

> 用前一段窗口，预测“下一个值”

所以最自然的做法是拿序列最后时刻的表示，作为整段窗口的摘要。

---

## 六、完整训练流程

### 6.1 训练 + 验证

```python
import numpy as np
import torch
from torch import nn

np.random.seed(42)
torch.manual_seed(42)

t = np.arange(0, 200)
series = np.sin(t * 0.1) + np.random.randn(200) * 0.05
series = series.astype(np.float32)

window_size = 12
X, y = [], []
for i in range(len(series) - window_size):
    X.append(series[i:i + window_size])
    y.append(series[i + window_size])

X = torch.tensor(np.array(X)).unsqueeze(-1)
y = torch.tensor(np.array(y)).unsqueeze(-1)

train_size = int(len(X) * 0.8)
X_train, X_val = X[:train_size], X[train_size:]
y_train, y_val = y[:train_size], y[train_size:]

class LSTMForecaster(nn.Module):
    def __init__(self, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = LSTMForecaster(hidden_size=32)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(200):
    model.train()
    pred = model(X_train)
    loss = loss_fn(pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(X_val), y_val)
        print(f"epoch={epoch:3d}, train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}")
```

### 6.2 这段训练代码真正值得你盯住什么？

最关键的是：

- 输入 shape 对不对
- 最后只取 `out[:, -1, :]`
- 损失是否确实在下降

这三点一旦搞明白，你就已经真正迈进序列建模实战了。

---

## 七、做一次真实预测

### 7.1 单窗口预测

```python
model.eval()
with torch.no_grad():
    sample_x = X_val[0:1]
    pred = model(sample_x)
    print("预测值:", float(pred.item()))
    print("真实值:", float(y_val[0].item()))
```

### 7.2 画出预测和真实值

```python
import matplotlib.pyplot as plt

model.eval()
with torch.no_grad():
    val_pred = model(X_val).squeeze(-1).numpy()
    val_true = y_val.squeeze(-1).numpy()

plt.figure(figsize=(10, 4))
plt.plot(val_true, label="true")
plt.plot(val_pred, label="pred")
plt.legend()
plt.title("Validation Prediction")
plt.grid(True, alpha=0.3)
plt.show()
```

真正做序列任务时，图往往比单个指标更能帮你发现问题：

- 模型是不是跟不上波峰波谷
- 是不是有整体滞后
- 是不是学成了一条平线

---

## 八、序列建模实战里最常见的坑

### 8.1 数据泄漏

如果你切训练集 / 验证集方式不对，很可能会把未来信息泄露给模型。

时间序列任务里，最稳妥的原则通常是：

> 按时间顺序切，不要乱打乱。 

### 8.2 窗口太短或太长

- 太短：模型看不到足够历史
- 太长：训练更难，噪声也更多

### 8.3 只看 loss，不看曲线

序列预测里，画图常常非常重要。  
因为两个 loss 相近的模型，走势可能完全不同。

### 8.4 以为学到了“因果”，其实只是学到了短期模式

这是所有序列预测都要小心的地方。  
模型会预测，不代表它真的理解机制。

---

## 九、一个很重要的工程直觉

实际项目里，序列任务不一定都用 RNN / LSTM。  
今天很多任务也会用：

- Transformer
- Temporal Convolution
- 传统统计模型

但不管你以后用什么模型，这一节教你的窗口构造、时序切分、验证方式，依然是基础。

---

## 小结

这一节最重要的不是“把 LSTM 跑起来”，而是理解：

> **序列实战的关键，在于怎样把连续数据切成训练样本，并让模型在不泄漏未来信息的前提下学到变化规律。**

当你能把数据构造、训练流程、验证和预测画图这几步串起来时，序列建模才算真正落地。

---

## 练习

1. 把 `window_size` 从 12 改成 6 和 24，比较预测效果。
2. 把模型从 LSTM 改成 GRU，看看训练曲线是否不同。
3. 故意把训练集和验证集随机打乱，再思考为什么这对时间序列是危险的。
4. 想一想：如果你的序列有明显周周期，窗口长度应该怎样设计？
