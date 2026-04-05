---
title: "2.7 训练流程 🔧"
sidebar_position: 5
description: "把模型、损失函数、优化器和 DataLoader 串起来，写出一个完整可运行的 PyTorch 训练循环。"
keywords: [training loop, optimizer, loss, model.train, model.eval, PyTorch]
---

# 训练流程 🔧

## 学习目标

- 看懂并写出标准的 PyTorch 训练循环
- 明白 `train()`、`eval()`、`zero_grad()`、`backward()`、`step()` 的顺序
- 能对一个小任务完成训练、验证和预测
- 形成可复用的训练模板

---

## 先建立一张地图

训练循环这节最适合新人的理解方式不是“背模板”，而是先看清训练到底在重复什么：

```mermaid
flowchart LR
    A["取一个 batch"] --> B["模型前向预测"]
    B --> C["计算损失"]
    C --> D["反向传播梯度"]
    D --> E["优化器更新参数"]
    E --> F["进入下一个 batch"]
```

这五步不断循环，就是深度学习训练最核心的节奏。

## 这节和第四阶段、PyTorch 前几节是怎么接上的

如果你是从第四阶段一路走过来，可以先这样理解：

- 第四阶段里，`fit()` 已经帮你把训练这件事包起来了
- 到了这一节，你开始自己把训练拆开写

如果你是从 PyTorch 前几节接过来，也可以这样看：

- `Tensor` 解决“数据装在哪里”
- `Autograd` 解决“梯度怎么来”
- `nn.Module` 解决“网络怎么组织”
- `DataLoader` 解决“数据怎么分 batch 喂”
- 而这一节负责把这些东西真正串成会跑的训练过程

## 一、训练循环为什么重要？

深度学习代码里最值得反复练的，不是某一个层，而是**训练循环**。

因为不管你是做：

- 图像分类
- 文本分类
- 目标检测
- 大模型微调

训练主流程都逃不开这条线：

```mermaid
flowchart LR
    A["取一个 batch"] --> B["前向计算"]
    B --> C["计算损失"]
    C --> D["清空旧梯度"]
    D --> E["反向传播"]
    E --> F["优化器更新参数"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style B fill:#fff3e0,stroke:#e65100,color:#333
    style C fill:#ffebee,stroke:#c62828,color:#333
    style D fill:#f3e5f5,stroke:#6a1b9a,color:#333
    style E fill:#fffde7,stroke:#f9a825,color:#333
    style F fill:#e8f5e9,stroke:#2e7d32,color:#333
```

### 1.1 训练循环为什么比背网络结构更值得先练？

因为网络结构会变：

- CNN 会变
- RNN 会变
- Transformer 会变

但训练循环的大骨架长期都很稳定。  
所以这一节的价值非常高，它是在帮你抓住深度学习里最不容易过时的那部分。

---

## 二、先记住标准模板

先不要急着背，先多看几遍：

```python
for batch_x, batch_y in train_loader:
    pred = model(batch_x)
    loss = loss_fn(pred, batch_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

它其实只在做三件事：

1. 算预测
2. 算误差
3. 按误差更新参数

### 2.1 一个新人最该先背下来的口令

如果你每次写训练循环都会乱，可以先记这个最短口令：

`前向 -> 算 loss -> 清梯度 -> 反传 -> 更新`

只要这一条顺了，后面再加验证、日志、早停都不难。

### 2.2 为什么这个顺序不能乱？

因为每一步都依赖前一步的结果：

- 没有前向，就没有预测
- 没有预测，就没有 loss
- 没有 loss，就没法 backward
- 不清旧梯度，新的梯度就会和旧梯度混在一起

所以训练循环不是“几个 API 凑一起”，而是一条有严格顺序的因果链。

---

## 三、一个完整可运行例子

:::info 运行环境
下面代码可以直接运行：

```bash
pip install torch
```
:::

我们做一个二维回归任务。  
输入两个特征，目标值满足近似关系：

> `y ≈ 3*x1 + 2*x2 + 5`

```python
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, random_split

torch.manual_seed(42)

# 1. 造一份可直接运行的模拟数据
X = torch.randn(200, 2)
noise = torch.randn(200, 1) * 0.3
y = 3 * X[:, [0]] + 2 * X[:, [1]] + 5 + noise

dataset = TensorDataset(X, y)
train_dataset, val_dataset = random_split(
    dataset,
    [160, 40],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=40, shuffle=False)

# 2. 定义模型
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 1)
)

# 3. 定义损失函数和优化器
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# 4. 训练
for epoch in range(1, 101):
    model.train()
    train_loss_sum = 0.0

    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss_sum += loss.item() * len(batch_x)

    train_loss = train_loss_sum / len(train_dataset)

    # 5. 验证
    model.eval()
    with torch.no_grad():
        val_loss_sum = 0.0
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            val_loss_sum += loss.item() * len(batch_x)
        val_loss = val_loss_sum / len(val_dataset)

    if epoch % 20 == 0 or epoch == 1:
        print(f"epoch={epoch:3d}, train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# 6. 测试预测
test_x = torch.tensor([[1.0, 2.0], [-1.0, 0.5], [0.0, 0.0]])
with torch.no_grad():
    test_pred = model(test_x)

print("\n测试样本预测:")
for x_row, y_row in zip(test_x, test_pred):
    print(f"x={x_row.tolist()} -> pred={round(y_row.item(), 2)}")
```

---

## 四、逐行拆解这段代码

### 1. `model.train()`

告诉模型进入训练模式。  
如果模型里有 `Dropout`、`BatchNorm` 这样的层，它们会切换到训练行为。

### 2. `pred = model(batch_x)`

前向传播。  
也就是“拿当前参数做一次预测”。

### 3. `loss = loss_fn(pred, batch_y)`

告诉模型：“你这次和真实答案差多少。”

### 4. `optimizer.zero_grad()`

清空旧梯度。  
因为 PyTorch 默认会累计梯度。

### 5. `loss.backward()`

反向传播。  
把损失对各参数的梯度算出来。

### 6. `optimizer.step()`

根据梯度真正更新参数。

### 4.1 新人第一次自己写时，最容易漏哪一步？

最常见的是这两处：

- 忘了 `optimizer.zero_grad()`
- 验证阶段忘了 `model.eval()` 和 `torch.no_grad()`

这两个问题都会让训练结果看起来“怪怪的”，但又不一定立刻报错。

### 4.2 一个更适合新人的“每轮训练检查表”

你可以在脑子里每轮都过一遍这张小表：

| 步骤 | 我要确认什么 |
|---|---|
| 前向 | 输入 shape 对吗？输出 shape 对吗？ |
| loss | 输出和标签能对上吗？ |
| zero_grad | 旧梯度清了吗？ |
| backward | 梯度真的算出来了吗？ |
| step | 参数真的更新了吗？ |

这张表对排错特别有帮助，因为很多训练 bug 都发生在这 5 个问题里。

---

## 五、为什么验证要用 `eval()` 和 `no_grad()`？

验证阶段的目标不是学习，而是检查模型表现。

所以我们一般会这样写：

```python
model.eval()
with torch.no_grad():
    ...
```

原因有两个：

- `eval()`：让某些层切换成推理模式
- `no_grad()`：不记录梯度，省内存、省时间

### 5.1 初学阶段先把训练态和验证态分清，有多重要？

这一步特别容易被忽略，因为很多最小例子里没明显问题。  
但从这一节开始，你最好养成一个稳定习惯：

- 训练前：`model.train()`
- 验证前：`model.eval()`
- 验证时：`with torch.no_grad():`

因为后面一旦出现：

- Dropout
- BatchNorm
- 更大的模型

训练态和验证态不分清，结果会越来越容易出错。

---

## 六、一个更适合记忆的“厨房版类比”

把训练看成开餐厅会很好记：

| 深度学习步骤 | 餐厅类比 |
|---|---|
| `batch_x` | 一批顾客订单 |
| `model(batch_x)` | 厨师按当前手法做菜 |
| `loss_fn` | 顾客给评分 |
| `backward()` | 找出是哪里做得不好 |
| `step()` | 下次做菜时调整手法 |

训练就是反复营业、反复改进。

---

## 七、常见变体

### 1. 分类任务

回归常用 `MSELoss()`，分类更常见：

```python
loss_fn = nn.CrossEntropyLoss()
```

### 2. 不同优化器

最常见的两个：

- `SGD`
- `Adam`

初学阶段，`Adam` 往往更省心一些。

### 3. 统计指标

训练时除了 loss，还常常统计：

- 准确率 `accuracy`
- 精确率 `precision`
- 召回率 `recall`
- F1

---

## 八、最容易写错的地方

### 1. 忘记 `zero_grad()`

后果：梯度不断累加，训练结果不可信。

### 2. 验证时忘记 `model.eval()`

有些模型层在训练 / 验证模式下行为不同，会影响结果。

### 3. 验证时也在算梯度

虽然可能能跑，但浪费内存与算力。

### 4. `loss.item()` 和 `loss` 混着用

- `loss` 是张量，能参与反向传播
- `loss.item()` 是普通 Python 数字，适合打印和统计

### 5. 只看 loss，不看训练和验证的关系

新人最常见的另一个问题是：

- 看到训练 loss 在降，就以为一切正常

但更稳的判断方式应该是：

- 训练 loss 在降吗？
- 验证 loss 也在同步变好吗？
- 两者是不是开始分叉？

这其实已经是在为后面的过拟合诊断做准备。

---

## 九、一个你可以保存下来的通用骨架

```python
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        pred = model(batch_x)
        loss = loss_fn(pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            pred = model(batch_x)
            val_loss = loss_fn(pred, batch_y)
```

以后你看到任何 PyTorch 项目，基本都能在里面认出这条主线。

---

## 小结

如果这节你只记住一句话，那就是：

> **训练循环就是“前向算一次，反向改一次，然后重复很多次”。**

把这条链路练熟，后面学 CNN、Transformer、微调大模型时，你不会总被框架代码吓住。

## 这节最该带走什么

如果再多带走一句，我希望你记住：

> **训练循环不是模板记忆题，而是一条“预测 -> 衡量误差 -> 根据误差改参数”的闭环。**

所以这一节真正要稳住的是：

- 顺序不能乱
- 训练态和验证态要分开
- 排错时先查 shape、loss、梯度、参数更新这几步

---

## 练习

1. 把上面例子里的优化器从 `Adam` 改成 `SGD`，看看收敛速度有什么差异。
2. 把隐藏层从 `8` 改成 `16`，观察训练和验证损失变化。
3. 把数据中的噪声 `0.3` 改成 `1.0`，看看模型训练难度会发生什么变化。
