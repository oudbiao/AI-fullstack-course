---
title: "2.2 自动求导"
sidebar_position: 2
description: "理解 requires_grad、backward、梯度累计和 no_grad，真正明白训练时参数为什么会更新。"
keywords: [autograd, backward, gradient, requires_grad, no_grad, PyTorch]
---

# 自动求导

## 学习目标

- 理解梯度到底是什么
- 掌握 `requires_grad=True` 的作用
- 明白 `loss.backward()` 做了什么
- 理解梯度累计、清零和 `torch.no_grad()`

---

## 一、为什么要有自动求导？

训练模型的核心目标只有一句话：

> **让模型参数朝“损失更小”的方向移动。**

问题是：怎么知道该往哪个方向移动？

答案就是**梯度（gradient）**。

你可以把梯度想成“山坡的坡度”：

- 梯度大，说明这里很陡
- 梯度方向告诉你损失增长最快的方向
- 我们想要让损失下降，所以要沿着**负梯度方向**更新参数

如果每次都手工推导梯度，会非常痛苦。  
PyTorch 的 `autograd` 就像一个自动记账员：

- 你只管写“怎么算出 loss”
- 它会帮你把梯度链路记录下来
- 你调用 `backward()`，它就自动把梯度算出来

---

## 二、一个最小例子

```python
import torch

# 一个需要学习的参数
w = torch.tensor(2.0, requires_grad=True)

# 定义一个简单函数：loss = (w * 3 - 10)^2
loss = (w * 3 - 10) ** 2

print("loss:", loss.item())

# 自动求导
loss.backward()

print("w 的梯度:", w.grad.item())
```

### 这里发生了什么？

PyTorch 记录了这条计算链：

```text
w -> w*3 -> w*3-10 -> (w*3-10)^2
```

当你执行：

```python
loss.backward()
```

它会沿着这条链，按链式法则把梯度一路传回来，最后得到：

```python
w.grad
```

这就是“当前 `w` 再往前走一点点，loss 会怎么变”的信息。

---

## 三、从梯度到参数更新

有了梯度，就能做一次最简单的梯度下降：

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
lr = 0.1

for step in range(5):
    loss = (w * 3 - 10) ** 2
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad

    print(f"step={step}, w={w.item():.4f}, loss={loss.item():.4f}")

    w.grad.zero_()
```

### 这一段每步在干什么？

| 代码 | 作用 |
|---|---|
| `loss = ...` | 计算当前损失 |
| `loss.backward()` | 求当前损失对 `w` 的梯度 |
| `w -= lr * w.grad` | 用梯度更新参数 |
| `w.grad.zero_()` | 把旧梯度清掉，准备下轮计算 |

---

## 四、为什么要清零梯度？

这是 PyTorch 初学者最容易踩坑的点之一。

PyTorch 默认会**累计梯度**，而不是自动覆盖。

看下面的例子：

```python
import torch

x = torch.tensor(3.0, requires_grad=True)

y1 = x ** 2
y1.backward()
print("第一次 backward 后的梯度:", x.grad.item())

y2 = 2 * x
y2.backward()
print("第二次 backward 后的梯度:", x.grad.item())
```

你会发现第二次梯度不是新的结果，而是“第一次 + 第二次”的和。

这就是为什么训练循环里通常都会写：

```python
optimizer.zero_grad()
```

或者：

```python
tensor.grad.zero_()
```

---

## 五、`requires_grad=True` 到底控制了什么？

只有被标记为 `requires_grad=True` 的张量，PyTorch 才会为它追踪梯度。

```python
import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=False)

y = a * b + 1
y.backward()

print("a.grad:", a.grad.item())
print("b.grad:", b.grad)
```

输出里你会看到：

- `a.grad` 有值
- `b.grad` 是 `None`

这很符合直觉：  
如果某个值不是“需要学习的参数”，就没必要对它求梯度。

---

## 六、`torch.no_grad()` 是干什么的？

训练时我们要记录梯度。  
但推理、评估、参数手动更新时，我们往往**不需要**梯度。

这时就可以用：

```python
with torch.no_grad():
    ...
```

它的作用是：

- 关闭梯度追踪
- 节省内存
- 加快推理

```python
import torch

w = torch.tensor(5.0, requires_grad=True)

with torch.no_grad():
    y = w * 2

print("y.requires_grad:", y.requires_grad)
```

---

## 七、把它放回“模型训练”的语境里

真实训练时，我们通常不是只更新一个数字 `w`，而是更新一整组参数。

比如一个线性模型：

> `y = wx + b`

这里的 `w` 和 `b` 都是参数，都要学习。  
训练时发生的事其实还是一样：

1. 用当前参数做预测
2. 计算预测和真实值之间的损失
3. 自动求出每个参数的梯度
4. 用优化器按梯度方向更新参数

所以自动求导不是“额外功能”，而是深度学习训练的发动机。

---

## 八、一个带两个参数的可运行例子

```python
import torch

# 我们希望模型学到：y = 2x + 1
x = torch.tensor([1.0, 2.0, 3.0, 4.0])
y_true = torch.tensor([3.0, 5.0, 7.0, 9.0])

w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
lr = 0.05

for epoch in range(200):
    y_pred = w * x + b
    loss = ((y_pred - y_true) ** 2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

    w.grad.zero_()
    b.grad.zero_()

    if epoch % 40 == 0:
        print(
            f"epoch={epoch:3d}, loss={loss.item():.4f}, "
            f"w={w.item():.4f}, b={b.item():.4f}"
        )
```

如果一切正常，`w` 会逼近 `2`，`b` 会逼近 `1`。

---

## 九、常见误区

### 1. `backward()` 会自动更新参数

不会。  
`backward()` 只负责**算梯度**，真正更新参数的是你自己写的更新逻辑，或优化器的 `step()`。

### 2. 每轮不清梯度也没关系

不行。  
如果你不清零，梯度会一直累加，训练结果通常会错掉。

### 3. 推理也照样开着梯度

能跑，但浪费。  
评估或部署时，应该尽量包上 `torch.no_grad()`。

---

## 十、小结

这一节最关键的结论只有三句：

1. 梯度告诉我们“参数该往哪改”
2. `backward()` 负责求梯度，不负责更新参数
3. PyTorch 默认累计梯度，所以训练循环里必须清零

理解了自动求导，你就真正踏进了“模型训练”这件事本身。

---

## 练习

1. 把上面 `y = 2x + 1` 的例子改成 `y = 3x - 2`，重新训练一次。
2. 删除 `w.grad.zero_()` 和 `b.grad.zero_()`，观察训练会发生什么。
3. 试着把学习率 `lr` 改成 `0.5` 和 `0.005`，比较收敛速度和稳定性。
