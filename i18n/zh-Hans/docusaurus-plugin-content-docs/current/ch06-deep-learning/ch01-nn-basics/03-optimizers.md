---
title: "6.1.5 优化器"
sidebar_position: 5
description: "一节跟着操作的优化器课程：SGD、Momentum、Adam、学习率敏感性、过冲和实用优化器选择"
keywords: [优化器, SGD, momentum, Adam, learning rate, PyTorch, 梯度下降]
---

# 6.1.5 优化器

![优化器路径对比图](/img/course/optimizer-comparison.webp)

:::tip 本节概览
梯度算出来以后，优化器决定参数怎么移动。优化器名字很重要，但学习率往往更重要。
:::

## 你会做出什么

这一节会运行一个很小的 PyTorch 优化实验：

- 在同一个简单 loss 上比较 SGD、Momentum 和 Adam；
- 直接看到过冲；
- 测试学习率敏感性；
- 学会安全的优化器选择顺序。

![梯度到参数更新的优化器决策图](/img/course/ch06-optimizer-gradient-to-update-map.webp)

## 环境准备

```bash
python -m pip install -U torch
```

## 运行完整实验

新建 `optimizer_lab.py`：

```python
import torch


def run_optimizer(name, optimizer_factory, steps=25):
    torch.manual_seed(42)
    w = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer = optimizer_factory([w])
    for step in range(1, steps + 1):
        loss = (w - 2).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step in [1, 5, 10, 25]:
            print(f"{name:<8} step={step:<2} w={w.item():.3f} loss={loss.item():.4f}")


print("optimizer_comparison")
run_optimizer("sgd", lambda params: torch.optim.SGD(params, lr=0.1))
run_optimizer("momentum", lambda params: torch.optim.SGD(params, lr=0.1, momentum=0.9))
run_optimizer("adam", lambda params: torch.optim.Adam(params, lr=0.1))

print("learning_rate_check")
for lr in [0.01, 0.1, 1.1]:
    torch.manual_seed(42)
    w = torch.nn.Parameter(torch.tensor([5.0]))
    optimizer = torch.optim.SGD([w], lr=lr)
    for _ in range(10):
        loss = (w - 2).pow(2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    final_loss = (w - 2).pow(2).item()
    print(f"lr={lr:<4} final_w={w.item():.3f} final_loss={final_loss:.4f}")
```

运行：

```bash
python optimizer_lab.py
```

预期输出：

```text
optimizer_comparison
sgd      step=1  w=4.400 loss=9.0000
sgd      step=5  w=2.983 loss=1.5099
sgd      step=10 w=2.322 loss=0.1621
sgd      step=25 w=2.011 loss=0.0002
momentum step=1  w=4.400 loss=9.0000
momentum step=5  w=0.259 loss=0.8571
momentum step=10 w=2.013 loss=0.6767
momentum step=25 w=2.475 loss=0.0200
adam     step=1  w=4.900 loss=9.0000
adam     step=5  w=4.502 loss=6.7648
adam     step=10 w=4.014 loss=4.4535
adam     step=25 w=2.739 loss=0.6569
learning_rate_check
lr=0.01 final_w=4.451 final_loss=6.0085
lr=0.1  final_w=2.322 final_loss=0.1038
lr=1.1  final_w=20.575 final_loss=345.0386
```

![优化器实验结果图](/img/course/ch06-optimizer-lr-result-dashboard.webp)

## 读懂实验

这里的 loss 是：

```text
loss = (w - 2)^2
```

最佳值是 `w=2`。所有优化器都从 `w=5` 出发。

在这个简单例子里，学习率合适的 SGD 表现非常好：

```text
sgd step=25 w=2.011 loss=0.0002
```

Momentum 移动更快，但可能过冲：

```text
momentum step=5 w=0.259
```

Adam 是深度学习里很常见的默认选择，但不是魔法。在这个小问题上，`lr=0.1` 的 Adam 反而比调好的 SGD 慢。重点不是“Adam 不好”，而是：

> 一定要看训练行为。优化器选择和学习率是一起工作的。

## 学习率是第一个旋钮

学习率实验故意很直接：

```text
lr=0.01 final_w=4.451 final_loss=6.0085
lr=0.1  final_w=2.322 final_loss=0.1038
lr=1.1  final_w=20.575 final_loss=345.0386
```

太小：训练很慢。

合适：逐渐接近最优点。

太大：训练发散。

## 留下的证据

在笔记里保留一张 optimizer 对比表：

```text
same_loss: (w - 2)^2
same_start: w = 5
sgd_result: approaches w = 2 with lr=0.1
momentum_result: moves faster but overshoots
bad_lr_result: lr=1.1 diverges
```

这比背 optimizer 名字更有用。它说明真正的规则：梯度给方向，optimizer 设置决定移动的幅度和方式。

## 优化器直觉

| 优化器 | 直觉 | 适合先用在 |
|---|---|---|
| SGD | 直接沿负梯度方向移动 | 简单基线、受控实验 |
| SGD + Momentum | 保留之前步骤的速度 | 噪声方向上更平滑 |
| Adam | 根据梯度历史自适应步长 | 很多神经网络的强默认选择 |

真实神经网络里，Adam 或 AdamW 经常是实用起点。最终训练仍然要看任务验证指标。

## 实用选择顺序

1. 神经网络基线先用 Adam 或 AdamW。
2. 先调学习率，再争论优化器名字。
3. 观察训练 loss 和验证 loss 曲线。
4. 如果验证不稳定，降低 LR 或加入 schedule。
5. 如果训练慢但稳定，尝试 LR schedule 或更换优化器。

## 常见排查清单

| 现象 | 可能原因 | 修复方式 |
|---|---|---|
| loss 爆炸 | 学习率太高 | 降低 LR |
| loss 下降很慢 | LR 太低或输入尺度差 | 谨慎提高 LR，归一化输入 |
| training loss 降低但 validation 变差 | 过拟合 | 正则化、加数据、早停 |
| loss 来回震荡 | momentum/LR 太激进 | 降低 LR 或 momentum |
| Adam 能跑但最终效果弱 | 优化器掩盖了其他问题 | 检查数据、结构、正则化 |

## 练习

1. 把 SGD 学习率改成 `0.05`、`0.2`、`0.8`。
2. 把 momentum 从 `0.9` 改成 `0.5`。过冲是否减少？
3. 用 `AdamW` 替换 `Adam`。
4. 每一步打印 `w.grad`，把梯度和更新联系起来。
5. 为每个优化器画出 `w` 随 step 的变化。

<details>
<summary>参考答案与讲解</summary>

1. 较小学习率会移动更慢；合适的学习率会更快收敛；过大的学习率可能来回震荡甚至发散。
2. 降低 momentum 通常会减少越过最优区域的现象，但也可能减少有用的加速。要同时看 `w` 的路径和最终 loss。
3. 在这个玩具问题上，`AdamW` 可能和 Adam 很像；它的关键区别是 weight decay 与自适应更新解耦，这在大模型里更重要。
4. `w.grad` 指向让 loss 增大的方向，所以 optimizer 通常会沿反方向更新参数。实际步长还取决于 optimizer 状态和学习率。
5. `w` 的轨迹图能看出 optimizer 是爬得太慢、顺滑接近最优、越过最优，还是来回震荡；这比只看最后一个数字可靠。

</details>

## 过关检查

你能解释下面几点，就完成本节：

- 梯度说明哪个方向会改变 loss；
- 优化器决定参数走多远；
- 学习率会让训练变慢、收敛或发散；
- momentum 可以加速，但也可能过冲；
- Adam 有用，但不能替代观察训练曲线。
