---
title: "1.1 从神经元到多层感知机"
sidebar_position: 1
description: "理解人工神经元、感知机模型和常用激活函数，搭建你的第一个多层感知机（MLP）"
keywords: [神经元, 感知机, 激活函数, ReLU, Sigmoid, Tanh, MLP, 多层感知机]
---

# 从神经元到多层感知机

:::tip 本节定位
深度学习的一切都从**人工神经元**开始。本节从最简单的感知机出发，认识各种激活函数，再组装成多层感知机（MLP）——这是所有神经网络的基础。
:::

## 学习目标

- 理解从生物神经元到人工神经元的映射
- 掌握感知机模型
- 掌握常用激活函数：ReLU、Sigmoid、Tanh 等
- 理解多层感知机（MLP）的结构

---

## 零、先建立一张地图

这一节更适合新人的理解顺序不是“背神经网络名词”，而是先看这条线：

```mermaid
flowchart LR
    A["输入特征 x"] --> B["加权求和 z = x·w + b"]
    B --> C["激活函数 a = f(z)"]
    C --> D["一个神经元输出"]
    D --> E["多个神经元组成一层"]
    E --> F["多层堆起来就是 MLP"]
```

所以你真正要先看懂的是：

- 神经元先做了什么线性计算
- 激活函数为什么必须存在
- 一层和多层到底是怎么长出来的

## 这节和第四阶段最直接的连续性是什么

如果你刚学完第四阶段，可以先把一个神经元理解成：

- 线性回归 / 逻辑回归那种“加权求和”的升级版

也就是说，神经元并不是凭空出现的新对象，它其实是在第四阶段熟悉的线性模型骨架上，多加了一步：

```mermaid
flowchart LR
    A["第四阶段熟悉的部分<br/>z = x·w + b"] --> B["第五阶段新增的一步<br/>a = f(z)"]
    B --> C["多个神经元再堆成层"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style C fill:#e8f5e9,stroke:#2e7d32,color:#333
```

所以这一节真正新增的核心，其实只有两件事：

- 激活函数
- 多层堆叠

## 一、从生物到人工

```mermaid
flowchart LR
    subgraph BIO["生物神经元"]
        D["树突<br/>接收信号"] --> S["细胞体<br/>汇总加工"]
        S --> A["轴突<br/>输出信号"]
    end
    subgraph ART["人工神经元"]
        X["输入 x1, x2, ..."] --> W["加权求和<br/>z = w1·x1 + w2·x2 + b"]
        W --> ACT["激活函数<br/>a = f(z)"]
        ACT --> O["输出"]
    end

    style D fill:#e3f2fd,stroke:#1565c0,color:#333
    style S fill:#e3f2fd,stroke:#1565c0,color:#333
    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style X fill:#fff3e0,stroke:#e65100,color:#333
    style W fill:#fff3e0,stroke:#e65100,color:#333
    style ACT fill:#fff3e0,stroke:#e65100,color:#333
    style O fill:#fff3e0,stroke:#e65100,color:#333
```

核心对应关系：

| 生物 | 人工 |
|------|------|
| 树突（接收信号） | 输入 x |
| 突触强度 | 权重 w |
| 细胞体（汇总） | 加权求和 z = Σ(wi·xi) + b |
| 激活/抑制 | 激活函数 f(z) |
| 轴突（输出） | 输出 a = f(z) |

### 1.1 一个最小“人工神经元”计算例子

新人最容易发虚的地方是：知道公式，但脑子里没有“这一步到底算出了什么”。

先看一个最小例子：

```python
import numpy as np

# 一个样本的 3 个特征
x = np.array([0.8, 0.3, 0.5])

# 一个神经元的 3 个权重
w = np.array([0.2, -0.4, 0.6])
b = 0.1

# 第一步：线性组合
z = np.dot(x, w) + b
print("z =", round(z, 4))

# 第二步：过激活函数
relu_out = max(0, z)
print("ReLU(z) =", round(relu_out, 4))
```

你可以把这一步理解成：

- 权重在表达“每个输入有多重要”
- 偏置在表达“整体阈值往哪边推一点”
- 激活函数决定“这个神经元到底要不要被激活”

### 1.1.1 如果先不谈深度学习，可以把神经元想成什么？

一个很适合新人的理解方式是：

- 把神经元先看成“带一个门”的线性模型

先算：

- `z = x·w + b`

再决定：

- 这个结果要原样通过、压到 `0~1`、还是小于 0 直接截掉

这个“门”就是激活函数。  
所以神经元并不是神秘新物种，而是在“线性打分”外面再加一层非线性选择。

---

## 二、感知机——最简单的人工神经元

### 2.1 模型

感知机是一个做**二分类**的简单模型：

> **z = w1·x1 + w2·x2 + ... + wn·xn + b**
>
> **输出 = 1 如果 z > 0，否则 = 0**

```python
import numpy as np
import matplotlib.pyplot as plt

class Perceptron:
    """最简单的感知机"""
    def __init__(self, n_features, lr=0.1):
        self.w = np.zeros(n_features)
        self.b = 0
        self.lr = lr

    def predict(self, x):
        z = np.dot(x, self.w) + self.b
        return 1 if z > 0 else 0

    def train(self, X, y, epochs=20):
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = yi - pred
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
            if errors == 0:
                print(f"第 {epoch+1} 轮收敛！")
                break

# AND 门
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 0, 0, 1])

p = Perceptron(2)
p.train(X, y)
print(f"权重: {p.w}, 偏置: {p.b}")
for xi, yi in zip(X, y):
    print(f"  输入 {xi} → 预测 {p.predict(xi)}, 真实 {yi}")
```

### 2.2 感知机的局限

感知机只能解决**线性可分**问题。XOR 问题就无法解决——这正是多层网络出现的原因。

```python
# XOR 问题——感知机无法解决
X_xor = np.array([[0,0], [0,1], [1,0], [1,1]])
y_xor = np.array([0, 1, 1, 0])

p_xor = Perceptron(2)
p_xor.train(X_xor, y_xor, epochs=100)

print("\nXOR 预测结果:")
for xi, yi in zip(X_xor, y_xor):
    print(f"  输入 {xi} → 预测 {p_xor.predict(xi)}, 真实 {yi}")
```

### 2.3 感知机这一段最该带走什么？

不是“感知机还值得不用”，而是它帮你看清了一个特别关键的事实：

> **只有线性打分时，模型表达能力会很快遇到边界。**

这正是后面为什么需要：

- 激活函数
- 多层网络

所以感知机最重要的教学价值，是帮你第一次看到“为什么单层不够”。

---

## 三、激活函数

### 3.1 为什么需要激活函数？

如果没有激活函数，多层网络就退化为一个线性模型——无论叠多少层，效果等同于单层。激活函数引入**非线性**，让网络能拟合任意复杂的函数。

### 3.1.1 这句话为什么这么重要？

因为它解释了“深度”为什么不是单纯堆层数。

如果每一层都只是线性变换，那很多层合起来本质上还是一个更大的线性变换。  
真正让多层网络变得有意义的，不是层数本身，而是：

- 每层之间插入了非线性

所以可以先记一个最重要的判断：

- 没有非线性，深度网络就学不出复杂形状

### 3.2 常用激活函数

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 200)

# 各种激活函数
activations = {
    'Sigmoid': (1 / (1 + np.exp(-x)), 'σ(x) = 1/(1+e⁻ˣ)'),
    'Tanh': (np.tanh(x), 'tanh(x)'),
    'ReLU': (np.maximum(0, x), 'max(0, x)'),
    'Leaky ReLU': (np.where(x > 0, x, 0.01 * x), 'max(0.01x, x)'),
}

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
colors = ['#e74c3c', '#3498db', '#2ecc71', '#9b59b6']

for ax, (name, (y, formula)), color in zip(axes.ravel(), activations.items(), colors):
    ax.plot(x, y, linewidth=2, color=color)
    ax.axhline(0, color='gray', linewidth=0.5)
    ax.axvline(0, color='gray', linewidth=0.5)
    ax.set_title(f'{name}: {formula}', fontsize=12)
    ax.set_xlim(-5, 5)
    ax.grid(True, alpha=0.3)

plt.suptitle('常用激活函数', fontsize=14)
plt.tight_layout()
plt.show()
```

### 3.3 对比与选择

| 激活函数 | 输出范围 | 优点 | 缺点 | 使用场景 |
|---------|---------|------|------|---------|
| **ReLU** | [0, +∞) | 计算快、缓解梯度消失 | 神经元"死亡" | **隐藏层首选** |
| **Sigmoid** | (0, 1) | 输出概率解释 | 梯度消失、非零中心 | 二分类输出层 |
| **Tanh** | (-1, 1) | 零中心 | 梯度消失 | RNN（较少用） |
| **Leaky ReLU** | (-∞, +∞) | 避免神经元死亡 | 多一个超参数 | ReLU 改进 |
| **GELU** | 约 (-0.17, +∞) | 平滑、效果好 | 计算稍慢 | Transformer |
| **Swish** | 约 (-0.28, +∞) | 平滑、自门控 | 计算稍慢 | 新架构 |

:::info ReLU 的"神经元死亡"
当输入始终为负时，ReLU 输出永远为 0，梯度也为 0，参数不再更新。Leaky ReLU 通过给负数一个小斜率（0.01）来缓解。
:::

### 3.4 初学阶段怎么选激活函数最不容易乱？

一个足够稳的记法是：

- 隐藏层先默认用 `ReLU`
- 二分类输出层常见 `Sigmoid`
- 多分类输出层常见 `Softmax`
- Transformer 里经常会碰到 `GELU`

先把这四条记住，已经足够支撑你进入大部分后续章节。

### 3.5 第一次看激活函数图，最值得先盯哪几件事？

不要一上来纠结每条曲线的精确公式。先只看这三件事：

1. 输出范围是什么
2. 小于 0 的部分会怎么处理
3. 这条曲线是不是平滑、是不是容易让梯度太小

这三件事会直接决定：

- 输出能不能解释成概率
- 梯度会不会消失
- 训练时会不会比较稳

---

## 四、多层感知机（MLP）

### 4.1 结构

把多个神经元**按层排列**，前一层的输出作为下一层的输入：

```mermaid
flowchart LR
    subgraph INPUT["输入层"]
        I1["x1"]
        I2["x2"]
        I3["x3"]
    end
    subgraph HIDDEN["隐藏层"]
        H1["h1"]
        H2["h2"]
        H3["h3"]
        H4["h4"]
    end
    subgraph OUTPUT["输出层"]
        O1["y1"]
        O2["y2"]
    end

    I1 --> H1 & H2 & H3 & H4
    I2 --> H1 & H2 & H3 & H4
    I3 --> H1 & H2 & H3 & H4
    H1 --> O1 & O2
    H2 --> O1 & O2
    H3 --> O1 & O2
    H4 --> O1 & O2

    style INPUT fill:#e3f2fd,stroke:#1565c0,color:#333
    style HIDDEN fill:#fff3e0,stroke:#e65100,color:#333
    style OUTPUT fill:#e8f5e9,stroke:#2e7d32,color:#333
```

### 4.1.1 多层到底强在哪里？

一个更适合新人的理解方式是：

- 第一层先学一些比较基础的小模式
- 后一层再把这些模式重新组合
- 层数越往后，表示就越抽象

哪怕在最简单的 MLP 里，你也可以把它先粗略理解成：

- 前面层在学“中间表示”
- 最后一层在用这些表示做输出判断

这正是“自动学表示”开始发生的地方。

### 4.2 用 NumPy 实现 MLP 解决 XOR

```python
np.random.seed(42)

# XOR 数据
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

# 网络: 2 → 4 → 1
W1 = np.random.randn(2, 4) * 0.5
b1 = np.zeros((1, 4))
W2 = np.random.randn(4, 1) * 0.5
b2 = np.zeros((1, 1))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_deriv(a):
    return a * (1 - a)

lr = 1.0
losses = []

for epoch in range(5000):
    # 前向传播
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # 损失
    loss = np.mean((y - a2) ** 2)
    losses.append(loss)

    # 反向传播
    dz2 = (a2 - y) * sigmoid_deriv(a2)
    dW2 = a1.T @ dz2 / 4
    db2 = np.mean(dz2, axis=0, keepdims=True)

    dz1 = (dz2 @ W2.T) * sigmoid_deriv(a1)
    dW1 = X.T @ dz1 / 4
    db1 = np.mean(dz1, axis=0, keepdims=True)

    # 更新
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

print(f"最终损失: {losses[-1]:.6f}")
print("XOR 预测:")
for xi, yi, pred in zip(X, y, a2):
    print(f"  {xi} → {pred[0]:.4f}, 真实 {yi[0]}")

plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLP 解决 XOR')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 五、小结

| 概念 | 要点 |
|------|------|
| 人工神经元 | 加权求和 + 激活函数 |
| 感知机 | 最简单的神经元，只能线性分类 |
| 激活函数 | 引入非线性；隐藏层用 ReLU |
| MLP | 多层堆叠，可拟合任意函数 |

### 5.1 这节最该带走什么

如果只带走一句话，我希望你记住：

> **神经网络的起点不是“很多层”，而是“在线性计算后加上非线性，再把这种结构不断堆起来”。**

所以这一节真正要稳住的是：

- 神经元先算线性，再过激活
- 感知机的边界会逼出多层网络
- 激活函数决定网络有没有真正的非线性表达能力
- MLP 是后面很多复杂结构的最小原型

```mermaid
mindmap
  root((神经网络基础))
    人工神经元
      加权求和
      偏置
      激活函数
    激活函数
      ReLU 🔧首选
      Sigmoid 输出层
      Tanh 零中心
      GELU Transformer
    MLP
      输入层
      隐藏层
      输出层
      能解决 XOR
```

---

## 动手练习

### 练习 1：实现 OR 门感知机

修改 AND 门的训练数据为 OR 门（0|0→0, 0|1→1, 1|0→1, 1|1→1），训练感知机并画出决策边界。

### 练习 2：MLP 分类月牙数据

用 `sklearn.datasets.make_moons` 生成月牙数据，手写 NumPy MLP（2→8→1），训练后画出决策边界。
