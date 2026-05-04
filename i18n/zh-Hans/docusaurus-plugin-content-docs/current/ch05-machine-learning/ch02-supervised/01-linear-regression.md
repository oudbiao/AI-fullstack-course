---
title: "2.2 线性回归"
sidebar_position: 3
description: "从简单线性回归到多元线性回归，理解最小二乘法、梯度下降求解、多项式回归、正则化（Ridge / Lasso / Elastic Net）"
keywords: [线性回归, 最小二乘法, 梯度下降, 正则化, Ridge, Lasso, Elastic Net, 多项式回归]
---

# 线性回归

![线性回归拟合与损失曲面图](/img/course/linear-regression-loss-landscape.png)

:::tip 本节定位
线性回归是**最简单也最重要**的机器学习算法。它是理解所有后续算法的基石——逻辑回归、神经网络、甚至 GPT 的底层都能看到它的影子。
:::

## 学习目标

- 理解简单线性回归与多元线性回归
- 掌握最小二乘法与正规方程
- 理解梯度下降法求解（与第 4 站衔接）
- 掌握多项式回归与过拟合
- 理解正则化（Ridge、Lasso、Elastic Net）

## 先说一个很重要的学习预期

这一节很长，也很容易让新人误以为自己要一次学会很多事：

- 回归任务
- 损失函数
- 正规方程
- 梯度下降
- 多项式回归
- 正则化

但更适合新人的第一目标其实只有一个：

> **先把“一个机器学习模型到底是怎么从任务定义，走到损失、求解、评估和改进”的主线看顺。**

如果这条主线立住了，后面逻辑回归、神经网络和很多更复杂的模型都会顺很多。

---

## 先建立一张地图

第一次学线性回归，最容易出现两种情况：

- 会把公式写出来，但不知道每一步到底在解决什么问题
- 会调用 `LinearRegression()`，但不知道为什么要先做 baseline、为什么要看残差、为什么正则化能防过拟合

更稳的理解顺序是：

![线性回归学习主线图](/img/course/ch05-linear-regression-learning-flow.png)

![线性回归直觉漫画](/img/course/ch05-linear-regression-intuition-comic.png)

先看这张漫画再看公式：直线是一把可以调的尺子，残差是数据点到直线的垂直距离，MSE 把这些距离变成训练目标，正则化则像刹车，防止复杂模型过度弯曲。

如果你先抓住这条线，后面所有公式都会更容易落到一个明确的问题上。

:::tip 先准备运行环境
本节会用到 `numpy`、`matplotlib`、`pandas` 和 `scikit-learn`。如果你的环境是新建的，请先在项目根目录执行 `python -m pip install -r requirements-course-core.txt`。`pandas` 是下面多元回归示例里用来处理表格数据的库。
:::

---

## 一、简单线性回归

### 1.1 直觉：找一条"最佳拟合线"

**问题**：已知房屋面积和价格的数据，能否预测一个新面积对应的价格？

```python
import numpy as np
import matplotlib.pyplot as plt

# 模拟数据：面积 → 价格
rng = np.random.default_rng(seed=42)
X = rng.uniform(50, 200, 30)    # 面积（平方米）
y = 2.5 * X + 50 + rng.normal(size=30) * 30  # 价格（万元）

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='steelblue', s=50, alpha=0.7)
plt.xlabel('面积（平方米）')
plt.ylabel('价格（万元）')
plt.title('房屋面积 vs 价格')
plt.grid(True, alpha=0.3)
plt.show()
```

**目标**：找一条直线 `y = wx + b`，让这条线尽可能"贴近"所有的数据点。

- **w**（weight）= 斜率 = 面积每增加 1 平方米，价格增加多少
- **b**（bias）= 截距 = 面积为 0 时的基础价格

### 1.1.1 这两个参数到底在控制什么？

可以把这条直线想成一根可以拖动和旋转的木棍：

- `w` 决定木棍的倾斜程度
- `b` 决定木棍整体向上还是向下平移

所以线性回归训练的本质，其实就是不断调整这两个量，让整条线尽量贴近数据点。

### 1.1.2 一个更适合新人的类比

你可以先把线性回归想成：

- 你手里拿着一把可以旋转、也可以上下平移的尺子
- 你想让这把尺子尽量贴近眼前这堆点

这里最值得先记住的不是公式，而是：

- `w` 决定尺子斜不斜
- `b` 决定尺子整体往上还是往下
- 训练过程就是在反复微调这两个旋钮

:::info 先记一句最重要的话
线性回归不是在“背公式”，而是在做一件很朴素的事：**找一条规则，让输入变化和输出变化之间的关系尽量稳定地被描述出来。**
:::

### 1.2 什么是"最佳"？——损失函数

"贴近"需要数学定义。我们用**均方误差（MSE）**来衡量：

> **MSE = (1/n) × Σ(yi - ŷi)²**

其中 `ŷi = w×xi + b` 是模型的预测值。

**直觉**：每个数据点的预测误差取平方，然后求平均。MSE 越小，拟合越好。

### 1.3 为什么误差通常先平方？

这一步很重要，因为它决定了我们到底在惩罚什么。

- 平方后永远是正数，正负误差不会互相抵消
- 大误差会被放大，所以模型会更努力修正那些偏得很离谱的点
- 平方形式求导后很干净，方便得到解析解和梯度下降公式

但它也带来一个副作用：

- 如果数据里有极端异常值，MSE 会被它们强烈主导

所以新人第一次做回归项目时，可以先这样记：

- 默认先用 `MSE / RMSE`
- 如果你怀疑异常值很多，再去看 `MAE`

### 1.3.1 回归指标关键词先拆开

| 术语 | 它是什么意思 | 为什么重要 |
|---|---|---|
| `baseline` | 第一个简单对照模型 | 没有起点，就很难判断后续模型是否真的变好 |
| `residual` | `真实值 - 预测值`，残差 | 残差图能暴露单个分数看不到的模式 |
| `MSE` | Mean Squared Error，均方误差 | 强烈惩罚大误差，常用作优化目标 |
| `RMSE` | Root Mean Squared Error，均方根误差 | MSE 开方后回到目标值原单位 |
| `MAE` | Mean Absolute Error，平均绝对误差 | 不希望异常值过度主导时更稳 |
| `R²` | 模型解释目标波动的比例 | 适合快速概览拟合质量，但不能替代诊断 |

```python
def mse_loss(y_true, y_pred):
    """均方误差"""
    return np.mean((y_true - y_pred) ** 2)

# 试几条不同的线
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
params = [(1.0, 100, '斜率太小'), (2.5, 50, '刚刚好'), (4.0, -50, '斜率太大')]

for ax, (w, b, title) in zip(axes, params):
    y_pred = w * X + b
    loss = mse_loss(y, y_pred)
    ax.scatter(X, y, color='steelblue', s=30, alpha=0.7)
    x_line = np.linspace(40, 210, 100)
    ax.plot(x_line, w * x_line + b, 'r-', linewidth=2)
    ax.set_title(f'{title}\nw={w}, b={b}, MSE={loss:.0f}')
    ax.set_xlabel('面积')
    ax.set_ylabel('价格')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

这个例子中，中间那条线通常 MSE 最小：

```text
斜率太小: MSE ≈ 28463
刚刚好: MSE ≈ 575
斜率太大: MSE ≈ 14502
```

---

## 二、求解方法一：正规方程（解析解）

### 2.1 公式

对于线性回归，MSE 有**闭合公式解**：

> **w = (Xᵀ X)⁻¹ Xᵀ y**

这就是**正规方程（Normal Equation）**。

### 2.2 手动实现

```python
# 准备数据（增加截距列）
X_b = np.c_[np.ones(len(X)), X]  # 在 X 前面加一列 1（对应截距 b）
print(f"X_b 形状: {X_b.shape}")  # (30, 2)

# 正规方程求解
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
b_fit, w_fit = w[0], w[1]
print(f"截距 b = {b_fit:.2f}")
print(f"斜率 w = {w_fit:.2f}")

# 可视化
plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='steelblue', s=50, alpha=0.7, label='数据点')
x_line = np.linspace(40, 210, 100)
plt.plot(x_line, w_fit * x_line + b_fit, 'r-', linewidth=2,
         label=f'拟合线: y = {w_fit:.2f}x + {b_fit:.2f}')
plt.xlabel('面积（平方米）')
plt.ylabel('价格（万元）')
plt.title('正规方程求解线性回归')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

预期输出：

```text
X_b 形状: (30, 2)
截距 b = 57.36
斜率 w = 2.47
```

### 2.3 正规方程的优缺点

| 优点 | 缺点 |
|------|------|
| 直接算出结果，不需要迭代 | 需要计算矩阵逆，复杂度 O(n³) |
| 不需要调学习率 | 特征数量大时非常慢 |
| 一定能找到全局最优 | 特征数 > 样本数时无法使用 |

### 2.4 什么时候优先想到正规方程？

可以把正规方程理解成“直接算答案”的路线。它最适合：

- 特征数不多
- 数据规模不大
- 你想先快速验证线性关系是否存在

第一次做一个小型回归 baseline 时，正规方程或者 `sklearn` 的线性回归通常都很合适。
但如果你进入下面这些场景，就要开始更自然地切到梯度下降思维：

- 特征维度很多
- 数据规模明显变大
- 后面还要接神经网络训练
- 你已经不只是想“算一个答案”，而是想“进入一套统一训练框架”

---

## 三、求解方法二：梯度下降

### 3.1 与第 4 站的衔接

在第 4 站微积分章节，你已经学过梯度下降的原理。现在把它应用到线性回归：

```mermaid
flowchart LR
    A["随机初始化 w, b"] --> B["计算预测值 ŷ = wX + b"]
    B --> C["计算损失 MSE"]
    C --> D["计算梯度 ∂MSE/∂w, ∂MSE/∂b"]
    D --> E["更新参数<br/>w = w - lr × ∂MSE/∂w<br/>b = b - lr × ∂MSE/∂b"]
    E --> F{"收敛了？"}
    F -->|"否"| B
    F -->|"是"| G["输出 w, b"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style G fill:#e8f5e9,stroke:#2e7d32,color:#333
```

### 3.1.1 先别急着背梯度公式，先看它在表达什么

梯度下降最核心的意思不是“公式很多”，而是：

- 先用当前参数做一次预测
- 看看预测和真实值差多少
- 判断参数应该往哪个方向调
- 每次调一点点
- 重复很多次

如果你把这五步想明白，后面的 `dw`、`db` 就不再只是符号，而是在表达：

- `dw` 告诉我们“斜率该往哪边调”
- `db` 告诉我们“整条线该往上还是往下移”

### 3.2 梯度推导

MSE 对 w 和 b 的梯度：

> **∂MSE/∂w = -(2/n) × Σ xi(yi - ŷi)**
>
> **∂MSE/∂b = -(2/n) × Σ (yi - ŷi)**

### 3.3 从零实现

```python
# 梯度下降求解线性回归

# 参数初始化
w_gd = 0.0
b_gd = 0.0
lr = 0.00005   # 学习率（注意：特征值较大时学习率要小）
epochs = 500

# 记录训练过程
history = {'loss': [], 'w': [], 'b': []}

for epoch in range(epochs):
    # 前向：预测
    y_pred = w_gd * X + b_gd

    # 计算损失
    loss = mse_loss(y, y_pred)

    # 计算梯度
    dw = -2 * np.mean(X * (y - y_pred))
    db = -2 * np.mean(y - y_pred)

    # 更新参数
    w_gd -= lr * dw
    b_gd -= lr * db

    history['loss'].append(loss)
    history['w'].append(w_gd)
    history['b'].append(b_gd)

print(f"梯度下降结果: w = {w_gd:.2f}, b = {b_gd:.2f}")
print(f"正规方程结果: w = {w_fit:.2f}, b = {b_fit:.2f}")

# 可视化训练过程
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(history['loss'])
axes[0].set_title('损失曲线')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE')
axes[0].set_yscale('log')

axes[1].plot(history['w'], label='w')
axes[1].axhline(y=w_fit, color='r', linestyle='--', label=f'最优 w={w_fit:.2f}')
axes[1].set_title('w 的收敛过程')
axes[1].legend()

axes[2].plot(history['b'], label='b')
axes[2].axhline(y=b_fit, color='r', linestyle='--', label=f'最优 b={b_fit:.2f}')
axes[2].set_title('b 的收敛过程')
axes[2].legend()

for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

预期输出：

```text
梯度下降结果: w = 2.86, b = 0.29
正规方程结果: w = 2.47, b = 57.36
```

注意，这个简单梯度下降示例还没有完全追上正规方程结果，因为 `X` 的数值较大，而学习率故意设置得很小。这样做是为了教学安全：先看清更新循环，避免数值直接发散。真实项目中，做梯度下降前通常要先标准化特征。

---

## 四、多元线性回归

### 4.1 从一个特征到多个特征

实际问题中，房价不只取决于面积，还取决于房间数、楼层、距地铁距离等。

> **ŷ = w₁x₁ + w₂x₂ + ... + wpxp + b = wᵀx + b**

### 4.1.1 多元线性回归这里真正升级了什么？

很多新人第一次看到多元线性回归，会误以为“模型突然变复杂了很多”。
其实它本质上没有换问题，只是把：

- “一个输入决定输出”

变成了：

- “多个输入一起对输出做线性贡献”

真正升级的地方只有两点：

- 参数从一个 `w` 变成一组 `w1, w2, ..., wp`
- 我们开始必须认真处理特征尺度、共线性和特征解释

所以多元线性回归不是另一种算法，而是线性回归进入真实数据场景后的自然形态。

### 4.2 用 Scikit-learn 实现

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 模拟多特征房价数据
rng = np.random.default_rng(seed=42)
n = 200
data = pd.DataFrame({
    '面积': rng.uniform(50, 200, n),
    '房间数': rng.integers(1, 6, n),
    '楼层': rng.integers(1, 30, n),
    '距地铁(km)': rng.uniform(0.1, 5, n),
})
# 真实关系 + 噪声
data['价格'] = (2.5 * data['面积']
               + 30 * data['房间数']
               + 2 * data['楼层']
               - 20 * data['距地铁(km)']
               + 50
               + rng.normal(size=n) * 30)

print(data.head())
print(f"\n数据形状: {data.shape}")

# 准备数据
X = data[['面积', '房间数', '楼层', '距地铁(km)']].values
y = data['价格'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LinearRegression()
model.fit(X_train, y_train)

# 查看学到的参数
print("\n模型参数:")
for name, coef in zip(['面积', '房间数', '楼层', '距地铁(km)'], model.coef_):
    print(f"  {name}: {coef:.2f}")
print(f"  截距: {model.intercept_:.2f}")

# 评估
y_pred = model.predict(X_test)
print(f"\nMSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")
```

预期输出大致如下：

```text
数据形状: (200, 5)

模型参数:
  面积: 2.52
  房间数: 31.38
  楼层: 1.78
  距地铁(km): -21.83
  截距: 48.11

MSE: 860.36
R² Score: 0.9328
```

### 4.3 R² 分数

R² 是回归模型最常用的评估指标：

> **R² = 1 - Σ(yi - ŷi)² / Σ(yi - ȳ)²**

| R² 值 | 含义 |
|-------|------|
| 1.0 | 完美拟合 |
| 0.8~1.0 | 模型很好 |
| 0.5~0.8 | 模型一般 |
| < 0.5 | 模型较差 |
| < 0 | 还不如直接用平均值预测 |

```python
# 可视化预测 vs 实际
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6, s=30, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='完美预测线')
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title(f'预测 vs 实际 (R² = {r2_score(y_test, y_pred):.4f})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.tight_layout()
plt.show()
```

---

## 五、多项式回归——当数据不是直线

### 5.1 问题：直线不够用

```python
# 生成非线性数据
rng = np.random.default_rng(seed=42)
X_nl = np.linspace(-3, 3, 50)
y_nl = 0.5 * X_nl**2 - X_nl + 2 + rng.normal(size=50) * 0.8

# 线性回归强行拟合
lr = LinearRegression()
lr.fit(X_nl.reshape(-1, 1), y_nl)
y_pred_linear = lr.predict(X_nl.reshape(-1, 1))

plt.figure(figsize=(8, 5))
plt.scatter(X_nl, y_nl, color='steelblue', s=30, alpha=0.7)
plt.plot(X_nl, y_pred_linear, 'r-', linewidth=2, label='线性回归（欠拟合）')
plt.title('直线无法拟合曲线数据')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 5.2 多项式回归

**思路**：把原始特征 `x` 扩展为 `[x, x², x³, ...]`，然后依然用线性回归。

```python
from sklearn.preprocessing import PolynomialFeatures

# 创建多项式特征
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_nl.reshape(-1, 1))
print(f"原始特征: {X_nl[:3]}")
print(f"多项式特征:\n{X_poly[:3]}")  # [x, x²]

# 用线性回归拟合多项式特征
lr_poly = LinearRegression()
lr_poly.fit(X_poly, y_nl)
y_pred_poly = lr_poly.predict(X_poly)

plt.figure(figsize=(8, 5))
plt.scatter(X_nl, y_nl, color='steelblue', s=30, alpha=0.7)
plt.plot(X_nl, y_pred_linear, 'r--', linewidth=2, label='线性回归')
plt.plot(X_nl, y_pred_poly, 'g-', linewidth=2, label='多项式回归 (degree=2)')
plt.title('多项式回归可以拟合曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

特征扩展的预期输出：

```text
原始特征: [-3.         -2.87755102 -2.75510204]
多项式特征:
[[-3.          9.        ]
 [-2.87755102  8.28029988]
 [-2.75510204  7.59058726]]
```

### 5.3 多项式阶数与过拟合

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 9))

degrees = [1, 2, 3, 5, 10, 18]
x_smooth = np.linspace(-3.2, 3.2, 200)

for ax, deg in zip(axes.ravel(), degrees):
    poly = PolynomialFeatures(degree=deg, include_bias=False)
    X_p = poly.fit_transform(X_nl.reshape(-1, 1))
    X_s = poly.transform(x_smooth.reshape(-1, 1))

    lr = LinearRegression()
    lr.fit(X_p, y_nl)

    y_s = lr.predict(X_s)
    y_s = np.clip(y_s, -10, 20)  # 防止极端值

    train_score = lr.score(X_p, y_nl)

    ax.scatter(X_nl, y_nl, color='steelblue', s=20, alpha=0.6)
    ax.plot(x_smooth, y_s, 'r-', linewidth=2)
    ax.set_title(f'degree = {deg}, R² = {train_score:.3f}')
    ax.set_ylim(-5, 15)
    ax.grid(True, alpha=0.3)

plt.suptitle('多项式阶数与过拟合', fontsize=14, y=1.02)
plt.tight_layout()
plt.show()
```

:::warning 过拟合警告
degree 越高，训练集 R² 越接近 1，但模型在新数据上可能表现很差。这就是**过拟合**。解决方案：正则化。
:::

### 5.4 多项式回归最容易学偏的点

新人第一次学到这里，最容易误会成：

- “只要曲线拟合得更漂亮，模型就更好”

但真正该问的是：

- 训练集变好后，测试集有没有一起变好？
- 模型复杂度升高后，泛化能力有没有下降？

所以多项式回归最有教学价值的地方，不是“它能画出更弯的线”，而是它第一次非常直观地让你看到：

- 模型太简单会欠拟合
- 模型太复杂会过拟合
- 模型效果不是只看训练集

---

## 六、正则化——防止过拟合

### 6.1 正则化的思想

正则化 = 在损失函数中加一个**惩罚项**，惩罚过大的参数值。

```mermaid
flowchart LR
    A["原始损失<br/>MSE"] --> C["总损失"]
    B["惩罚项<br/>参数大小"] --> C
    C --> D["优化目标：<br/>既拟合数据，<br/>又不让参数太大"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style B fill:#fff3e0,stroke:#e65100,color:#333
    style D fill:#e8f5e9,stroke:#2e7d32,color:#333
```

**直觉**：惩罚大的参数 → 模型更简单 → 减少过拟合。

对带正则化的线性模型来说，特征尺度非常重要。如果一个特征数值特别大，另一个特征数值特别小，惩罚项就不会公平地看待它们。所以后面的代码使用 `Pipeline(PolynomialFeatures -> StandardScaler -> Ridge/Lasso/ElasticNet)`。

### 6.2 三种正则化

| 方法 | 惩罚项 | 效果 |
|------|--------|------|
| **Ridge（L2）** | `α × Σ(wi²)` | 参数缩小但不为零 |
| **Lasso（L1）** | `α × Σ|wi|` | 部分参数变为零（特征选择） |
| **Elastic Net** | L1 + L2 混合 | 兼具两者优点 |

### 6.3 Ridge 回归（L2 正则化）

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

# 用高阶多项式 + Ridge 对比
X_base = X_nl.reshape(-1, 1)
X_smooth_base = x_smooth.reshape(-1, 1)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
alphas = [0, 0.1, 10]
titles = ['无正则化 (α=0)', 'Ridge α=0.1', 'Ridge α=10']

for ax, alpha, title in zip(axes, alphas, titles):
    if alpha == 0:
        model = make_pipeline(
            PolynomialFeatures(degree=10, include_bias=False),
            StandardScaler(),
            LinearRegression()
        )
    else:
        model = make_pipeline(
            PolynomialFeatures(degree=10, include_bias=False),
            StandardScaler(),
            Ridge(alpha=alpha)
        )

    model.fit(X_base, y_nl)
    y_s = np.clip(model.predict(X_smooth_base), -10, 20)

    ax.scatter(X_nl, y_nl, color='steelblue', s=20, alpha=0.6)
    ax.plot(x_smooth, y_s, 'r-', linewidth=2)
    ax.set_title(title)
    ax.set_ylim(-5, 15)
    ax.grid(True, alpha=0.3)

plt.suptitle('Ridge 正则化的效果（degree=10）', fontsize=13)
plt.tight_layout()
plt.show()
```

### 6.4 Lasso 回归（L1 正则化）——自动特征选择

```python
from sklearn.linear_model import Lasso

# Lasso 能让部分参数变为零 → 自动特征选择
ridge = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    Ridge(alpha=1.0)
)
ridge.fit(X_base, y_nl)

lasso = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    Lasso(alpha=0.1, max_iter=20000)
)
lasso.fit(X_base, y_nl)

ridge_coef = ridge.named_steps["ridge"].coef_
lasso_coef = lasso.named_steps["lasso"].coef_

# 可视化参数
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(range(len(ridge_coef)), np.abs(ridge_coef), color='steelblue')
axes[0].set_title('Ridge 参数（全部非零）')
axes[0].set_xlabel('特征序号')
axes[0].set_ylabel('|参数值|')

axes[1].bar(range(len(lasso_coef)), np.abs(lasso_coef), color='coral')
axes[1].set_title('Lasso 参数（部分为零 → 特征选择）')
axes[1].set_xlabel('特征序号')
axes[1].set_ylabel('|参数值|')

for ax in axes:
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# 看看 Lasso 保留了哪些特征
print("Lasso 参数:", np.round(lasso_coef, 4))
print(f"非零参数个数: {np.sum(lasso_coef != 0)} / {len(lasso_coef)}")
```

预期输出：

```text
Lasso 参数: [-1.5605  1.1533 -0.      0.1151 -0.      0.     -0.      0.     -0.      0.    ]
非零参数个数: 3 / 10
```

### 6.5 Elastic Net

```python
from sklearn.linear_model import ElasticNet

# Elastic Net = L1 + L2 的混合
# l1_ratio 控制 L1 和 L2 的比例（1.0 = 纯 Lasso，0.0 = 纯 Ridge）
en = make_pipeline(
    PolynomialFeatures(degree=10, include_bias=False),
    StandardScaler(),
    ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=20000)
)
en.fit(X_base, y_nl)
en_coef = en.named_steps["elasticnet"].coef_

print("Elastic Net 参数:", np.round(en_coef, 4))
print(f"非零参数个数: {np.sum(en_coef != 0)} / {len(en_coef)}")
```

预期输出：

```text
Elastic Net 参数: [-1.3582  0.8776 -0.2011  0.3548 -0.      0.0579 -0.      0.     -0.      0.    ]
非零参数个数: 5 / 10
```

### 6.6 正则化对比总结

| | Ridge（L2） | Lasso（L1） | Elastic Net |
|---|-----------|-----------|-------------|
| 惩罚项 | `α × Σ(wi²)` | `α × Σ\|wi\|` | 两者的加权和 |
| 参数效果 | 缩小但不为零 | 部分为零 | 部分为零 |
| 适用场景 | 所有特征都有用 | 有很多无用特征 | 特征多且有相关性 |
| sklearn 类 | `Ridge` | `Lasso` | `ElasticNet` |

---

## 七、完整实战：糖尿病数据集

```python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(f"数据集: {X.shape[0]} 样本, {X.shape[1]} 特征")
print(f"特征名: {diabetes.feature_names}")

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 对比多个模型
models = {
    "线性回归": make_pipeline(StandardScaler(), LinearRegression()),
    "Ridge α=1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    "Ridge α=10": make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
    "Lasso α=0.1": make_pipeline(StandardScaler(), Lasso(alpha=0.1, max_iter=10000)),
    "Lasso α=1": make_pipeline(StandardScaler(), Lasso(alpha=1.0, max_iter=10000)),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    results[name] = {'MSE': mse, 'R²': r2}
    print(f"{name:15s} | MSE: {mse:.1f} | R²: {r2:.4f}")

# 可视化 R² 对比
fig, ax = plt.subplots(figsize=(10, 5))
names = list(results.keys())
r2_scores = [v['R²'] for v in results.values()]
colors = ['steelblue', 'coral', 'coral', 'seagreen', 'seagreen']
bars = ax.bar(names, r2_scores, color=colors, alpha=0.8)

for bar, score in zip(bars, r2_scores):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f'{score:.4f}', ha='center', fontsize=10)

ax.set_ylabel('R² Score')
ax.set_title('不同正则化方法对比（糖尿病数据集）')
ax.set_ylim(0, 0.6)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()
```

终端示例输出：

```text
数据集: 442 样本, 10 特征
特征名: ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
线性回归        | MSE: 2900.2 | R²: 0.4526
Ridge α=1     | MSE: 2892.0 | R²: 0.4541
Ridge α=10    | MSE: 2875.8 | R²: 0.4572
Lasso α=0.1   | MSE: 2884.6 | R²: 0.4555
Lasso α=1     | MSE: 2824.6 | R²: 0.4669
```

### 7.1 做完这节，下一步该怎么走？

如果你刚学完线性回归，最稳的下一步不是立刻追更复杂的模型，而是先把下面这条最小建模链练熟：

1. 明确目标是不是连续值预测
2. 先做一个线性回归 baseline
3. 看训练集和测试集误差
4. 看残差图和 `R²`
5. 如果欠拟合，再考虑特征构造或更复杂模型
6. 如果过拟合，再考虑正则化、交叉验证和调参

这条链，其实就是后面整套机器学习项目的雏形。

![线性回归残差诊断图](/img/course/ch05-linear-regression-residual-diagnostics.png)

读这张图时，不要只盯 `R²`。先看残差是不是随机散开；如果残差呈弯曲形状，说明模型可能欠拟合；如果少数点误差特别大，先检查异常值；如果误差随预测值变大而变大，可能要考虑变换目标或换指标。

### 7.2 如果你第一次做回归题，最稳的默认顺序

第一次做回归题时，不建议一上来就比较很多模型。
更稳的顺序通常是：

1. 先确认目标是不是连续值
2. 先做线性回归 baseline
3. 先看 `RMSE / MAE / R²`
4. 再看残差图有没有明显模式
5. 再决定下一步是改特征，还是加正则化，还是换更复杂模型

如果你把这 5 步先练熟，线性回归这一节就真的学进去了。

---

## 小结

```mermaid
mindmap
  root((线性回归))
    简单线性回归
      y = wx + b
      一个特征
    多元线性回归
      y = w1x1 + w2x2 + ... + b
      多个特征
    求解方法
      正规方程（解析解）
      梯度下降（迭代）
    多项式回归
      特征扩展
      容易过拟合
    正则化
      Ridge（L2）
      Lasso（L1）
      Elastic Net
```

| 要点 | 说明 |
|------|------|
| 核心思想 | 找一个线性函数拟合数据，最小化 MSE |
| 正规方程 | 直接求解，小数据快，大数据慢 |
| 梯度下降 | 迭代求解，大数据友好 |
| 多项式回归 | 用高次特征拟合非线性数据 |
| 正则化 | 惩罚大参数，防止过拟合 |

## 这节最该带走什么

如果只带走一句话，我希望你记住：

> **线性回归最重要的不是那条直线，而是它第一次把“建模、损失、求解、诊断、泛化”这一整套机器学习思路串了起来。**

所以学完这节，真正的收获应该是：

- 知道什么叫 baseline
- 知道为什么要先定义损失
- 知道正规方程和梯度下降是在解同一个问题
- 知道模型不是越复杂越好
- 知道评估和诊断会决定下一步怎么做

:::info 连接后续
- **下一节**：逻辑回归——从回归到分类，只需加一个 Sigmoid
- **第 4 站回顾**：梯度下降（3.3 节）、交叉熵（2.4 节）
:::

---

## 动手练习

### 练习 1：手动实现梯度下降

用上面的多特征房价数据，手动实现多元线性回归的梯度下降（提示：先对特征做标准化）。

### 练习 2：正则化调参

用 `load_diabetes()` 数据集，尝试不同的 Ridge alpha 值（0.01, 0.1, 1, 10, 100），画出 alpha 与测试集 R² 的关系图，找到最优 alpha。

### 练习 3：多项式阶数选择

生成非线性数据，用不同阶数的多项式回归拟合，分别计算训练集和测试集的 R²。画出"阶数 vs R²"图，观察过拟合的拐点。
