---
title: "5.4.4 偏差-方差权衡"
description: "一节跟着操作的偏差方差课程：欠拟合、过拟合、模型复杂度、训练测试差距、学习曲线和实用修复"
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "偏差, 方差, 欠拟合, 过拟合, 学习曲线, 验证曲线, 模型复杂度"
---
![偏差方差权衡三联图](/img/course/bias-variance-tradeoff.webp)

:::tip[本节概览]
偏差和方差不是只存在于理论里的词。它们是一种诊断方式：模型是太简单、太不稳定，还是受限于数据质量。
:::

## 什么时候需要偏差-方差诊断？

当模型效果不好时，最危险的反应是马上换一个更复杂的模型，或者开始乱调一堆参数。偏差-方差权衡先帮你判断：问题到底是“模型学不会”，还是“模型太会背训练集”。

可以这样想：

| 现象 | 不先诊断会怎样 | 偏差-方差要回答什么 |
|---|---|---|
| 训练分数和验证分数都低 | 盲目加数据或调小模型 | 是不是模型太简单、特征没信号、标签太乱？ |
| 训练分数很高，验证分数低 | 继续加复杂度让训练更好看 | 是不是模型记住了训练细节，泛化差？ |
| 换划分后分数波动大 | 误以为一次好分数可靠 | 数据量、分群或验证方式是否不稳定？ |

所以本节不是为了背“高偏差 / 高方差”两个词，而是为了学会在改模型之前先诊断。

## 你会做出什么

本节用决策树演示：

- 模型复杂度如何改变训练分数和测试分数；
- 如何通过 train-test gap 判断欠拟合和过拟合；
- 学习曲线如何说明更多数据是否可能有帮助；
- 高偏差和高方差分别应该采取什么行动。

![偏差方差行动诊断图](/img/course/ch05-bias-variance-action-map.webp)

## 环境准备

```bash
python -m pip install -U scikit-learn numpy
```

## 运行完整实验

新建 `bias_variance_lab.py`：

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.tree import DecisionTreeClassifier


X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print("complexity_lab")
for depth in [1, 3, 5, None]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)
    model.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    gap = train_acc - test_acc
    print(
        f"max_depth={str(depth):<4} "
        f"train={train_acc:.3f} test={test_acc:.3f} gap={gap:.3f} "
        f"leaves={model.get_n_leaves()}"
    )

print("learning_curve_lab")
model = DecisionTreeClassifier(max_depth=3, random_state=42)
train_sizes, train_scores, val_scores = learning_curve(
    model,
    X,
    y,
    cv=5,
    train_sizes=[0.2, 0.4, 0.6, 0.8, 1.0],
    scoring="accuracy",
)
for size, train_mean, val_mean in zip(train_sizes, train_scores.mean(axis=1), val_scores.mean(axis=1)):
    print(f"train_size={size:<3} train={train_mean:.3f} cv={val_mean:.3f} gap={train_mean - val_mean:.3f}")
```

运行：

```bash
python bias_variance_lab.py
```

预期输出：

```text
complexity_lab
max_depth=1    train=0.923 test=0.923 gap=-0.001 leaves=2
max_depth=3    train=0.977 test=0.944 gap=0.032 leaves=7
max_depth=5    train=0.995 test=0.937 gap=0.058 leaves=15
max_depth=None train=1.000 test=0.923 gap=0.077 leaves=18
learning_curve_lab
train_size=91  train=0.989 cv=0.847 gap=0.142
train_size=182 train=0.986 cv=0.870 gap=0.116
train_size=273 train=0.978 cv=0.903 gap=0.075
train_size=364 train=0.975 cv=0.917 gap=0.057
train_size=455 train=0.974 cv=0.919 gap=0.055
```

![偏差方差实验结果图](/img/course/ch05-bias-variance-result-map.webp)

## 读懂复杂度实验

`max_depth` 越大，树越复杂：

```text
max_depth=1    train=0.923 test=0.923 gap=-0.001 leaves=2
max_depth=None train=1.000 test=0.923 gap=0.077 leaves=18
```

`max_depth=1` 很简单。训练和测试很接近，但分数不是最好。这可能是高偏差：模型太简单。

`max_depth=None` 把训练集记得很完美，但测试准确率下降。这是高方差：模型学到了训练细节，却不能泛化。

实用上最好的模型往往在中间：

```text
max_depth=3 train=0.977 test=0.944 gap=0.032
```

它没有在训练集上满分，但泛化更好。

## 学习曲线

![学习曲线诊断图](/img/course/ch05-learning-curve-diagnosis-map.webp)

学习曲线展示训练数据增加时会发生什么：

```text
train_size=91  train=0.989 cv=0.847 gap=0.142
train_size=455 train=0.974 cv=0.919 gap=0.055
```

数据增多后，验证分数上升，gap 变小。这说明更多数据可能有帮助，但模型仍然可以通过更好的特征或调参继续改进。

## 诊断规则

| 模式 | 可能问题 | 尝试 |
|---|---|---|
| train 低，validation 低 | 高偏差 / 欠拟合 | 更强模型、更好特征、减弱正则化 |
| train 高，validation 低 | 高方差 / 过拟合 | 简化模型、加强正则化、增加数据 |
| train 高，validation 高 | 拟合良好 | 在最终 holdout 上测试并监控漂移 |
| validation 按 fold 波动大 | 不稳定或存在数据分群 | 检查 fold，增加数据，使用更稳模型 |

不要只靠一个指标诊断。要同时看训练分数、验证分数、gap，以及错误是否集中在某个分群。

## 实用修复

高偏差时：

- 增加有用特征；
- 使用表达能力更强的模型；
- 减弱过强正则化；
- 如果是迭代模型，训练更久。

高方差时：

- 降低模型复杂度；
- 加强正则化；
- 收集更多样且有代表性的数据；
- 使用交叉验证和最终 holdout；
- 考虑能降低方差的集成模型。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
评估设置：划分、交叉验证、指标、基线和对比目标
结果：分数表、曲线、混淆矩阵、验证结果，或搜索结果
决策：是否更改数据、特征、模型、阈值或超参数
失败检查：泄漏、验证不稳定、指标错误或在测试集上调参
期望产出：支持下一步建模决策的评估记录
```

## 常见排查清单

| 现象 | 可能原因 | 修复方式 |
|---|---|---|
| train 和 validation 都差 | 模型表达不了模式 | 改进特征或模型类别 |
| train 满分，validation 较差 | 过拟合 | 限制深度、剪枝、正则化 |
| 更多数据提升 validation | 方差或数据不足 | 收集更有代表性的数据 |
| 更多数据没帮助 | 高偏差或标签噪声 | 改进特征、标签或模型 |
| validation 按 fold 跳动 | 数据不均匀 | 检查分群分布 |

## 练习

1. 给树加入 `min_samples_leaf=5`。gap 怎么变？
2. 尝试 `max_depth=2, 4, 6, 8`。测试准确率在哪里最高？
3. 把树换成逻辑回归。问题更像偏差还是方差？
4. 在复杂度实验中改用 5 折交叉验证，而不是一次测试切分。
5. 查看最佳树的错误样本。错误是否集中在某个类别？

<details>
<summary>参考实现与讲解</summary>

1. `min_samples_leaf=5` 通常会缩小训练/测试差距，因为树不能轻易记住很小的叶子。如果两个分数都下降，模型可能变得过于简单。
2. 测试准确率往往在中等深度达到峰值。太浅会欠拟合，太深会过拟合，即使训练准确率继续上升。
3. 逻辑回归是判断 bias 的有用基线。如果它和树都低，可能是特征或模型族不够；如果树训练很好但测试差，主要是 variance 问题。
4. 5 折交叉验证比一次切分更稳。应选择平均分较高且波动可接受的复杂度。
5. 错误集中在某个类别，通常提示该类别特征不足、标签模糊或类别不均衡；错误分散则可能是噪声更难消除。

</details>

## 过关检查

你能解释下面几点，就完成本节：

- 高偏差表示模型太简单或缺少信号；
- 高方差表示模型对训练细节太敏感；
- train-validation gap 是实用诊断工具；
- 学习曲线能说明更多数据是否可能有帮助；
- 修复方式取决于现象，不取决于术语本身。
