---
title: "E.C.1 支持向量机"
description: "把 SVM 用作中小数据、边界清晰、特征已缩放任务的实用 baseline。"
sidebar:
  order: 12
head:
  - tag: meta
    attrs:
      name: keywords
      content: "SVM, support vector machine, max margin, kernel, classification, classic ML"
---

# E.C.1 支持向量机

![SVM 最大间隔与支持向量图](/img/course/elective-svm-margin-support-vectors.webp)

![SVM 参数 C 与 kernel 选择图](/img/course/elective-svm-c-kernel-decision-map.webp)

SVM 会寻找一个间隔尽量大的决策边界。离边界最近的点叫支持向量，它们是最影响边界位置的关键样本。

## 准备内容

- Python 3.10+
- 当前稳定版 `scikit-learn` 和 `numpy`

```bash
python -m pip install -U scikit-learn numpy
```

## 关键术语

- **Margin（间隔）**：边界到最近样本的距离。
- **Support vector（支持向量）**：靠近边界的关键样本。
- **`C`**：控制对错误的容忍度。`C` 越大，通常越贴合训练集。
- **Kernel（核函数）**：控制边界是线性的，还是非线性的。
- **Scaling（缩放）**：SVM 通常需要把特征范围标准化。

## 运行线性 SVM baseline

创建 `svm_baseline.py`：

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X = np.array([
    [1.0, 1.2],
    [1.3, 0.9],
    [1.1, 1.0],
    [4.0, 4.2],
    [4.3, 3.8],
    [3.9, 4.1],
])
y = np.array([0, 0, 0, 1, 1, 1])

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="linear", C=1.0),
)

model.fit(X, y)
pred = model.predict([[1.2, 1.1], [4.2, 4.0]])
svc = model.named_steps["svc"]

print("predictions:", pred.tolist())
print("support_per_class:", svc.n_support_.tolist())
```

运行：

```bash
python svm_baseline.py
```

预期输出：

```text
predictions: [0, 1]
support_per_class: [2, 1]
```

这是最小可用 SVM 习惯：缩放特征、训练模型、预测、检查支持向量。

## 改变边界

运行这个独立对比：

```python
from sklearn.svm import SVC

for kernel in ["linear", "rbf"]:
    if kernel == "linear":
        model = SVC(kernel="linear", C=1.0)
    else:
        model = SVC(kernel="rbf", C=1.0, gamma="scale")
    print(model)
```

预期输出开头类似：

```text
SVC(kernel='linear')
SVC()
```

边界简单时先用 `linear`。如果线性 SVM 明显欠拟合，或者边界看起来是弯曲的，再尝试 `rbf`。

## 实用判断

适合尝试 SVM：

1. 数据量中小。
2. 特征已经有意义。
3. 类别边界比较清楚。
4. 想在重模型之前做一个强 baseline。

如果数据很大，或者预测延迟要求极低，就要谨慎。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
模型家族：SVM、KNN、朴素贝叶斯、LDA 或其他传统基线
数据视图：特征缩放、类别平衡、决策边界和训练/测试划分
指标：准确率/F1、混淆矩阵、边距、邻近行为或投影质量
失败检查：缩放、高维度、假设薄弱、泄漏或基线拟合差
期望产出：经典机器学习基线结果，以及一条局限性说明
```

## 常见错误

- 忘记 `StandardScaler()`。
- 还没试线性就直接上复杂 kernel。
- 没检查特征质量，就先调 `C` 和 kernel。

## 练习

在边界附近加两个噪声点，对比 `C=0.1`、`C=1.0`、`C=10.0`。记录每个版本用了多少支持向量。

<details>
<summary>参考实现与讲解</summary>

好的答案会记录一个小表：`C`、预测或分数、支持向量数量。较低的 `C` 通常允许更宽、更软的间隔，也更能容忍噪声点；较高的 `C` 会更努力地把训练点分对，因此边界可能更受新噪声样本影响。

具体支持向量数量取决于你新增的点，不要编一个通用数字。正确解释应聚焦趋势：软间隔和拟合噪声训练样本之间的取舍。

</details>
