---
title: "5.4.1 评估路线图：调参之前先相信分数"
sidebar_position: 9
description: "紧凑版模型评估路线图：指标、交叉验证、偏差方差、超参数调优和证据。"
keywords: [模型评估指南, 交叉验证, 偏差方差, 超参数调优]
---

# 5.4.1 评估路线图：调参之前先相信分数

模型评估回答一个问题：模型真的好，还是分数只是碰巧好看？

## 先看评估地图

![模型评估学习地图](/img/course/ml-evaluation-roadmap.png)

![模型评估章节流程](/img/course/ch05-evaluation-chapter-flow.png)

| 主题 | 先问的问题 |
|---|---|
| 指标 | 哪个分数匹配这个任务？ |
| 交叉验证 | 分数在不同划分下稳定吗？ |
| 偏差方差 | 模型太简单还是太灵活？ |
| 调参 | 哪个参数变化真的更好？ |

## 跑一次交叉验证检查

创建 `evaluation_first_loop.py`，安装 `scikit-learn` 后运行。

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

X, y = load_iris(return_X_y=True)
model = DecisionTreeClassifier(max_depth=2, random_state=42)
scores = cross_val_score(model, X, y, cv=5)

print("fold_scores:", [float(round(score, 3)) for score in scores])
print("mean_accuracy:", round(scores.mean(), 3))
```

预期输出：

```text
fold_scores: [0.933, 0.967, 0.9, 0.867, 1.0]
mean_accuracy: 0.933
```

一个分数只是快照，多折结果能告诉你这个结果是否稳定到值得相信。

## 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [5.4.2 评估指标](./01-metrics.md) | accuracy、precision、recall、F1、R2、RMSE |
| 2 | [5.4.3 交叉验证](./02-cross-validation.md) | 稳定估计、数据划分风险 |
| 3 | [5.4.4 偏差与方差](./03-bias-variance.md) | 欠拟合、过拟合、学习曲线 |
| 4 | [5.4.5 超参数调优](./04-hyperparameter-tuning.md) | 网格搜索、对比记录 |

## 通过标准

能为任务选择指标，解释一次分数稳定性检查，并避免在评估方法不可信时急着调参，就算通过。
