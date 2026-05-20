---
title: "E.C 经典机器学习路线图"
sidebar_position: 0
description: "经典机器学习补充模块的简明实操路线图：SVM、KNN、朴素贝叶斯和 LDA，作为中小数据任务的强 baseline。"
---

# E.C 经典机器学习路线图

当数据量不大、特征清楚，或者你需要在重模型之前先做强 baseline 时，再回来学这个模块。

## 先看 baseline 地图

![经典机器学习补充模块地图](/img/course/elective-classic-ml-module-map.webp)

![KNN 邻居投票图](/img/course/elective-knn-neighbor-voting.webp)

经典机器学习帮你回答：这个问题是否已经能被简单特征解决。

## 跑最小 KNN baseline

```python
def distance(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5

train = [
    ([0.1, 0.2], "low"),
    ([0.2, 0.1], "low"),
    ([0.8, 0.9], "high"),
    ([0.9, 0.8], "high"),
]

point = [0.75, 0.85]
nearest = min(train, key=lambda row: distance(row[0], point))
print("prediction:", nearest[1])
print("neighbor:", nearest[0])
```

预期输出：

```text
prediction: high
neighbor: [0.8, 0.9]
```

这是最小 baseline 习惯：定义特征，比较距离，预测，并把结果留给后续对比。

## 按这个顺序学

| 步骤 | 课程 | 练习产物 |
|---|---|---|
| 1 | [E.C.1 SVM](./01-svm.md) | 解释 margin、support vectors、`C` 和 kernel 选择 |
| 2 | [E.C.2 KNN](./02-knn.md) | 建一个距离投票 baseline |
| 3 | [E.C.3 朴素贝叶斯](./03-naive-bayes.md) | 把证据计数转成类别概率 |
| 4 | [E.C.4 LDA](./04-lda.md) | 把特征投影到更容易分开类别的方向 |

## 通过标准

你能构建一个经典 baseline，解释为什么适合，并和更重的模型或后续项目结果做比较，就算通过本模块。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
model_family: SVM, KNN, Naive Bayes, LDA, or another classical baseline
dataset_view: feature scale, class balance, decision boundary, and train/test split
metric: accuracy/F1, confusion matrix, margin, neighbor behavior, or projection quality
failure_check: scaling, high dimensionality, weak assumptions, leakage, or poor baseline fit
Expected_output: classical-ML baseline result with one limitation note
```
