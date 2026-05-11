---
title: "5 机器学习入门到实战"
sidebar_position: 0
description: "学习实用建模闭环：定义任务、划分数据、训练 baseline、评估、查看错误、改进特征并写报告。"
keywords: [机器学习, Scikit-learn, 监督学习, 无监督学习, 回归, 分类, 聚类]
---

# 5 机器学习入门到实战

![机器学习主视觉](/img/course/ch05-machine-learning.webp)

第 5 章只解决一件事：把数据问题变成**可训练、可评估、可改进的机器学习项目**。

## 先看建模闭环

![机器学习建模主线闭环](/img/course/ch05-modeling-loop-backbone.webp)

先看图。可靠的机器学习工作大多遵循这个闭环：

```text
定义任务 -> 划分数据 -> 训练 baseline -> 评估 -> 查看错误 -> 改进
```

先做 baseline，再追模型名。baseline 能告诉你：后面的改动到底有没有带来真实提升。

## 学习顺序与任务表

下面这一张表同时作为本章学习指南和任务清单。

| 页面 | 跟着做 | 留下的证据 |
|---|---|---|
| [5.1 机器学习基础](ch01-ml-basics/00-roadmap.md) | 识别分类、回归、聚类、异常检测、特征、标签、训练/测试划分和 sklearn 流程 | 一份问题定义说明 |
| [5.1.5 机器学习历史](ch01-ml-basics/04-history-breakthroughs.md) | 可选背景：浏览经典算法为什么陆续出现 | 一条“这个算法为什么存在”的说明 |
| [5.2 监督学习](ch02-supervised/00-roadmap.md) | 先跑回归和分类样例，再比较多个模型 | 一个 baseline 分数和一个改进分数 |
| [5.3 无监督学习](ch03-unsupervised/00-roadmap.md) | 没有标签时尝试聚类、降维和异常检测 | 一张图或一段聚类解释 |
| [5.4 模型评估](ch04-evaluation/00-roadmap.md) | 选择指标、使用交叉验证、诊断偏差/方差、谨慎调参 | 指标选择说明和错误样本 |
| [5.5 特征工程](ch05-feature-engineering/00-roadmap.md) | 处理缺失值、类别、缩放、特征构造、特征选择和 Pipeline | 特征处理记录和泄漏检查 |
| [5.6 项目](ch06-projects/00-roadmap.md) 和 [5.6.6 工作坊](ch06-projects/05-hands-on-ml-workshop.md) | 在房价、流失、分群或 Kaggle 前，先做可复现证据包 | README、模型对比、错误分析和下一步计划 |

本章常见术语：

| 术语 | 含义 |
|---|---|
| `feature` | 模型可以使用的输入列 |
| `label` / `target` | 模型要学习预测的答案 |
| `baseline` | 必须先超过的最简单模型或规则 |
| `metric` | 衡量模型的尺子，例如 F1、AUC、MAE、RMSE |
| `leakage` | 测试集或目标信息意外进入训练过程 |
| `Pipeline` | 把预处理和模型封装在一起，降低泄漏风险 |

## 第一个可运行闭环

如果还没有 sklearn，先安装：

```bash
python -m pip install scikit-learn
```

然后运行下面这个自包含 baseline。它使用内置数据集，划分数据，训练 dummy baseline，再训练一个真实模型，并对比两者。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline")
print(classification_report(y_test, baseline.predict(X_test), zero_division=0))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("Logistic regression")
print(classification_report(y_test, model.predict(X_test), zero_division=0))
```

预期形态：

```text
Baseline
...
Logistic regression
...
```

不要只比较最终分数。继续问：哪些类别容易，哪些类别难，真实场景里哪类错误最重要？

## 深度阶梯

| 层级 | 你能证明什么 |
|---|---|
| 最低通过 | 能说出任务类型、划分数据、训练 baseline，并读懂分数。 |
| 项目可用 | 能解释为什么选这个指标，并展示一条错误样本，而不是只相信一个分数。 |
| 深度检查 | 能检查泄漏、比较两种特征方案，并说明真实产品或数据更新后会有什么变化。 |

## 常见失败

| 现象 | 先检查什么 | 常见修复 |
|---|---|---|
| 分数高得奇怪 | 数据泄漏或训练/测试划分错误 | 训练前检查特征和切分方式 |
| 训练高、测试低 | 过拟合 | 简化模型、正则化或增加数据 |
| 所有模型都弱 | 标签差、特征弱或指标不合适 | 查看错误样本和标签定义 |
| 准确率不错但业务风险高 | 类别不平衡或漏判代价高 | 使用 recall、precision、F1、AUC 或阈值复盘 |
| 结果无法复现 | 随机种子、数据版本或依赖变化 | 固定 seed 并记录版本 |

## 通关检查

能回答下面五个问题，就可以进入第 6 章：

- 这是分类、回归、聚类还是异常检测？
- baseline 是什么？真实模型必须超过哪个分数？
- 哪个指标匹配目标？什么时候准确率会误导？
- 你怎样检查数据泄漏？
- 模型擅长什么、不擅长什么、下一步先改哪里？

需要打印式清单时，打开 [5.0 学习指南与任务单](./study-guide.md)。下一章会从 sklearn 模型进入神经网络和深度学习训练。
