---
title: "11.3.1 文本分类路线图：文本输入、标签输出"
sidebar_position: 0
description: "文本分类的简短实操路线：构建基线，比较特征，训练分类器，并分析标签错误。"
keywords: [文本分类指南, 情感分析, TF-IDF, 文本分类项目]
---

# 11.3.1 文本分类路线图：文本输入、标签输出

文本分类接收一段文本，预测一个标签，例如情感、主题、意图或风险类型。

## 先看分类流水线

![文本分类章节学习顺序图](/img/course/ch11-classification-chapter-flow.webp)

![传统分类 baseline 图](/img/course/ch11-traditional-classification-baseline-map.webp)

![神经分类 embedding pooling 图](/img/course/ch11-neural-classification-embedding-pooling-map.webp)

复杂模型之前先做基线。大多数分类问题不是模型不够强，而是标签模糊或样本分布偏。

## 跑一个关键词基线

```python
texts = ["great course and clear examples", "confusing setup error"]
positive_words = {"great", "clear", "good", "useful"}

for text in texts:
    score = sum(word in positive_words for word in text.split())
    label = "positive" if score > 0 else "needs_review"
    print(label, "-", text)
```

预期输出：

```text
positive - great course and clear examples
needs_review - confusing setup error
```

简单基线不是最终模型，但能快速暴露标签规则和失败案例。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 传统方法 | 构建 TF-IDF 或关键词基线 |
| 2 | 深度学习方法 | 比较 embeddings、pooling、CNN/RNN/Transformer 特征 |
| 3 | 项目实战 | 追踪划分、指标、标签歧义和错误样例 |

## 通过标准

如果你能训练或模拟一个分类器，报告 accuracy/F1，并解释至少一个标签模糊案例，就通过了本章。

<details>
<summary>参考答案与讲解</summary>

1. 合格答案要从文本单元和输出类型说起：token、span、句子标签、序列、embedding 或生成文本。
2. 证据应包含小样本、模型或 pipeline 选择、评价指标，以及至少一个被检查过的错误案例。
3. 自检时要能区分预处理问题和模型问题，例如分词错误、标签歧义、数据不平衡或生成幻觉。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
label_schema: label definitions and boundary examples
dataset_split: fixed train/test examples or evaluation set
prediction: predicted label, expected label, and confidence or score
failure_check: class imbalance, label overlap, leakage, or confusing wording
Expected_output: metrics plus error samples grouped by failure reason
```
