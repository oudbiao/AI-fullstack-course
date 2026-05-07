---
title: "E.C.3 朴素贝叶斯"
sidebar_position: 14
description: "用词袋模型和 Multinomial Naive Bayes 搭建一个小型文本分类 baseline。"
keywords: [naive bayes, multinomial nb, text classification, probability, smoothing]
---

# E.C.3 朴素贝叶斯

![朴素贝叶斯证据累积图](/img/course/elective-naive-bayes-evidence.png)

朴素贝叶斯会比较“哪个类别更可能生成这些证据”。在文本任务里，词频常常已经足够构建一个便宜又有用的 baseline。

## 准备内容

- Python 3.10+
- 当前稳定版 `scikit-learn`

```bash
python -m pip install -U scikit-learn
```

## 关键术语

- **Bag of words（词袋）**：用词频表示文本。
- **Conditional probability（条件概率）**：给定类别时，某个证据出现的概率。
- **朴素假设**：给定类别后，各特征之间近似独立。
- **Smoothing（平滑）**：避免没见过的词直接变成不可能。
- **`alpha`**：`MultinomialNB` 里的平滑强度。

## 运行文本分类器

创建 `naive_bayes_text.py`：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

texts = [
    "How long does a refund take?",
    "How do I apply for a refund?",
    "When can I issue an invoice?",
    "Where is the e-invoice sent?",
    "What should I do if I forget my password?",
    "Where is the password reset entry?",
]

labels = [
    "refund",
    "refund",
    "invoice",
    "invoice",
    "password",
    "password",
]

model = make_pipeline(
    CountVectorizer(),
    MultinomialNB(alpha=1.0),
)

model.fit(texts, labels)
pred = model.predict([
    "How do I handle a refund?",
    "When can I issue an e-invoice?",
])
print("predictions:", pred.tolist())
```

运行：

```bash
python naive_bayes_text.py
```

预期输出：

```text
predictions: ['refund', 'invoice']
```

这是一个完整 baseline：文本转词频，词频转概率，概率转标签。

## 改变平滑

把 `alpha=1.0` 改成 `0.1` 和 `2.0`。在小数据集里，平滑会明显影响模型对罕见词的信任程度。

## 实用判断

适合尝试朴素贝叶斯：

1. 任务是文本分类。
2. 需要快速 baseline。
3. 数据少，标签相对简单。
4. 需要一定可解释性。

当语义、上下文或词序很重要时，再换更强模型。

## 常见错误

- 以为“朴素”就等于没用。
- 忘记特征表达仍然很重要。
- 只拿它和大模型比，而不是把它当便宜 baseline。

## 练习

添加一个 `certificate` 类别和两个样本。再测试一个证书问题是否能被分到新标签。
