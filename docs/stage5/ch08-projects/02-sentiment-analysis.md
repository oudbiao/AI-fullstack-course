---
title: "8.3 项目：文本情感分析"
sidebar_position: 2
description: "围绕一个真正可展示的情感分析项目，从标签边界、baseline、错误分析到交付方式，走完完整闭环。"
keywords: [sentiment analysis project, text classification, baseline, negation, sarcasm, NLP]
---

# 项目：文本情感分析

:::tip 本节定位
情感分析项目很适合做作品集，不是因为它最炫，而是因为它很适合训练“项目判断力”：

- 标签怎么定
- baseline 怎么建
- 错误怎么解释
- 结果怎么展示

这一节的目标不是堆复杂模型，而是把一个小项目真正做完整。
:::

## 学习目标

- 学会给情感分析任务设计稳定的标签边界
- 学会搭一个可解释 baseline 并读懂结果
- 学会把错误分析做成项目亮点，而不是事后补丁
- 学会把一个小 NLP 项目包装成可交付作品

---

## 一、项目题目先要收窄

### 1.1 最稳的起点是二分类

先做：

- positive
- negative

而不是一开始就做：

- positive / neutral / negative / irony / mixed

### 1.2 为什么二分类适合练手？

因为：

- 标签更清楚
- 数据更容易准备
- 错误更容易分析

### 1.3 一个适合作品集的题目

例如：

> **做一个“课程评价情感分析器”，判断评论是正向还是负向。**

这个题目非常适合，因为用户语料、标签边界和业务意义都比较清楚。

---

## 二、项目最小闭环长什么样？

1. 定义标签边界
2. 准备小型标注数据
3. 做 baseline
4. 做错误分析
5. 设计一个最小推理接口或展示页

如果这 5 步都清楚，你的项目通常就已经比“只训个模型”更像作品级了。

## 三、推荐推进顺序

对新人来说，更稳的顺序通常是：

1. 先把标签定义写出来
2. 再做最简单 baseline
3. 再补一个传统 ML baseline
4. 最后再考虑更强的深度模型

这样你不会在一开始就被模型复杂度带跑。

---

## 四、先跑一个最小 baseline 项目

为了让逻辑足够清楚，这里先用一个关键词统计型 baseline。  
它当然不够强，但非常适合解释项目闭环。

```python
from collections import Counter

train_data = [
    ("这门课讲得很清楚", "positive"),
    ("案例很多，学起来很顺", "positive"),
    ("内容太乱了", "negative"),
    ("讲得太快，听不懂", "negative"),
]

test_data = [
    ("这门课真的很清楚", "positive"),
    ("内容有点乱", "negative"),
    ("案例很多但是讲太快", "negative"),
]


positive_words = Counter()
negative_words = Counter()

for text, label in train_data:
    tokens = list(text)
    if label == "positive":
        positive_words.update(tokens)
    else:
        negative_words.update(tokens)


def predict(text):
    score = 0
    for token in text:
        score += positive_words[token]
        score -= negative_words[token]
    return "positive" if score >= 0 else "negative", score


results = []
for text, gold in test_data:
    pred, score = predict(text)
    results.append({"text": text, "gold": gold, "pred": pred, "score": score})
    print(results[-1])
```

### 4.1 这个 baseline 为什么有教学价值？

因为它很容易解释：

- 为什么判成正面
- 为什么判成负面

这让你能真正做“错误分析”，而不是只盯一个数字。

### 4.2 再补一个“否定词翻转”的最小升级版

情感分析里最典型的一类错误就是：

- 明明有正向词
- 但前面被一个否定词翻掉了

比如：

- “不推荐”
- “不清楚”
- “不值得”

下面这个极小版本不是工业方案，  
但很适合初学者第一次体会：

- **规则补丁为什么会直接改变错例分布**

```python
negation_words = {"不", "没", "无"}


def predict_with_negation(text):
    score = 0
    chars = list(text)
    for idx, token in enumerate(chars):
        token_score = positive_words[token] - negative_words[token]

        # 如果前一个字是否定词，就把当前情感方向翻一下
        if idx > 0 and chars[idx - 1] in negation_words:
            token_score *= -1

        score += token_score

    return "positive" if score >= 0 else "negative", score


extra_cases = [
    ("不推荐这门课", "negative"),
    ("讲得不清楚", "negative"),
    ("案例不少，但不系统", "negative"),
]

for text, gold in extra_cases:
    pred, score = predict_with_negation(text)
    print({"text": text, "gold": gold, "pred": pred, "score": score})
```

这段代码的教学价值不在“规则够强”，  
而在于它会让你第一次很清楚地看到：

- baseline 为什么会错
- 一种具体补丁会修掉哪类错误
- 错误分析怎样真正反过来推动方案升级

---

## 五、真正让项目变强的是错误分析

### 5.1 先把错例挑出来

```python
errors = [row for row in results if row["gold"] != row["pred"]]
print(errors)
```

### 5.2 常见错误类型

对情感分析来说，最值得单独看的是：

- 否定词  
  例如“不差”“不推荐”
- 反讽  
  例如“真棒，又崩了”
- 混合评价  
  例如“内容很好，但太难了”

### 5.3 为什么错误分析这么值钱？

因为它能直接告诉你下一步该怎么做：

- 补数据
- 改标签标准
- 升级模型

### 5.4 给自己做一张最小错误分桶表

初学者做情感分析项目时，很容易只说：

- “模型错了几条”

但更有价值的是先分桶：

```python
error_buckets = {
    "negation": [],
    "sarcasm": [],
    "mixed_sentiment": [],
    "other": [],
}

examples = [
    ("不推荐这门课", "negative", "positive"),
    ("真棒，又卡住了", "negative", "positive"),
    ("内容很好，但节奏太快", "negative", "positive"),
]

for text, gold, pred in examples:
    if "不" in text:
        error_buckets["negation"].append(text)
    elif "真棒" in text and "又" in text:
        error_buckets["sarcasm"].append(text)
    elif "但" in text:
        error_buckets["mixed_sentiment"].append(text)
    else:
        error_buckets["other"].append(text)

for k, v in error_buckets.items():
    print(k, len(v), v)
```

这张表非常适合展示在项目里，  
因为它会让别人立刻看出：

- 你不是只会报分数
- 你知道错误是有类型的
- 你下一步怎么改是有依据的

---

## 六、这个项目怎么往作品级再推一步？

### 6.1 补一个传统强基线

例如：

- TF-IDF + LogisticRegression

让你的项目至少有：

- 规则基线
- 传统 ML 基线

### 6.2 再补一个深度 baseline

例如：

- embedding + pooling
- BERT 分类

### 6.3 展示时不要只放总分

很推荐展示：

- 标签定义
- baseline 对比
- 典型错例
- 负样本 hardest cases

这样项目会非常完整。

### 6.4 一个更像真实项目的展示顺序

如果你把这题做成作品集页面，  
比较推荐按这个顺序展示：

1. 任务定义和标签边界
2. baseline 方法
3. baseline 对比表
4. 错误分桶
5. 你针对哪类错误做了什么升级
6. 最终保留下来的方案和原因

这样别人看到的就不是“做了一个情感分类器”，  
而是：

- 你真的知道一个 NLP 小项目该怎么从 baseline 走到作品级

---

## 七、最容易踩的坑

### 7.1 标签标准不一致

这是很多情感项目的第一大坑。

### 7.2 只看准确率

不看具体错在哪，你很难真正优化。

### 7.3 一开始就追求最复杂模型

没有 baseline，复杂模型的提升就很难讲清。

---

## 项目交付时最好补上的内容

- 一张标签定义表
- 一张 baseline 对比表
- 一组典型错例
- 一段你对下一步升级路线的判断

---

## 小结

这节最重要的是建立一个项目习惯：

> **情感分析项目最有价值的地方，不是模型多复杂，而是你能否把标签边界、baseline、错误分析和升级路线讲成一个完整闭环。**

只要这一点做到位，即使题目不大，也会非常像作品级课程。

## 这节最该带走什么

- 情感分析最值钱的不是复杂模型，而是清楚的标签边界和错误分析
- 一个简单 baseline 只要够可解释，就非常有教学价值
- 真正能拉开项目质量差距的，往往是你有没有把错例变成下一步动作

## 练习

1. 自己设计 12 条课程评价，并给出正负标签。
2. 在 baseline 上再手工加入一个“否定词翻转规则”，看看能否修掉某类错误。
3. 想一想：为什么情感分析特别适合做错误分析展示？
4. 如果你要把这个项目扩成三分类，你会先改标签标准还是先换模型？为什么？
