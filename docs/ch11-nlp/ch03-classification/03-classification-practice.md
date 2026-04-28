---
title: "3.4 文本分类实战"
sidebar_position: 9
description: "围绕一个客服工单分类任务，从标签设计、数据整理、基线训练到错误分析，走完一条真正的文本分类项目闭环。"
keywords: [text classification practice, intent classification, baseline, error analysis, NLP project]
---

# 文本分类实战

:::tip 本节定位
前两节分别讲了：

- 传统文本分类
- 深度学习文本分类

这一节要把它们放回真实项目里。  
真正的文本分类项目，难点通常不只在模型，而还在：

- 标签怎么定
- 数据怎么做
- 基线怎么比
- 错误怎么分析

这节课会围绕一个小型客服意图分类任务，把这条闭环走一遍。
:::

## 学习目标

- 学会给文本分类任务定义清楚标签边界
- 学会做一个能解释结果的轻量基线
- 学会从错误案例里看出数据或标签问题
- 通过可运行示例建立完整项目骨架

---

## 一、项目问题先要定义清楚

### 1.1 场景

我们做一个最小客服工单意图分类器，目标类别为：

- `refund`
- `invoice`
- `password`

### 1.2 为什么这个题目适合练手？

因为它同时具备：

- 输入清楚：用户一句话
- 输出清楚：意图类别
- 错误易分析：分错后通常能追到词和标签边界

### 1.3 第一个关键点不是模型，而是标签边界

例如：

- “退款多久到账” 是 `refund`
- “发票什么时候能开” 是 `invoice`
- “忘记密码怎么办” 是 `password`

这件事必须先清楚。

---

## 二、先做一个可解释基线

这里我们不用外部依赖，  
直接写一个最小关键词统计基线，让你先看到完整闭环。

```python
from collections import Counter, defaultdict

train_data = [
    ("退款多久到账", "refund"),
    ("怎么申请退款", "refund"),
    ("发票什么时候可以开", "invoice"),
    ("电子发票发到哪里", "invoice"),
    ("忘记密码怎么办", "password"),
    ("密码重置入口在哪", "password"),
]

test_data = [
    ("退款怎么处理", "refund"),
    ("电子发票什么时候开", "invoice"),
    ("重置密码需要多久", "password"),
]


def tokenize(text):
    return list(text)


class KeywordClassifier:
    def __init__(self):
        self.class_word_counts = defaultdict(Counter)
        self.class_counts = Counter()

    def fit(self, data):
        for text, label in data:
            self.class_counts[label] += 1
            self.class_word_counts[label].update(tokenize(text))

    def predict_one(self, text):
        tokens = tokenize(text)
        scores = {}

        for label, word_counts in self.class_word_counts.items():
            score = 0
            for token in tokens:
                score += word_counts[token]
            scores[label] = score

        return max(scores, key=scores.get), scores

    def evaluate(self, data):
        correct = 0
        details = []
        for text, gold in data:
            pred, scores = self.predict_one(text)
            correct += int(pred == gold)
            details.append({"text": text, "gold": gold, "pred": pred, "scores": scores})
        return correct / len(data), details


clf = KeywordClassifier()
clf.fit(train_data)
acc, details = clf.evaluate(test_data)

print("accuracy:", round(acc, 4))
for item in details:
    print(item)
```

### 2.1 这个示例为什么有价值？

因为它把一个分类项目最核心的 4 件事都放进来了：

1. 训练集
2. 测试集
3. 可运行基线
4. 明细输出

### 2.2 为什么我们故意从“很简单”的基线开始？

因为这样你更容易：

- 看懂预测为什么这样来
- 找到数据问题
- 知道更强模型到底比基线强在哪

---

## 三、文本分类项目里最有价值的不是总分，而是错误分析

### 3.1 先看总准确率

准确率能让你知道：

- 这版系统大概行不行

### 3.2 但真正有洞察的是逐条明细

你需要看：

- 哪类样本最容易分错
- 错在词面相似、标签重叠，还是训练数据不够

### 3.3 一个简单的错误分析函数

```python
def error_cases(details):
    return [item for item in details if item["gold"] != item["pred"]]


errors = error_cases(details)
print("errors:", errors)
```

如果错误很多，你应该先问：

- 类别边界是不是太模糊
- 训练样本是不是不平衡
- 关键词基线是不是天生不够

---

## 四、什么时候该从传统方法升级到深度方法？

### 4.1 当错误主要来自语义表达变化

例如：

- 没出现训练里常见关键词
- 但语义其实是同一类

### 4.2 当你发现词袋特征不够用了

比如：

- 句子更长
- 否定和上下文影响更大
- 类别边界更微妙

### 4.3 但升级前先保留基线

基线非常重要，因为它能帮助你回答：

- 深度模型到底提升了什么

---

## 五、一个项目闭环应该怎么讲？

### 5.1 任务定义

先说清楚：

- 输入是什么
- 输出是什么
- 标签是怎么定的

### 5.2 基线

说明：

- 用了什么最小方法
- 为什么用它

### 5.3 评估与错误分析

至少展示：

- 准确率
- 几个典型成功案例
- 几个典型失败案例

### 5.4 下一步优化方向

例如：

- 扩充数据
- 引入 TF-IDF + 线性模型
- 再升级到 embedding / 深度模型

---

## 六、最常见误区

### 6.1 误区一：一开始就上最复杂模型

这样很容易失去对任务本身的判断。

### 6.2 误区二：只看总准确率

不看错误明细，很难真正改进。

### 6.3 误区三：标签定义含糊

标签一旦模糊，再强模型也会学得不稳。

---

## 小结

这节最重要的是建立一个项目习惯：

> **文本分类项目最先要把标签边界、可解释基线和错误分析做扎实，而不是一上来追求最复杂模型。**

只要这个习惯建立起来，后面做更复杂 NLP 项目时会稳很多。

---

## 练习

1. 给示例再加一个新类别，例如 `shipping`，并扩充几条训练样本。
2. 用错误明细看看哪些预测最容易混淆，猜一猜原因。
3. 你会在什么情况下决定从这个关键词基线升级到深度模型？
4. 如果标签定义本身模糊，你会先改模型还是先改数据？为什么？
