---
title: "1.3 预训练语言模型速览"
sidebar_position: 3
description: "从“先在大语料上学通用模式，再迁移到具体任务”讲起，理解预训练模型为什么会成为现代 NLP 和大模型的共同底座。"
keywords: [pretrained models, transfer learning, BERT, GPT, T5, foundation models]
---

# 预训练语言模型速览

:::tip 本节定位
在大模型时代，“预训练”这个词几乎无处不在。  
但很多新人第一次听到时，会把它理解成一句很空的话：

- 先在大数据上学一遍

这当然没错，但还不够。

真正该建立的判断是：

> **为什么大家不再从零开始做每个 NLP 任务，而是先训练一个通用底座，再在上面做迁移。**

这节课就是这层直觉的速成入口。
:::

## 学习目标

- 理解什么叫预训练、什么叫迁移和下游适配
- 理解预训练模型为什么能“一模多用”
- 区分 encoder-only、decoder-only、encoder-decoder 的大方向
- 通过一个可运行示例理解“共享底座 + 不同任务头”的思路

---

## 一、为什么预训练模型会成为现代 NLP 的主流？

### 1.1 因为很多任务本质上共享语言能力

无论是做：

- 情感分类
- 问答
- 摘要
- 对话
- 检索

它们都离不开一些通用基础：

- 词义理解
- 句法关系
- 上下文建模
- 常识和语言模式

如果每个任务都从零学这些能力，  
成本会非常高。

### 1.2 预训练的核心思想

于是大家开始先做一件事：

- 用海量通用语料训练一个基础模型

让它先学会：

- 一般语言规律
- 通用表示
- 基础知识分布

然后再把它迁移到具体任务上。

这就像：

- 先读完大部分通识教材
- 再去做具体科目的专项训练

### 1.3 为什么这比“每个任务从头训练”好很多？

因为你不必每次都重新学习语言本身。  
下游任务只需要在已有底座上做：

- 任务头训练
- 微调
- Prompt 适配
- 检索增强

这大幅降低了门槛。

---

## 二、预训练模型到底给了我们什么？

### 2.1 给了一个“已经懂一点语言”的底座

从零随机初始化的模型，最开始什么都不会。  
而预训练模型至少已经学会了一些：

- 语法模式
- 搭配关系
- 高频事实
- 常见任务格式

这意味着下游任务不再是从完全空白开始。

### 2.2 给了可复用的表示

很多预训练模型最宝贵的地方，不只是“会回答”，  
还在于它能输出一组比较好的隐藏表示。

这些表示可以被下游任务拿去做：

- 分类
- 检索
- 匹配
- 排序

### 2.3 给了迁移学习的可能

迁移学习的核心就是：

> **在大任务上学通用能力，在小任务上少量适配。**

这也是为什么预训练模型一出来，  
整个 NLP 流程都被改写了。

---

## 三、先跑一个“共享底座 + 两个任务头”的示例

下面这段代码不会训练真正的大模型，  
但它会抓住预训练模型最核心的结构直觉：

- 有一个共享编码器
- 编码器学到通用表示
- 不同任务只在上面挂不同 head

```python
from math import sqrt

word_vectors = {
    "refund": [0.9, 0.8, 0.1],
    "order": [0.8, 0.7, 0.2],
    "password": [0.1, 0.2, 0.9],
    "reset": [0.1, 0.1, 0.95],
    "great": [0.7, 0.2, 0.1],
    "bad": [0.2, 0.8, 0.1],
}


def encode(text):
    tokens = text.lower().split()
    valid = [word_vectors[token] for token in tokens if token in word_vectors]
    dim = len(valid[0])
    return [sum(vec[i] for vec in valid) / len(valid) for i in range(dim)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def softmax(scores):
    exps = [2.71828 ** s for s in scores]
    total = sum(exps)
    return [x / total for x in exps]


# 同一个底座，挂两个不同任务头
intent_head = {
    "refund_intent": [1.0, 0.9, 0.1],
    "password_intent": [0.1, 0.2, 1.0],
}

sentiment_head = {
    "positive": [1.0, 0.2, 0.0],
    "negative": [0.1, 1.0, 0.0],
}


def classify(vector, head):
    labels = list(head.keys())
    scores = [dot(vector, head[label]) for label in labels]
    probs = softmax(scores)
    best = max(zip(labels, probs), key=lambda x: x[1])
    return best, dict(zip(labels, [round(p, 3) for p in probs]))


text_a = "refund order"
text_b = "reset password"
text_c = "great refund"
text_d = "bad refund"

for text in [text_a, text_b]:
    vec = encode(text)
    best, probs = classify(vec, intent_head)
    print("intent:", text, "->", best, probs)

for text in [text_c, text_d]:
    vec = encode(text)
    best, probs = classify(vec, sentiment_head)
    print("sentiment:", text, "->", best, probs)
```

### 3.1 这段代码到底对应什么真实思路？

它对应的是预训练时代最重要的工作流之一：

1. 先有一个共享语言底座
2. 不同任务复用这个底座
3. 只在上面换一个 head 或做少量适配

这就是为什么一个预训练模型能被拿来做很多任务。

### 3.2 为什么这比“每个任务都重新学一遍词向量”强？

因为底座已经学到很多通用信息。  
下游任务不需要从零开始理解：

- `refund` 大概和售后有关
- `reset password` 大概和登录问题有关

它只需要在底座之上再做定向映射。

### 3.3 真实世界里“head”会是什么？

在真实模型里，它可能是：

- 一个分类层
- 一个生成头
- 一个检索投影层
- 一个 token-level 预测头

思路都是一样的：

- 底座共享
- 任务头分化

---

## 四、预训练模型大致有哪些路线？

### 4.1 Encoder-only：更偏理解和表示

代表：

- BERT

这类模型通常更适合：

- 分类
- 抽取
- 匹配
- 检索编码

### 4.2 Decoder-only：更偏生成

代表：

- GPT
- LLaMA
- Qwen

这类模型通常更适合：

- 对话
- 写作
- 代码生成
- 开放式补全

### 4.3 Encoder-Decoder：更适合“输入到输出”任务

代表：

- T5
- BART

这类模型天然适合：

- 摘要
- 翻译
- 改写
- 问答生成

---

## 五、预训练模型之后，还能怎么适配任务？

### 5.1 线性探针 / 任务头微调

最轻的一种方式是：

- 冻结底座
- 只训练顶层 head

这在小任务里很常见。

### 5.2 全量微调

让整个模型一起更新。  
优点是灵活，缺点是成本高。

### 5.3 参数高效微调

例如：

- LoRA
- Adapter

这是大模型时代非常重要的路线，  
因为它让“在大底座上适配任务”的门槛下降了很多。

### 5.4 Prompt 和 RAG

并不是所有任务都要真的改模型参数。  
很多问题也可以通过：

- Prompt
- RAG
- 工具调用

来解决。

所以预训练模型的价值，不只是“给你一个可微调模型”，  
还是“给你一个可复用底座”。

---

## 六、最容易踩的误区

### 6.1 误区一：预训练模型什么都会

它有强底座，但不代表：

- 知识永远最新
- 行为一定稳定
- 一上来就完美适配你的业务

### 6.2 误区二：用了预训练模型，就不用关心数据了

不对。  
无论是微调还是评估，数据质量仍然决定最后效果。

### 6.3 误区三：只要模型大，就一定比小模型更适合当前任务

有时任务很简单，  
或者成本非常敏感，  
大模型未必是最优解。

---

## 小结

这节最重要的不是记住多少模型名，  
而是建立一个现代 NLP 的核心判断：

> **预训练模型的价值，在于先用大语料学到通用语言底座，再把这个底座迁移到很多不同任务上。**

一旦这条主线建立起来，  
你后面再学：

- 微调
- Prompt
- RAG
- Agent

都会更知道自己是在“利用已有底座”，而不是每次从零开始。

---

## 练习

1. 用自己的话解释：为什么预训练模型能被多个不同任务复用？
2. 参考示例，再给共享底座加一个新的任务头，例如“主题分类”。
3. 为什么说预训练模型给的是底座，不是自动解决所有任务的万能按钮？
4. 想一想：如果你的任务数据很少，预训练模型相比从零训练最大的优势是什么？
