---
title: "4.2 BiLSTM + CRF"
sidebar_position: 11
description: "从“先看上下文，再保证标签序列合法”讲起，理解 BiLSTM + CRF 为什么长期是序列标注里的经典组合。"
keywords: [BiLSTM, CRF, sequence labeling, NER, transition constraints, decoding]
---

# BiLSTM + CRF

:::tip 本节定位
如果只用普通分类器给每个 token 独立打标签，  
很容易出现一种问题：

- 单个 token 看着合理
- 整体标签序列却不合理

例如：

- 直接出现 `I-PER`，前面却没有 `B-PER`

BiLSTM + CRF 这个经典组合，解决的就是：

> **既让模型看到上下文，又让最终标签序列更符合结构约束。**
:::

## 学习目标

- 理解 BiLSTM 和 CRF 各自负责什么
- 理解为什么 token 独立分类不够
- 通过可运行示例理解标签转移约束
- 理解这个经典结构为什么在 NER 时代很常见

---

## 一、为什么“每个 token 单独分类”不够？

### 1.1 局部最优不等于全局合理

假设模型逐 token 预测：

- `张三` -> `I-PER`
- `在` -> `O`
- `北京` -> `B-LOC`

其中第一个标签就很奇怪：

- `I-PER` 不应该凭空开头

这说明只看单点分类分数，  
并不能保证整条标签链是合法的。

### 1.2 序列标注本质上是结构化预测

输出不是一堆互相独立的标签，  
而是一整条标签序列。

所以模型应该同时考虑：

- 当前 token 的局部特征
- 相邻标签之间是否合理

---

## 二、BiLSTM 和 CRF 分别在做什么？

### 2.1 BiLSTM：给每个 token 一个上下文化表示

BiLSTM 的作用可以先粗略理解成：

- 从左到右看
- 从右到左也看
- 最后把两个方向的上下文信息合起来

这样每个 token 的表示就不只是它自己，  
还带着周围上下文。

### 2.2 CRF：在标签层做全局约束

CRF 更关心的是：

- 某个标签后面接另一个标签是否合理

例如：

- `B-PER -> I-PER` 很合理
- `O -> I-PER` 往往不合理

所以 CRF 不是替代上下文编码，  
而是在标签解码层再做一层结构化优化。

### 2.3 组合起来的意义

可以先记成：

- BiLSTM 负责“看上下文”
- CRF 负责“保序列合法”

这就是它为什么长期是 NER 里的经典组合。

---

## 三、先跑一个最小标签转移打分示例

下面这个例子不会训练真正的 BiLSTM+CRF，  
但它会非常直观地展示：

- 同一组 token 发射分数
- 在不同标签路径下，总分为什么不同

```python
tags = ["B-PER", "I-PER", "O"]

emission_scores = [
    {"B-PER": 2.2, "I-PER": 0.4, "O": 0.3},  # 张
    {"B-PER": 0.2, "I-PER": 1.9, "O": 0.4},  # 三
    {"B-PER": 0.1, "I-PER": 0.2, "O": 2.0},  # 在
]

transition_scores = {
    ("START", "B-PER"): 0.8,
    ("START", "I-PER"): -5.0,
    ("START", "O"): 0.3,
    ("B-PER", "I-PER"): 1.2,
    ("B-PER", "O"): 0.4,
    ("I-PER", "I-PER"): 1.1,
    ("I-PER", "O"): 0.5,
    ("O", "B-PER"): 0.6,
    ("O", "I-PER"): -4.0,
    ("O", "O"): 0.7,
}


def score_path(path):
    total = 0.0
    prev = "START"
    for i, tag in enumerate(path):
        total += transition_scores.get((prev, tag), -10.0)
        total += emission_scores[i][tag]
        prev = tag
    return round(total, 3)


path_good = ["B-PER", "I-PER", "O"]
path_bad = ["I-PER", "I-PER", "O"]

print("good path score:", score_path(path_good))
print("bad path score :", score_path(path_bad))
```

### 3.1 这个例子最关键的启发是什么？

它说明了 CRF 为什么有意义：

- `bad` 路径里单个 token 局部分数不一定很差
- 但整条路径因为转移不合理，会被整体拉低

### 3.2 为什么 `START -> I-PER` 要给很低分？

因为从序列开始直接进 `I-PER` 通常是不合法的。  
这正是 CRF 这种结构化层的价值：

- 把标签体系知识显式编码进解码过程

---

## 四、BiLSTM + CRF 为什么长期这么常见？

### 4.1 因为它刚好覆盖了两个核心难点

1. token 上下文理解
2. 标签链结构约束

### 4.2 为什么它在 BERT 时代前后都还经常出现？

即使后来编码器变强，  
CRF 层在某些任务里仍然有价值，  
因为它解决的是：

- 标签结构一致性

而不是 token 表示本身。

---

## 五、最容易踩的坑

### 5.1 误区一：BiLSTM + CRF 是一个整体黑盒

更好的理解是：

- BiLSTM 看上下文
- CRF 管标签序列

### 5.2 误区二：只要有 CRF 就一定更强

不一定。  
如果数据或任务本身很简单，CRF 不一定显著提升。

### 5.3 误区三：只看单点准确率

序列标注更该看：

- 实体级结果
- 标签链是否合理

---

## 小结

这节最重要的是建立一个清楚判断：

> **BiLSTM + CRF 的经典性，来自它同时解决了“每个 token 看上下文”和“整条标签链要合法”这两个问题。**

只要这个判断清楚了，你以后再看 BERT+CRF、Transformer+CRF 也会更顺。

---

## 练习

1. 自己再构造一条不合法路径，看看总分会不会明显更低。
2. 为什么说 CRF 关心的是“标签之间的关系”，而不是 token 本身的语义？
3. 想一想：如果一个任务对标签序列合法性要求很强，CRF 为什么会更有价值？
4. 用自己的话解释：BiLSTM 和 CRF 分别负责哪一层问题？
