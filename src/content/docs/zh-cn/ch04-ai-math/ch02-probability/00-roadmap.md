---
title: "4.2.1 概率路线图：给 AI 一套描述不确定性的语言"
description: "面向 AI 的紧凑版概率统计路线图：概率、分布、推断、信息论和损失函数。"
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "概率指南, 统计指南, 概率分布, 贝叶斯, MLE, 信息论"
---
概率和统计解释模型为什么会输出置信度、数据为什么会波动，以及训练为什么使用 loss，而不只是对/错标签。

## 先看地图

![概率统计学习地图](/img/course/ch04-probability-roadmap-vertical.webp)

本小章流向是：

![概率统计章节流程](/img/course/ch04-probability-chapter-flow.webp)

| 术语 | 先问的问题 |
|---|---|
| 概率 | 这件事有多可能发生？ |
| 分布 | 很多随机结果整体长什么样？ |
| 推断 | 看见数据后能得出什么结论？ |
| 熵 | 结果有多不确定？ |
| 交叉熵 | 预测的概率分布错得有多远？ |
| KL 散度 | 两个分布有多不同？ |

:::tip[公式前先问什么]
看到概率公式时，先问：随机对象是什么？条件是什么？我们是在描述数据本身、更新信念，还是衡量预测分布错了多少？这样再看 Bayes、交叉熵或 KL，就不容易变成纯符号记忆。
:::

## 跑最小闭环

创建 `probability_first_loop.py`。它只用 Python 标准库。

```python
import math

labels = [1, 0, 1, 1]
predicted_probs = [0.9, 0.2, 0.6, 0.8]

losses = []
for y, p in zip(labels, predicted_probs):
    loss = -(y * math.log(p) + (1 - y) * math.log(1 - p))
    losses.append(loss)

cross_entropy = sum(losses) / len(losses)
print("cross_entropy:", round(cross_entropy, 3))
print("predicted_probs:", predicted_probs)
```

预期输出：

```text
cross_entropy: 0.266
predicted_probs: [0.9, 0.2, 0.6, 0.8]
```

交叉熵越低，说明预测概率越接近标签。这就是概率和模型训练直接相连的地方。

## 按这个顺序学

| 顺序 | 阅读 | 先抓住什么 |
|---|---|---|
| 1 | [4.2.2 概率基础](/zh-cn/ch04-ai-math/ch02-probability/01-probability-basics/) | 事件、条件概率、贝叶斯更新 |
| 2 | [4.2.3 概率分布](/zh-cn/ch04-ai-math/ch02-probability/02-distributions/) | 伯努利、二项、正态分布 |
| 3 | [4.2.4 统计推断](/zh-cn/ch04-ai-math/ch02-probability/03-statistical-inference/) | MLE、MAP、置信度、A/B 测试 |
| 4 | [4.2.5 信息论](/zh-cn/ch04-ai-math/ch02-probability/04-information-theory/) | 熵、交叉熵、KL 散度 |
| 5 | [4.2.6 历史基础](/zh-cn/ch04-ai-math/ch02-probability/05-history-foundations/) | 贝叶斯、Fisher、Shannon、EM 的位置 |

## 通过标准

能说清一个概率术语在衡量哪种不确定性，并能解释分类器输出 `0.93` 为什么有用但不是绝对真相，就算通过。


<details>
<summary>检查思路与讲解</summary>

- 概率路线通过的标志是：你能从单次事件，走到重复采样估计，再走到条件更新。
- 证据至少保留一次模拟、一个分布图、一个 MLE/MAP 估计，以及一个熵或交叉熵计算。
- 关键习惯是说清假设：先验比例、独立性、样本量、零假设或预测概率。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
随机过程：事件、分布、样本、似然、熵，或 Bayes 更新
模拟或公式：用来让不确定性可见的代码或公式
输出：概率、样本统计量、区间、熵，或更新后的信念
失败检查：基率混淆、p 值误用、样本偏差或把概率和确定性混为一谈
期望产出：数值结果加通俗解释
```
