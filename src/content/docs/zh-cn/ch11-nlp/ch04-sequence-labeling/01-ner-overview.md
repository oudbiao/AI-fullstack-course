---
title: "11.4.2 序列标注任务"
description: "从“整句一个标签”和“每个 token 一个标签”的差别讲起，理解序列标注为什么是信息抽取任务的重要基础。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "sequence labeling, token classification, NER, BIO, span extraction, NLP"
---
![BIO 标签到实体恢复图](/img/course/bio-ner-recovery.webp)

:::tip[本节定位]
文本分类的输出通常是：

- 整句一个标签

而序列标注的输出更细：

- 每个 token 一个标签

这一步非常关键，因为它把 NLP 从“整句判断”推进到了：

> **在句子内部定位具体信息。**

也正是从这里开始，我们才更自然地走向命名实体识别、信息抽取、槽位填充这类任务。
:::
## 学习目标

- 理解序列标注和整句分类的根本区别
- 理解 BIO / BIOES 这类标签体系为什么常用
- 通过可运行示例理解 token 级标注过程
- 建立序列标注和信息抽取任务之间的联系

---

## 一、序列标注到底在解决什么问题？

### 它不只是判断“这句话是什么”，而是判断“这句话里哪一段是什么”

例如句子：

- “张三在北京大学工作”

如果做文本分类，也许只会输出：

- 这是一个关于人物与地点的句子

但序列标注更关心：

- `张三` 是人名
- `北京大学` 是机构名

### 为什么这很重要？

因为很多真实业务并不满足于整句理解。
它们更关心：

- 人名
- 地址
- 机构名
- 金额
- 时间

这些具体片段的位置和边界。

### 一个类比

文本分类像给整篇文章贴标签。
序列标注像拿荧光笔在句子里圈重点。

---

## 二、为什么输出通常要按 token 来标？

### 因为实体是连续片段

很多要抽的信息不是单个词，而是一段连续 span。
例如：

- `上海交通大学`
- `2025年6月1日`

### 用 token 级标签可以表达边界

这就是为什么常见标签体系不是简单地写：

- PERSON
- LOCATION

而是会写：

- `B-PER`
- `I-PER`
- `O`

### BIO 的直觉

- `B-`：实体开始
- `I-`：实体内部
- `O`：不属于任何实体

这样系统就能更明确地区分：

- 一个实体从哪里开始
- 到哪里结束

---

## 三、先跑一个最小 BIO 标注示例

```python
tokens = ["张三", "在", "北京", "大学", "工作"]
tags = ["B-PER", "O", "B-ORG", "I-ORG", "O"]

for tok, tag in zip(tokens, tags):
    print(tok, tag)
```

预期输出：

```text
张三 B-PER
在 O
北京 B-ORG
大学 I-ORG
工作 O
```

token 列表和标签列表长度必须一致。做任何序列标注数据集时，第一件事就是先确认这种一一对齐没有错。

### 这个例子最核心的地方是什么？

它让你看到：

- 序列输入
- 对应的序列输出

这就是序列标注最本质的形式：

> **输入一串 token，输出同样长度的一串标签。**

### 为什么 `北京 大学` 会被标成 `B-ORG / I-ORG`？

因为这里想表达的是：

- 这是同一个连续实体

而不是两个分开的实体。

---

## 四、从标签序列还原实体

下面这个例子会把 token + BIO 标签恢复成实体片段。

```python
tokens = ["张三", "在", "北京", "大学", "工作"]
tags = ["B-PER", "O", "B-ORG", "I-ORG", "O"]


def decode_entities(tokens, tags):
    entities = []
    current_tokens = []
    current_type = None

    for token, tag in zip(tokens, tags):
        if tag == "O":
            if current_tokens:
                entities.append(("".join(current_tokens), current_type))
                current_tokens = []
                current_type = None
            continue

        prefix, entity_type = tag.split("-", 1)

        if prefix == "B":
            if current_tokens:
                entities.append(("".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = entity_type
        elif prefix == "I" and current_type == entity_type:
            current_tokens.append(token)
        else:
            # 标签不合法时，简单切断并重开
            if current_tokens:
                entities.append(("".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = entity_type

    if current_tokens:
        entities.append(("".join(current_tokens), current_type))

    return entities


print(decode_entities(tokens, tags))
```

预期输出：

```text
[('张三', 'PER'), ('北京大学', 'ORG')]
```

这一步把 token 级标签还原成项目真正要用的结果：实体文本和实体类型。

### 这段代码为什么很重要？

因为它把“标注任务”和“抽取结果”连起来了。
真实系统里我们真正关心的通常不是标签本身，而是：

- 实体 span
- 实体类型

### 为什么要关心非法标签序列？

序列标注不是每个 token 独立猜标签就结束。模型可能输出这种不合法或很难解释的序列：

```text
tokens: 张三 在 北京 大学 工作
tags:   I-PER O  I-ORG I-ORG O
```

问题是：`I-PER` 前面没有 `B-PER`，`I-ORG` 前面也没有清楚的实体开始。这样的输出会让实体边界变得模糊。

| 问题 | 为什么影响抽取 |
|---|---|
| `I-*` 没有对应开头 | 不知道实体从哪里开始 |
| 相邻实体类型突然切换 | 可能把两个实体粘在一起或切坏 |
| token 和 tag 长度不一致 | 无法一一还原 span |

这也是为什么后面会出现 CRF 或解码约束：它们不只是为了分数更高，也是在减少“不像合法实体序列”的输出。

---

## 五、序列标注和信息抽取是什么关系？

### NER 是典型序列标注任务

最经典的就是：

- 命名实体识别

### 但它不只用于 NER

还可以做：

- 槽位填充
- 关键词抽取
- 事件触发词定位

### 所以它是“信息抽取的底层技能”

很多抽取系统后面会更复杂，
但最基础的一步常常仍然是：

- 先把关键 span 标出来

---

## 六、最容易踩的坑

### 误区一：把序列标注当成普通分类

它和整句分类最大的差别就在于：

- 输出是对齐序列

### 误区二：只看标签，不看边界恢复

真实系统更关心最终抽出的实体片段，
不是标签表本身。

### 误区三：标签体系随便定

如果标签设计混乱，模型和评估都会一起乱。

---

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
模式：实体类型、BIO 标签，或序列标注规则
预测：词级标签和提取的片段
指标：实体精确率/召回率/F1 和边界情况
失败检查：跨度边界、嵌套实体、未知词或标注不一致
期望产出：金标与预测 span 对照表，至少包含一个漏判
```

## 小结

这节最重要的是建立一个判断：

> **序列标注的核心，是对输入序列中的每个 token 做标签判断，从而恢复出句子内部的关键片段与边界。**

只要这个直觉稳住了，后面学 NER、BiLSTM+CRF 和信息抽取项目时就会顺很多。

---

## 练习

1. 把示例再加一个时间实体，例如 `2025年`，自己写一组 BIO 标签。
2. 为什么说 BIO 标签体系的关键作用是表达实体边界？
3. 用自己的话解释：序列标注和文本分类最大的区别是什么？
4. 想一想：如果标签序列里出现不合法的 `I-XXX`，系统该怎么处理更稳？

<details>
<summary>参考实现与讲解</summary>

1. 如果 `2025` 是单 token 时间实体，标成 `B-TIME`；只有跨多个 token 时才用 `B-TIME I-TIME ...`。
2. BIO 的核心作用是表达边界：哪个 token 开始实体，后面哪些 token 继续同一个实体。
3. 序列标注是每个 token 输出一个标签，文本分类是整段文本输出一个标签。
4. 非法 `I-XXX` 应通过后处理修复或拒绝，记录为错误，并回溯训练标签或解码规则。

</details>
