---
title: "4.3 NER 实战"
sidebar_position: 12
description: "围绕一个简历信息抽取小项目，走完标签设计、数据组织、实体恢复和错误分析这条 NER 实战闭环。"
keywords: [NER, named entity recognition, information extraction, BIO, project, NLP]
---

# NER 实战

:::tip 本节定位
前两节已经把：

- 序列标注任务
- BiLSTM + CRF 的核心思想

讲清楚了。  
这一节我们把它放回项目里，做一个更像真实业务的练习：

> **从简历文本里抽取姓名、学校和技能。**

这类任务非常适合练 NER，因为它同时包含：

- 明确 span
- 明确类型
- 很多边界细节
:::

## 学习目标

- 学会定义一个最小 NER 项目边界
- 学会从 token 标签恢复实体
- 学会做实体级别错误分析
- 通过可运行示例建立信息抽取项目骨架

---

## 一、项目问题先要定义清楚

### 1.1 场景

输入：

- 一段简历或候选人简介文本

输出：

- 姓名
- 学校
- 技能

### 1.2 为什么这比“随便抽点实体”更适合练手？

因为它边界清楚：

- 类别数不多
- 实体类型明确
- 结果很容易做业务解释

### 1.3 第一个关键点不是模型，而是标签体系

例如：

- `张三` -> `B-NAME`
- `清华大学` -> `B-SCHOOL I-SCHOOL ...`
- `Python` -> `B-SKILL`

这一步一旦含糊，后面模型和评估都会一起乱。

---

## 二、先做一个可运行标注与解码闭环

下面这个示例会做三件事：

1. 准备一个小型样本
2. 把 BIO 标签解码成实体
3. 做简单的预测对比与错误分析

```python
samples = [
    {
        "tokens": ["张三", "毕业于", "清华大学", "，", "熟悉", "Python", "和", "PyTorch"],
        "gold_tags": ["B-NAME", "O", "B-SCHOOL", "O", "O", "B-SKILL", "O", "B-SKILL"],
        "pred_tags": ["B-NAME", "O", "B-SCHOOL", "O", "O", "B-SKILL", "O", "B-SKILL"],
    },
    {
        "tokens": ["李四", "来自", "北京大学", "，", "掌握", "Java"],
        "gold_tags": ["B-NAME", "O", "B-SCHOOL", "O", "O", "B-SKILL"],
        "pred_tags": ["B-NAME", "O", "O", "O", "O", "B-SKILL"],
    },
]


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
            if current_tokens:
                entities.append(("".join(current_tokens), current_type))
            current_tokens = [token]
            current_type = entity_type

    if current_tokens:
        entities.append(("".join(current_tokens), current_type))

    return entities


for sample in samples:
    gold_entities = decode_entities(sample["tokens"], sample["gold_tags"])
    pred_entities = decode_entities(sample["tokens"], sample["pred_tags"])

    print("tokens:", sample["tokens"])
    print("gold :", gold_entities)
    print("pred :", pred_entities)
    print("miss :", [x for x in gold_entities if x not in pred_entities])
    print()
```

### 2.1 这段代码为什么是“项目最小闭环”？

因为它已经包含了：

- 数据表示
- 预测结果
- 实体恢复
- 错误分析

这比只打印一串标签更接近真实项目形态。

### 2.2 为什么这里按实体比较，而不是只按 token 比较？

因为业务真正关心的通常是：

- 实体有没有抽出来
- 类型对不对

而不是某一个 token 单点是否打对。

---

## 三、NER 项目最该先看什么指标？

### 3.1 实体级 Precision / Recall / F1

这是最常见也最有意义的一组指标。

### 3.2 为什么 token accuracy 不够？

因为序列里往往很多都是：

- `O`

只看 token accuracy 很容易显得“很高”，  
但真正的实体抽取效果可能并不好。

### 3.3 一个极简实体召回例子

```python
def entity_recall(gold_entities, pred_entities):
    if not gold_entities:
        return 1.0
    hit = sum(entity in pred_entities for entity in gold_entities)
    return hit / len(gold_entities)


for sample in samples:
    gold_entities = decode_entities(sample["tokens"], sample["gold_tags"])
    pred_entities = decode_entities(sample["tokens"], sample["pred_tags"])
    print(entity_recall(gold_entities, pred_entities))
```

---

## 四、NER 项目最常见的失败点

### 4.1 实体边界错

例如学校名只抽了一半。

### 4.2 类型错

例如把技能识别成学校。

### 4.3 漏实体

例如样本 2 里把 `北京大学` 漏掉了。

### 4.4 为什么这很适合做错误分析？

因为 NER 的错误通常很具体，  
非常适合逐条看、逐类修。

---

## 五、真实项目下一步该怎么走？

### 5.1 扩充数据

尤其是：

- 长实体
- 稀有实体
- 容易混淆类型

### 5.2 从规则 / 经典模型升级到更强模型

例如：

- BiLSTM + CRF
- BERT token classification

### 5.3 加入后处理规则

很多业务项目里，  
合理的后处理规则能明显提升实体质量。

---

## 六、最常见误区

### 6.1 误区一：只看 token 级指标

NER 更该看实体级效果。

### 6.2 误区二：一开始就想覆盖所有实体类型

更稳的做法通常是：

- 先选 2~4 类核心实体做透

### 6.3 误区三：标签体系一开始不定清楚

标签边界不清，数据和评估都会一起发散。

---

## 小结

这节最重要的是建立一个实战习惯：

> **做 NER 项目时，先把实体类型、标签体系、实体恢复和实体级错误分析做扎实，再去追求更复杂模型。**

这样你留下的会是一个真正可解释、可改进的信息抽取项目，而不是只会跑训练脚本的半成品。

---

## 练习

1. 给示例再加一个 `ORG` 或 `TITLE` 实体类型，扩展样本。
2. 想一想：为什么 NER 项目更适合看实体级指标，而不是 token accuracy？
3. 如果系统经常把长学校名只抽一半，你会优先改数据、改模型，还是加后处理？为什么？
4. 你会如何把这个简历抽取项目进一步扩成作品集展示？
