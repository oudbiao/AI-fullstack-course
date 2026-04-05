---
title: "4.3 预训练方法"
sidebar_position: 13
description: "把 Causal LM、Masked LM、Span Corruption 等目标放在同一张图里，理解不同预训练方法究竟在让模型学什么。"
keywords: [causal language modeling, masked language modeling, span corruption, pretraining objectives, BERT, GPT, T5]
---

# 预训练方法

:::tip 本节定位
预训练方法本质上是在回答一句非常根本的话：

> **训练时，模型到底被要求完成什么任务？**

同样是一堆文本，如果训练目标不同，  
最后学出来的能力也会很不一样。

这就是为什么：

- BERT 走向理解任务
- GPT 走向生成任务
- T5 走向统一文本到文本

这节课要做的，就是把这些目标真正拆开。
:::

## 学习目标

- 理解不同预训练目标分别在教模型什么能力
- 区分 Causal LM、Masked LM、Span Corruption 的核心差异
- 通过一个可运行示例理解同一条文本如何被改造成不同训练样本
- 建立“任务目标和后续能力为什么强相关”的直觉

---

## 一、为什么预训练目标会决定模型路线？

### 1.1 因为模型会优先学会“训练里反复被要求做的事”

如果训练时模型不断被要求：

- 根据前文预测后文

它自然会更擅长：

- 续写
- 生成

如果训练时模型不断被要求：

- 根据左右文恢复被遮掉的 token

它自然更容易学会：

- 双向理解
- 语义补全

所以预训练目标不是表面任务，  
而是模型能力的方向盘。

### 1.2 一个类比：考试题型会塑造学习方式

你可以把模型想成学生。

- 如果天天考填空，它会练填空
- 如果天天考作文，它会练续写
- 如果天天考改写和摘要，它会练输入到输出映射

模型也是一样。

---

## 二、三条最重要的预训练路线

### 2.1 Causal Language Modeling：根据过去预测未来

这是 GPT 一系最经典的目标。

形式上很简单：

- 输入前面的 token
- 预测下一个 token

它的好处是：

- 训练目标和生成任务天然一致

也就是说，训练时模型不能看未来，  
推理时模型也不能看未来，  
两者没有错位。

### 2.2 Masked Language Modeling：根据上下文补空

这是 BERT 一系的经典目标。

做法是：

- 把输入里部分 token 遮掉
- 让模型根据左右文把它们补回来

这种目标非常适合双向建模，  
所以它更擅长：

- 理解
- 表示学习
- 分类和抽取类任务

但它天然不如 Causal LM 那么适合自由生成。

### 2.3 Span Corruption / Denoising：不是遮一个词，而是遮一段

T5 / BART 一类模型常用更一般化的去噪目标：

- 不是只 mask 一个 token
- 而是 mask 一整段 span
- 然后让模型恢复这段内容

这会更贴近：

- 摘要
- 改写
- 翻译
- 文本到文本转换

---

## 三、先用同一条文本构造三种训练样本

这一段代码的目标很直接：

- 给同一条句子
- 分别生成 Causal LM、Masked LM、Span Corruption 三种训练样本

这样你能非常直观地看到：

- “目标不同”到底意味着什么

```python
tokens = "transformer models learn patterns from large text corpora".split()


def build_causal_example(tokens):
    inputs = tokens[:-1]
    labels = tokens[1:]
    return inputs, labels


def build_masked_example(tokens, mask_positions):
    masked = tokens[:]
    labels = {}
    for pos in mask_positions:
        labels[pos] = masked[pos]
        masked[pos] = "[MASK]"
    return masked, labels


def build_span_corruption(tokens, start, end):
    corrupted_input = tokens[:start] + ["<extra_id_0>"] + tokens[end:]
    target = ["<extra_id_0>"] + tokens[start:end] + ["<extra_id_1>"]
    return corrupted_input, target


causal_inputs, causal_labels = build_causal_example(tokens)
masked_inputs, masked_labels = build_masked_example(tokens, mask_positions=[2, 5])
span_inputs, span_target = build_span_corruption(tokens, start=2, end=5)

print("causal inputs :", causal_inputs)
print("causal labels :", causal_labels)
print()
print("masked inputs :", masked_inputs)
print("masked labels :", masked_labels)
print()
print("span inputs   :", span_inputs)
print("span target   :", span_target)
```

### 3.1 这段代码最该看哪里？

先看这三件事：

1. 输入被改成了什么样
2. 标签到底让模型学什么
3. 为什么同一句话会被组织成完全不同的训练任务

如果这一点看懂了，  
你就会明白为什么：

- GPT、BERT、T5 最后能力画像不同

### 3.2 Causal LM 的标签为什么是右移一位？

因为它在做的就是：

- 给前文，猜下一个 token

所以最自然的训练数据组织就是：

- 输入：`x_1 ... x_{t-1}`
- 标签：`x_2 ... x_t`

### 3.3 Span Corruption 为什么常被看得更“通用”？

因为它比单点 mask 更接近真实文本变换。  
模型不仅要恢复一个词，  
而是要补回一段缺失内容。

这会让它更自然地走向：

- text-to-text

这也是 T5 路线很重要的原因。

---

## 四、这些目标分别更擅长什么？

### 4.1 Causal LM：生成、续写、对话

这类目标和后续生成任务高度一致，  
所以特别适合：

- 聊天
- 写作
- 代码补全
- 长文本续写

### 4.2 Masked LM：表示学习和理解

因为模型能同时看到左右上下文，  
所以很适合：

- 分类
- 检索编码
- 语义匹配
- 抽取任务

### 4.3 Span Corruption：输入到输出映射

如果你想要模型自然地做：

- 摘要
- 改写
- 翻译
- 问答生成

这类去噪和 seq2seq 目标会更顺手。

---

## 五、预训练目标不是独立存在的，它和架构绑在一起

### 5.1 为什么 Decoder-only 常配 Causal LM？

因为两者完全一致：

- decoder 只能看过去
- causal LM 也要求只能看过去

训练和生成闭环非常自然。

### 5.2 为什么 Encoder-only 常配 Masked LM？

因为 encoder 擅长双向建模。  
既然它能看全句，就很适合做：

- 被 mask 位置的恢复

### 5.3 为什么 Encoder-Decoder 常配去噪目标？

因为这类结构天然适合：

- 输入一段东西
- 输出另一段东西

所以 span corruption、denoising、文本到文本训练都很搭。

---

## 六、除了经典目标，还会有什么变化？

### 6.1 Prefix LM：部分双向、部分因果

有些方法会让输入前半段可以双向看，  
但生成段仍然保持因果约束。

这类目标适合：

- 既要读上下文
- 又要生成续写

### 6.2 多模态预训练：输入不只是一串文本

如果输入同时包含：

- 图像
- 音频
- 视频

那目标就会变成：

- 跨模态对齐
- 图文生成
- 多模态理解

虽然形式更复杂，但核心仍然一样：

- 训练目标决定模型优先学什么

### 6.3 自监督目标不代表完全“无偏”

即使标签是自动构造的，  
目标函数本身也在给模型施加偏好。

例如：

- 更偏生成
- 更偏理解
- 更偏结构恢复

这也是为什么预训练目标本身也是设计选择。

---

## 七、最容易踩的误区

### 7.1 误区一：预训练目标只是前期细节，后面微调会解决一切

不对。  
预训练目标会给模型打下长期能力偏向。

### 7.2 误区二：Masked LM 比 Causal LM 更高级，或者反过来

两者不是等级关系，  
而是针对不同路线的设计。

### 7.3 误区三：只记名字，不看标签长什么样

真正的理解是：

- 输入怎么组织
- 标签怎么构造
- 模型被要求学什么

---

## 小结

这一节最重要的，不是背下 `CLM / MLM / Span Corruption` 这几个缩写，  
而是抓住一条主线：

> **预训练目标本质上是在规定模型每天反复练什么，而模型最后最擅长的，通常就是它被反复练得最多的那类能力。**

这条主线一旦建立起来，  
你看后面的架构选择、微调方式和任务迁移，就会更顺。

---

## 练习

1. 把示例中的句子换成你自己的句子，再分别生成三种训练样本。
2. 用自己的话解释：为什么 Causal LM 更适合开放式生成？
3. 为什么说 Masked LM 更像“补空题”，而不是“续写题”？
4. 想一想：如果你的目标是做一个强摘要模型，你会更偏向哪类预训练目标？为什么？
