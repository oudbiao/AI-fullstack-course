---
title: "5.3 机器翻译实战【选修】"
sidebar_position: 15
description: "围绕一个最小翻译任务，走完数据对、基线系统、错误分析和下一步升级方向这条翻译项目闭环。"
keywords: [machine translation, seq2seq, translation project, alignment, error analysis]
---

# 机器翻译实战【选修】

:::tip 本节定位
翻译是 Seq2Seq 最经典的任务。  
它很适合用来练习一整条“输入文本 -> 输出文本”的项目闭环。

这节课不会硬上大模型训练，  
而是先把最关键的项目结构做清楚：

- 数据对长什么样
- 最小翻译系统怎么跑
- 错误应该怎么看
:::

## 学习目标

- 理解一个翻译项目的最小组成
- 学会从平行语料对组织数据
- 通过可运行示例建立最小翻译基线
- 学会做简单的翻译错误分析

---

## 一、机器翻译任务最核心的输入输出是什么？

### 1.1 输入

- 源语言句子

### 1.2 输出

- 目标语言句子

### 1.3 为什么这类任务特别适合 Seq2Seq？

因为：

- 输入和输出都不是固定长度
- 两边存在顺序和语义映射

这正是 Seq2Seq 的典型场景。

---

## 二、先看一个最小平行语料集

```python
parallel_data = [
    ("hello", "你好"),
    ("world", "世界"),
    ("i love ai", "我 爱 AI"),
    ("study hard", "努力 学习"),
]

for src, tgt in parallel_data:
    print(src, "->", tgt)
```

### 2.1 为什么平行语料是翻译项目的基础？

因为模型最终需要学习的是：

- 源语言 -> 目标语言

没有这类对齐数据，翻译任务就无从开始。

---

## 三、先跑一个最小翻译基线

```python
parallel_data = [
    ("hello", "你好"),
    ("world", "世界"),
    ("i", "我"),
    ("love", "爱"),
    ("study", "学习"),
]

phrase_table = {src: tgt for src, tgt in parallel_data}


def translate(sentence):
    tokens = sentence.split()
    output = [phrase_table.get(tok, "<unk>") for tok in tokens]
    return " ".join(output)


tests = [
    "hello world",
    "i love study",
    "love ai",
]

for sent in tests:
    print(sent, "->", translate(sent))
```

### 3.1 这个例子为什么仍然值得做？

因为它先帮你抓住翻译项目最底层的形式：

- 数据对
- 映射规则
- 输出质量

### 3.2 它的局限也很明显

- 不会处理词序变化
- 不会处理多义词
- 遇到未知词就 `<unk>`

也正因为这些局限明显，  
你更容易理解为什么后面需要更强模型。

---

## 四、翻译项目该怎么做错误分析？

### 4.1 常见错误一：漏译

例如某个词直接没翻出来。

### 4.2 常见错误二：错译

例如一个词翻到了错误义项。

### 4.3 常见错误三：词序不自然

这是最小词典基线特别容易出现的问题。

### 4.4 一个极简错误检查

```python
gold = {
    "hello world": "你好 世界",
    "i love study": "我 爱 学习",
}

for src, expected in gold.items():
    pred = translate(src)
    print({
        "src": src,
        "pred": pred,
        "gold": expected,
        "match": pred == expected,
    })
```

---

## 五、从这个最小项目往后升级，可以怎么走？

### 5.1 加更多平行语料

### 5.2 引入注意力和神经 Seq2Seq

### 5.3 再进一步走向 Transformer

所以这个小项目的意义，不在于它本身够强，  
而在于它让你看清：

- 翻译项目的基本骨架

---

## 六、最常见误区

### 6.1 误区一：翻译就是查字典

真实翻译远比逐词替换复杂。

### 6.2 误区二：只看一两条漂亮样例

真正项目里更重要的是系统性错误分析。

### 6.3 误区三：一开始就想直接训很大模型

更稳的做法通常是先把数据和基线结构理清。

---

## 小结

这节最重要的是把翻译项目看成：

> **一个围绕平行语料、映射学习和错误分析展开的典型 Seq2Seq 项目。**

先把这条闭环走顺，后面升级模型时就不会只剩“换更大模型”一种思路。

---

## 练习

1. 自己再补 5 组词对，扩展这个小词典基线。
2. 为什么最小翻译基线特别容易出词序问题？
3. 想一想：什么错误是词典基线无论如何都很难解决的？
4. 如果你要升级这个项目，第一步会先补数据还是先换模型？为什么？
