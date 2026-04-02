---
title: "1.2 文本预处理"
sidebar_position: 2
description: "从清洗、标准化、分词到停用词处理，掌握最基础也最常用的文本预处理方法。"
keywords: [文本预处理, 分词, stopwords, 标准化, regex, NLP]
---

# 文本预处理

## 学习目标

完成本节后，你将能够：

- 理解文本预处理到底在解决什么问题
- 掌握清洗、标准化、分词、停用词处理等常见步骤
- 写出一个可直接运行的预处理函数
- 明白“不是所有文本都该洗得越干净越好”

---

## 一、为什么文本要预处理？

原始文本往往很“脏”：

- 大小写不统一
- 标点很多
- 有链接、数字、表情、空格噪声
- 同一个意思可能有很多写法

你可以把文本预处理想成“洗菜”：

- 不洗，模型很难直接下锅
- 洗过头，又可能把有用信息一起洗掉

所以预处理不是越多越好，而是要**围绕任务**做。

---

## 二、常见预处理步骤

| 步骤 | 作用 |
|---|---|
| 小写化 | 统一大小写 |
| 去标点 | 减少噪声 |
| 去多余空格 | 格式更整洁 |
| 分词 | 切成更小单位 |
| 去停用词 | 去掉“的、了、and、the”这类高频低信息词 |
| 数字 / 链接处理 | 统一特殊内容 |

---

## 三、一个最小可运行的预处理函数

下面用英文句子演示，因为英文分词更容易用标准库直接完成。  
思路对中文同样适用，只是中文通常要借助专业分词工具。

```python
import re

stopwords = {"the", "is", "a", "an", "and", "to", "of", "in"}

def preprocess(text):
    text = text.lower()                          # 1. 小写化
    text = re.sub(r"http\\S+", " ", text)        # 2. 去链接
    text = re.sub(r"[^a-z0-9\\s]", " ", text)    # 3. 去标点
    text = re.sub(r"\\s+", " ", text).strip()    # 4. 合并多余空格

    tokens = text.split()                        # 5. 简单分词
    tokens = [t for t in tokens if t not in stopwords]  # 6. 去停用词
    return tokens

sample = "The movie is AMAZING, and the ending is full of surprises!"
print(preprocess(sample))
```

输出会类似：

```python
['movie', 'amazing', 'ending', 'full', 'surprises']
```

---

## 四、为什么要做小写化？

在很多英文任务里：

- `Apple`
- `apple`
- `APPLE`

可能本来想表达的是同一个词。  
如果不统一，词表会被无意义地拆散。

但注意：

### 小写化不是永远都对

比如你在做：

- 命名实体识别
- 品牌名识别
- 法律文书解析

大小写本身可能就是有信息的。

所以预处理永远要和任务绑定看。

---

## 五、分词：把句子拆成更小单位

分词是 NLP 最基础的动作之一。

### 英文

英文天然有空格，简单任务可以直接 `split()`：

```python
text = "I love learning NLP"
print(text.split())
```

### 中文

中文没有天然空格，分词更复杂。

比如：

> “我喜欢自然语言处理”

可能会被切成：

- 我 / 喜欢 / 自然语言处理
- 我 / 喜欢 / 自然 / 语言 / 处理

所以中文项目通常会用更专业的分词工具。  
但在入门阶段，你先理解“文本需要被切成处理单元”这个思想更重要。

---

## 六、停用词该不该删？

停用词是那些出现频率很高，但通常信息量不大的词，比如：

- the
- is
- and

删掉它们的好处：

- 降低噪声
- 减少特征维度

但也有风险。

比如情感分析里：

> “not good”

如果你把 `not` 当停用词删掉，意思就完全反了。

所以要记住：

> **停用词不是机械删除，而是任务驱动。**

---

## 七、一个完整的预处理小练习

下面对几条评论做统一处理：

```python
import re

stopwords = {"the", "is", "a", "an", "and", "to", "of", "in", "this"}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\\S+", " ", text)
    text = re.sub(r"[^a-z0-9\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords]
    return tokens

texts = [
    "This course is easy to follow!",
    "The examples are clear and practical.",
    "I love the hands-on exercises in this class."
]

for t in texts:
    print("原文:", t)
    print("处理后:", preprocess(t))
    print("-" * 40)
```

---

## 八、什么时候预处理要“轻一点”？

在传统机器学习时代，预处理通常做得比较重。  
但到了预训练模型和大模型时代，预处理常常要更谨慎。

比如：

- BERT / GPT 这类模型通常自带 tokenizer
- 你如果先做过度清洗，可能反而破坏原始信息

所以：

### 传统模型

- 往往更依赖人工预处理

### 预训练模型 / 大模型

- 往往更依赖官方 tokenizer
- 更强调保留原始文本结构

---

## 九、初学者常见误区

### 1. 觉得预处理越多越高级

不对。  
删太多信息，效果可能更差。

### 2. 不区分任务就套同一套清洗规则

文本分类、检索、NER、RAG，预处理策略都可能不同。

### 3. 中文也直接 `split()`

一般不够。  
中文分词通常需要专门工具或子词 tokenizer。

---

## 小结

文本预处理的核心不是“洗干净”，而是：

> **把文本整理成更适合当前任务处理的形式。**

下一节我们会继续往前走，解决另一个关键问题：  
**文本怎么表示成数字？**

---

## 练习

1. 在 `preprocess()` 里加上数字替换逻辑，把所有数字替换成 `<num>`。
2. 试着把 `not` 加进停用词，再观察情感句子的含义会不会被破坏。
3. 自己找 5 条短评论，跑一遍预处理，看看哪些信息被保留、哪些被删掉了。
