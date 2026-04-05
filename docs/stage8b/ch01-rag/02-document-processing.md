---
title: "1.3 文档处理与向量化"
sidebar_position: 2
description: "从清洗、切块、重叠、元数据到简单向量化，理解 RAG 前处理链路为什么决定效果上限。"
keywords: [chunking, 文档切块, 向量化, metadata, RAG preprocessing]
---

# 文档处理与向量化

## 学习目标

完成本节后，你将能够：

- 理解为什么 RAG 效果很大程度取决于前处理
- 掌握文档清洗、切块、重叠和元数据的直觉
- 写出一个简单可运行的切块与检索示例
- 理解“向量化”到底在做什么

---

## 一、为什么 RAG 不是“文档直接塞进去”？

因为真实文档往往很长、很乱、很杂。

例如一份 PDF 可能包含：

- 页眉页脚
- 目录
- 空行
- 标题层级
- 表格
- 重复文本

如果你把它原样塞给模型，常见问题包括：

- 上下文太长，塞不下
- 重点埋在长文里，不容易被检索到
- 噪声太多，影响检索质量

所以文档处理其实是在做一件事：

> **把资料整理成模型更容易找到、也更容易利用的知识块。**

---

## 二、文档处理常见的 4 步

### 1. 清洗

去掉无关噪声，比如：

- 多余空格
- 页码
- 重复标题

### 2. 切块（Chunking）

把长文切成适合检索的小片段。

### 3. 加元数据

给每块附加信息，如：

- 来源文件
- 标题
- 页码
- 标签

### 4. 向量化

把文本块变成可做相似度检索的向量。

---

## 三、切块为什么这么重要？

切块大小很像“做笔记时一张卡片写多少内容”。

- 太大：一次装太多，检索不精准
- 太小：上下文不够，回答容易断裂

这没有唯一标准，但一定要围绕任务调。

类比一下：

> 做开卷考试笔记时，你不会把整本书粘成一张超大海报，也不会把每个字都剪成一张纸条。

---

## 四、一个最小可运行的切块示例

```python
import re

text = """
退款政策：
课程购买后 7 天内，如果学习进度低于 20%，可以申请退款。
超过 7 天后，不再支持无条件退款。

证书说明：
完成所有必修项目并通过结课测试后，可以获得结业证书。

学习顺序：
建议先学习 Python、数据分析、机器学习，再进入深度学习和大模型阶段。
""".strip()

def split_into_sentences(text):
    parts = re.split(r"[。！？\\n]+", text)
    return [p.strip() for p in parts if p.strip()]

sentences = split_into_sentences(text)
print("句子列表:")
for s in sentences:
    print("-", s)
```

如果句子已经比较短，你可以直接把句子当 chunk。  
但更多时候，我们会把几句组合成一个块。

---

## 五、带重叠的切块

为什么很多 RAG 系统会做 chunk overlap？

因为信息可能刚好卡在块边界上。  
加一点重叠，可以减少“上下文被切断”的概率。

```python
def chunk_sentences(sentences, chunk_size=2, overlap=1):
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + chunk_size
        chunk = "。".join(sentences[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        if chunk_size - overlap <= 0:
            raise ValueError("chunk_size 必须大于 overlap")
    return chunks

chunks = chunk_sentences(sentences, chunk_size=2, overlap=1)

print("切块结果:")
for i, chunk in enumerate(chunks):
    print(f"[chunk {i}] {chunk}")
```

---

## 六、元数据为什么重要？

很多新人只关注文本内容，忽略元数据。  
但元数据往往直接影响检索和展示体验。

一个 chunk 常见的元数据有：

- `source`: 来自哪个文件
- `section`: 属于哪一节
- `page`: 来自哪一页
- `tags`: 属于什么主题

比如：

```python
chunks_with_meta = [
    {
        "text": "课程购买后 7 天内，如果学习进度低于 20%，可以申请退款",
        "source": "course_policy.pdf",
        "section": "退款政策",
        "page": 3
    },
    {
        "text": "完成所有必修项目并通过结课测试后，可以获得结业证书",
        "source": "course_policy.pdf",
        "section": "证书说明",
        "page": 5
    }
]

for item in chunks_with_meta:
    print(item)
```

元数据的价值在于：

- 便于过滤
- 便于引用来源
- 便于后续 UI 展示

---

## 七、向量化到底在做什么？

向量化的核心，是把文本块映射到一个“语义空间”里。

这样查询和文档块都能变成向量，然后比较相似度。

为了保证代码直接能跑，我们先用一个极简的词袋向量来模拟这个过程。

```python
import math
import re
from collections import Counter

chunks = [
    "课程购买后 7 天内，如果学习进度低于 20%，可以申请退款",
    "完成所有必修项目并通过结课测试后，可以获得结业证书",
    "建议先学习 Python、数据分析、机器学习，再进入深度学习和大模型阶段"
]

def tokenize(text):
    return re.findall(r"[\\w\\u4e00-\\u9fff]+", text.lower())

vocab = sorted(set(token for chunk in chunks for token in tokenize(chunk)))
vocab_index = {word: idx for idx, word in enumerate(vocab)}

def vectorize(text):
    vec = [0] * len(vocab)
    counts = Counter(tokenize(text))
    for word, count in counts.items():
        if word in vocab_index:
            vec[vocab_index[word]] = count
    return vec

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

query = "怎么申请退款"
query_vec = vectorize(query)

scores = []
for chunk in chunks:
    score = cosine_similarity(query_vec, vectorize(chunk))
    scores.append((score, chunk))

scores.sort(reverse=True)
for score, chunk in scores:
    print(round(score, 4), "->", chunk)
```

这就是“检索”的最小原理版。

---

## 八、真实项目里通常会更复杂

真实 RAG 系统里，向量化一般会用专门的 embedding 模型，而不是简单词频。

但思路是一致的：

1. 查询转向量
2. 文档块转向量
3. 在向量空间里找最相近的块

所以别被“向量数据库”这个词吓到。  
它本质上还是在做相似度检索，只是规模更大、效率更高。

---

## 九、文档处理最容易出问题的地方

### 1. chunk 太大

召回不精准，浪费上下文。

### 2. chunk 太小

信息不完整，模型看到的片段支离破碎。

### 3. 清洗过头

把标题、层级、表格结构等有价值信息也删掉了。

### 4. 没有元数据

后面很难解释“答案来自哪里”。

---

## 小结

这节课最关键的认识是：

> **RAG 的前处理不是配角，而是效果上限的重要来源。**

检索做不好，生成几乎不可能稳定做好。  
所以文档清洗、切块、元数据、向量化，都是必须认真设计的环节。

---

## 练习

1. 调整 `chunk_size` 和 `overlap`，观察切块结果有什么变化。
2. 往 `chunks` 里加入一条和退款完全无关的文本，再看检索分数排序。
3. 思考：如果一个政策条款跨了两段，怎么设计 chunk 才不容易把关键信息切断？
