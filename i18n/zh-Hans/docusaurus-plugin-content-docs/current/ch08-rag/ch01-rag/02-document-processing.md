---
title: "1.3 文档处理与向量化"
sidebar_position: 2
description: "从清洗、切块、重叠、元数据到简单向量化，理解 RAG 前处理链路为什么决定效果上限。"
keywords: [chunking, 文档切块, 向量化, metadata, RAG preprocessing]
---

# 文档处理与向量化

![文档解析与向量化流程图](/img/course/document-processing-vectorization.png)

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

![RAG 文档处理流水线](/img/course/ch08-document-processing-pipeline-map-v2.png)

:::tip 读图提示
先看“原始资料”到“知识块”的变化：PDF、Word、PPT 不会直接变成答案，而是要经过清洗、切块、元数据补全、向量化和入库。OCR 是 Optical Character Recognition 的缩写，意思是“光学字符识别”，专门把扫描件或图片里的文字识别出来。
:::

---

## 三、切块为什么这么重要？

切块大小很像“做笔记时一张卡片写多少内容”。

- 太大：一次装太多，检索不精准
- 太小：上下文不够，回答容易断裂

这没有唯一标准，但一定要围绕任务调。

类比一下：

> 做开卷考试笔记时，你不会把整本书粘成一张超大海报，也不会把每个字都剪成一张纸条。

![Chunk 大小与 overlap 取舍图](/img/course/ch08-chunk-size-overlap-tradeoff-map.png)

:::tip 读图提示
这张图要从中间的“证据完整度”看起：chunk 太大会让检索变钝，chunk 太小会切断证据，overlap 的价值是给边界处的信息多留一段缓冲。
:::

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

## 七、如果你的目标是“知识库驱动的课件生成助手”，切块方式要多想一步

这类项目和普通 FAQ 问答有一个很大的不同：

- 你不是只想“找到相关段落”
- 你还想把资料重新组织成“知识点 / 例题 / 练习”

所以第一次做时，切块不要只按长度想，
还要按“内容类型”想。

更稳的默认思路通常是：

| 内容类型 | 更适合怎么切 |
|---|---|
| 概念定义 | 保留完整定义和公式，不要切断 |
| 例题讲解 | 题目 + 解题过程尽量在同一块 |
| 练习题 | 一题一块，方便后面单独抽取 |
| 章节总结 | 保留标题和要点列表 |

这张表很重要，因为它会帮新人意识到：

> **切块不是固定的文本操作，它其实在服务后面的生成目标。**

![课件知识块元数据 schema 图](/img/course/ch08-courseware-chunk-metadata-schema-map.png)

:::tip 读图提示
课件生成最怕“找到了文字却不知道该放哪”。看图时重点关注 `topic`、`content_type`、`source_origin`、`page_or_slide` 这几个字段，它们会决定后面能不能按知识点、例题和练习稳定组装。
:::

## 八、一个更像课件生成项目的知识块示例

```python
courseware_chunks = [
    {
        "topic": "折扣应用题",
        "content_type": "concept",
        "section": "知识点回顾",
        "page": 1,
        "text": "折扣 = 原价 × 折扣率",
    },
    {
        "topic": "折扣应用题",
        "content_type": "example",
        "section": "例题讲解",
        "page": 2,
        "text": "商品原价 100 元，打 8 折后价格是多少？",
    },
    {
        "topic": "折扣应用题",
        "content_type": "exercise",
        "section": "课堂练习",
        "page": 3,
        "text": "一件衣服原价 80 元，打 7 折后是多少元？",
    },
]

for item in courseware_chunks:
    print(item["content_type"], "->", item["text"])
```

这个例子最值得新人注意的是：

- 同一个主题下，知识块最好还能再分概念、例题、练习
- 这样后面生成 Word 课件时，系统就知道什么该放进哪个栏目

---

## 九、向量化到底在做什么？

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
    return re.findall(r"[\w\u4e00-\u9fff\u3040-\u30ff]+", text.lower())

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

## 十、真实项目里通常会更复杂

真实 RAG 系统里，向量化一般会用专门的 embedding 模型，而不是简单词频。

但思路是一致的：

1. 查询转向量
2. 文档块转向量
3. 在向量空间里找最相近的块

所以别被“向量数据库”这个词吓到。
它本质上还是在做相似度检索，只是规模更大、效率更高。

---

## 十一、文档处理最容易出问题的地方

### 1. chunk 太大

召回不精准，浪费上下文。

### 2. chunk 太小

信息不完整，模型看到的片段支离破碎。

### 3. 清洗过头

把标题、层级、表格结构等有价值信息也删掉了。

### 4. 没有元数据

后面很难解释“答案来自哪里”。

### 5. 只按长度切块，不按任务切块

对课件生成项目来说，这会导致：

- 例题和解题过程被拆散
- 概念和练习混在一起
- 后面很难稳定组装成固定格式文档

---

## 文档处理验收表

做完文档处理后，不要只看“生成了多少 chunk”，而要检查这些 chunk 是否真的能支撑后续问答。

| 检查项 | 合格表现 | 常见问题 |
|---|---|---|
| 文本清洗 | 去掉页眉页脚、重复空白、无意义噪声 | 清洗过头，把标题和表格结构删掉 |
| chunk 完整性 | 一个 chunk 能表达完整事实或完整步骤 | 关键条件被切到相邻 chunk |
| chunk 粒度 | 能被准确召回，也不会太碎 | 太大召回不准，太小证据不完整 |
| 元数据 | 保留 source、section、page、topic、content_type | 答案无法引用来源，无法按主题过滤 |
| 样例抽查 | 随机抽 10 个 chunk 人工看一遍 | 只看数量，不看质量 |

最实用的做法是先做一份“chunk 抽查表”。每次改切块规则后，随机抽几条 chunk，判断它们是否适合被检索、引用和展示。

## 一个 chunk 质量抽查脚本

下面这个脚本不依赖外部库，只用于帮你建立检查习惯。真实项目里可以把抽查结果写入 CSV 或 Markdown。

```python
chunks_with_meta = [
    {
        "id": "policy_001_01",
        "text": "课程购买后 7 天内，如果学习进度低于 20%，可以申请退款",
        "source": "course_policy.pdf",
        "section": "退款政策",
        "page": 3,
        "content_type": "policy",
    },
    {
        "id": "policy_001_02",
        "text": "完成所有必修项目并通过结课测试后，可以获得结业证书",
        "source": "course_policy.pdf",
        "section": "证书说明",
        "page": 5,
        "content_type": "rule",
    },
]

required_fields = {"id", "text", "source", "section", "page", "content_type"}

for chunk in chunks_with_meta:
    missing = required_fields - set(chunk)
    too_short = len(chunk["text"]) < 10
    too_long = len(chunk["text"]) > 300
    print({
        "id": chunk.get("id"),
        "missing_fields": sorted(missing),
        "too_short": too_short,
        "too_long": too_long,
        "preview": chunk["text"][:40],
    })
```

这个脚本不会替你判断语义质量，但能先发现一类基础问题：字段缺失、chunk 过短、chunk 过长、来源不可追踪。

## 切块策略对比记录

建议每次尝试一种切块策略，都用固定格式记录结果。

| 策略 | 参数 | 优点 | 暴露的问题 | 是否保留 |
|---|---|---|---|---|
| 按句子切 | 1 句 1 块 | 简单，召回精准 | 很多证据不完整 | 只适合短 FAQ |
| 滑动窗口 | 2～4 句，overlap 1 | 不容易切断上下文 | chunk 数量增加 | 适合作为 baseline |
| 按标题层级切 | H2/H3 下内容成块 | 保留结构 | 长章节可能过大 | 适合教程和文档 |
| 按内容类型切 | 概念/例题/练习分开 | 适合生成课件 | 需要解析或标注 | 适合结构化项目 |

如果你不知道从哪里开始，建议先用“标题层级 + 滑动窗口”作为 baseline，再根据评估集调整。

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
4. 如果你的目标是生成课件，想一想：概念、例题、练习为什么不适合完全用同一种切块方式？
