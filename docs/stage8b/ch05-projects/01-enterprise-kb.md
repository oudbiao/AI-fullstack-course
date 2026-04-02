---
title: "5.1 项目：企业知识库问答"
sidebar_position: 21
description: "从知识整理、切块、检索、权限和引用来源出发，走通一个企业知识库问答系统的最小项目。"
keywords: [enterprise knowledge base, RAG project, retrieval, metadata, source citation, permissions]
---

# 项目：企业知识库问答

:::tip 本节定位
企业知识库问答是 LLM 应用里最典型、也最容易真正落地的一类项目。  
它之所以重要，不是因为名字听起来高级，而是因为它几乎把这一阶段所有核心能力都串起来了：

- 文档处理
- 检索
- 元数据
- 引用来源
- 系统边界

这一节我们先做一个最小但完整的企业知识库雏形。
:::

## 学习目标

- 理解企业知识库问答系统的核心模块
- 学会把文档组织成可检索知识单元
- 理解为什么企业场景特别需要元数据和权限边界
- 跑通一个带来源返回的最小知识库问答系统

---

## 一、企业知识库问答和普通 FAQ 有什么不同？

### 1.1 企业场景通常更复杂

普通 FAQ 往往是：

- 问题比较固定
- 答案比较短

而企业知识库更常见的是：

- 文档长
- 知识分散
- 版本多
- 权限敏感

例如同样是“退款”问题，企业里可能同时存在：

- 对外用户政策
- 内部客服处理规范
- 不同产品线版本说明

### 1.2 所以企业知识库项目的重点不只是“能答”

还要考虑：

- 答案来自哪里
- 是否查到了正确文档
- 是否命中了正确权限范围
- 是否能给出可追踪引用

这也是它和简单问答系统的关键差别。

---

## 二、先设计一个最小知识库

### 2.1 不只存文本，还要存元数据

```python
kb = [
    {
        "id": "doc_001",
        "section": "退款政策",
        "department": "support",
        "text": "课程购买后 7 天内且学习进度低于 20% 可申请退款。"
    },
    {
        "id": "doc_002",
        "section": "证书说明",
        "department": "teaching",
        "text": "完成所有必修项目并通过结课测试后，可以获得结业证书。"
    },
    {
        "id": "doc_003",
        "section": "内部客服 SOP",
        "department": "internal",
        "text": "客服在处理退款申请时，需要先核验订单号与学习进度。"
    }
]

for item in kb:
    print(item)
```

### 2.2 为什么元数据这么重要？

因为企业知识库里不只是“内容像不像”，还会涉及：

- 属于哪个部门
- 是否内部可见
- 属于哪类文档

也就是说：

> 企业知识库问答不仅是语义检索，也是有边界的检索。 

---

## 三、做一个最小检索器

### 3.1 先用 TF-IDF 走通主线

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

texts = [item["text"] for item in kb]
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
doc_vectors = vectorizer.fit_transform(texts)

def retrieve(query, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    ranked_idx = scores.argsort()[::-1][:top_k]

    results = []
    for idx in ranked_idx:
        results.append({
            "score": float(scores[idx]),
            **kb[idx]
        })
    return results

print(retrieve("退款政策是什么"))
```

### 3.2 这一步已经说明了什么？

它说明：

> 企业知识库的第一步不是让模型生成，而是先把最相关资料找出来。 

这也是后面所有 RAG 系统的基本盘。

---

## 四、加上“来源返回”

### 4.1 为什么企业场景很需要引用来源？

因为企业用户经常会追问：

- 这条规则是从哪来的？
- 你引用的是哪份文档？

如果系统不能说清来源，可信度会明显下降。

### 4.2 一个可运行示例

```python
def answer_with_sources(query):
    hits = retrieve(query, top_k=2)

    if not hits or hits[0]["score"] < 0.1:
        return {
            "answer": "知识库中没有找到足够相关的信息。",
            "sources": []
        }

    top_hit = hits[0]
    return {
        "answer": top_hit["text"],
        "sources": [
            {
                "id": top_hit["id"],
                "section": top_hit["section"],
                "department": top_hit["department"]
            }
        ]
    }

print(answer_with_sources("退款政策是什么？"))
```

这样系统输出就不只是答案，还包括：

- 文档 id
- 所属章节
- 所属部门

这就是“企业可追踪性”的雏形。

---

## 五、企业知识库为什么必须考虑权限？

### 5.1 一个典型问题

上面的 `doc_003` 是内部客服 SOP。  
如果对外用户也能检索到这类内部文档，系统就有问题。

### 5.2 一个最小权限过滤示例

```python
def retrieve_with_permission(query, allowed_departments, top_k=2):
    all_hits = retrieve(query, top_k=len(kb))
    filtered = [hit for hit in all_hits if hit["department"] in allowed_departments]
    return filtered[:top_k]

print("外部用户:")
print(retrieve_with_permission("退款怎么处理", allowed_departments={"support", "teaching"}))

print("\n内部客服:")
print(retrieve_with_permission("退款怎么处理", allowed_departments={"support", "teaching", "internal"}))
```

### 5.3 这一步为什么特别关键？

因为企业知识库项目真正上线时，往往首先不是技术难，而是：

> **信息边界不能乱。**

这也是为什么企业项目和普通问答 Demo 差别很大。

---

## 六、把整个系统串起来

### 6.1 一个最小“企业知识库助手”

```python
def enterprise_kb_assistant(query, allowed_departments):
    hits = retrieve_with_permission(query, allowed_departments=allowed_departments, top_k=2)

    if not hits or hits[0]["score"] < 0.1:
        return {
            "answer": "当前权限范围内没有找到足够相关的信息。",
            "sources": []
        }

    top_hit = hits[0]
    return {
        "answer": top_hit["text"],
        "sources": [
            {
                "id": top_hit["id"],
                "section": top_hit["section"]
            }
        ]
    }

print(enterprise_kb_assistant("退款政策是什么", {"support", "teaching"}))
print(enterprise_kb_assistant("退款怎么处理", {"support", "teaching", "internal"}))
```

这个例子虽然小，但已经具备了企业知识库项目的几个核心元素：

- 检索
- 权限过滤
- 答案返回
- 来源引用

---

## 七、真实企业知识库项目还会继续补什么？

### 7.1 文档处理

真实系统通常不会只存一行摘要，而会有：

- PDF
- Word
- Wiki 页面
- 表格

所以还要做：

- 清洗
- 切块
- 元数据抽取

### 7.2 更强检索

比如：

- embedding 检索
- 混合检索
- rerank

### 7.3 更强回答层

例如：

- 检索后再让模型生成答案
- 明确引用多条来源
- 在证据不足时拒答

所以本节项目可以理解成：

> 企业知识库问答的最小骨架。 

---

## 八、怎样评估一个企业知识库问答系统？

### 8.1 不能只看“回答像不像”

还要看：

- 是否检索到了正确文档
- 是否引用了正确来源
- 是否遵守了权限范围

### 8.2 一个简单的评估思路

```python
eval_queries = [
    ("退款政策是什么", "doc_001"),
    ("证书如何获得", "doc_002")
]

correct = 0
for query, gold_id in eval_queries:
    result = enterprise_kb_assistant(query, {"support", "teaching"})
    if result["sources"] and result["sources"][0]["id"] == gold_id:
        correct += 1

print("source hit rate =", correct / len(eval_queries))
```

这虽然简单，但已经比只看“句子顺不顺”更接近企业项目的真实需求。

---

## 九、初学者最常踩的坑

### 9.1 只做“像聊天”的回答，不做来源引用

这在企业场景里通常不够。

### 9.2 不做权限隔离

这是很危险的问题。

### 9.3 没有评估集

没有评估集，系统就很难真正迭代。

---

## 小结

这一节最重要的不是把知识库做成一个“能回话的程序”，而是理解：

> **企业知识库问答系统的核心，是在正确权限边界内，把正确文档找出来，并以可追踪的方式返回给用户。**

这就是企业知识库和普通 FAQ Demo 的本质差别。

---

## 练习

1. 给知识库再加两条文档，并设计新的元数据字段，例如 `version`。
2. 用自己的话解释：为什么来源引用和权限过滤在企业知识库里特别重要？
3. 改造 `enterprise_kb_assistant()`，让它一次返回 top-2 来源。
4. 想一想：如果系统检索对了文档，但答案还是不稳，下一步你会往哪一层继续升级？
