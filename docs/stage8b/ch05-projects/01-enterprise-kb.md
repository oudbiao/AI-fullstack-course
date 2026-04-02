---
title: "5.1 项目：企业知识库问答"
sidebar_position: 21
description: "从文档切块、检索、权限过滤、引用来源到错误分析，建立一个企业知识库问答系统的作品级项目闭环。"
keywords: [enterprise knowledge base, RAG project, retrieval, metadata, source citation, permissions]
---

# 项目：企业知识库问答

:::tip 本节定位
企业知识库问答之所以适合作为作品集项目，不是因为它名字高级，而是因为它非常真实：

- 有文档
- 有权限
- 有版本
- 有引用
- 还有“答错会影响业务”的压力

所以这类项目最重要的不是“看起来像在回答”，而是：

> **答案是否来自正确文档、是否在正确权限范围内、是否能回溯到来源。**
:::

## 学习目标

- 学会把企业文档组织成可检索知识单元
- 学会设计权限和引用这两条企业级关键约束
- 学会用一个最小检索器搭出可展示项目闭环
- 学会围绕错误分析和可追溯性来展示项目

---

## 一、为什么企业知识库问答比普通 FAQ 更难？

### 1.1 文档更长

企业知识往往不只是几条问答，  
而是：

- 政策文档
- 内部 SOP
- 培训手册
- 产品说明

### 1.2 权限更复杂

同一个问题，可能存在：

- 对外版本
- 内部版本

### 1.3 可信度要求更高

用户常常会追问：

- 这条规则从哪来的？
- 你引用的是哪份文件？

所以企业知识库问答更像：

- 检索系统
- 引用系统
- 权限系统

的组合。

---

## 二、先把项目边界定清楚

一个很适合作品集展示的最小范围可以是：

> **针对课程平台内部帮助中心，做一个“退款 / 发票 / 证书 / 内部客服 SOP”知识库问答系统。**

它至少要回答四类问题：

1. 对外规则类问题
2. 内部流程类问题
3. 权限不同导致答案不同的问题
4. 需要给出来源的问题

### 为什么这个范围好？

- 文档主题集中
- 权限边界真实
- 结果好坏容易解释

---

## 三、先设计知识单元，而不是先写模型

下面这个示例会做三件事：

1. 把文档切成最小知识单元
2. 给每段加元数据
3. 区分对外和内部可见范围

```python
kb = [
    {
        "id": "doc_001",
        "section": "退款政策",
        "department": "support",
        "visibility": "public",
        "text": "课程购买后 7 天内且学习进度低于 20% 可申请退款。",
        "keywords": {"退款", "7天", "进度", "20%"},
    },
    {
        "id": "doc_002",
        "section": "证书说明",
        "department": "teaching",
        "visibility": "public",
        "text": "完成所有必修项目并通过结课测试后，可以获得结业证书。",
        "keywords": {"证书", "结课测试", "项目"},
    },
    {
        "id": "doc_003",
        "section": "内部客服 SOP",
        "department": "internal",
        "visibility": "internal",
        "text": "客服处理退款申请时，需要先核验订单号、学习进度和支付渠道。",
        "keywords": {"退款", "客服", "SOP", "核验"},
    },
]

for item in kb:
    print(item)
```

### 3.1 为什么这里要加这么多元数据？

因为企业知识库检索不只是“内容像不像”，  
还要判断：

- 能不能给当前用户看
- 它属于哪个业务域
- 回答时来源怎么展示

这也是企业项目和普通问答 demo 的根本差异。

---

## 四、先做一个可解释检索器

为了让示例在当前环境里也能直接跑，我们先不用外部 embedding 库，  
而是用一个纯 Python 的关键词重叠检索器，先把项目骨架搭稳。

```python
def retrieve(query, allowed_visibility, top_k=2):
    candidates = []

    for item in kb:
        if item["visibility"] not in allowed_visibility:
            continue
        score = sum(keyword in query for keyword in item["keywords"])
        candidates.append((score, item))

    candidates.sort(key=lambda x: x[0], reverse=True)
    return [item for score, item in candidates[:top_k] if score > 0]


print("public user:")
print(retrieve("退款规则是什么？", allowed_visibility={"public"}))

print("\ninternal support:")
print(retrieve("客服核验流程是什么？", allowed_visibility={"public", "internal"}))
```

### 4.1 这个检索器虽然简单，但为什么很适合教学？

因为它让你清楚看到三件事：

1. 查询词怎么影响召回
2. 权限怎么影响候选集
3. 结果为什么会不同

### 4.2 为什么这里故意不直接用 embedding？

因为这节课要先把：

- 权限
- 来源
- 结构化知识单元

这些企业级关键点讲清楚。  
等骨架清楚后，再换更强检索方式才更稳。

---

## 五、把“回答 + 来源”一起做出来

```python
def answer_with_sources(query, allowed_visibility):
    hits = retrieve(query, allowed_visibility=allowed_visibility, top_k=2)

    if not hits:
        return {
            "answer": "当前权限范围内没有找到足够相关的信息。",
            "sources": [],
        }

    top = hits[0]
    return {
        "answer": top["text"],
        "sources": [
            {
                "id": top["id"],
                "section": top["section"],
                "department": top["department"],
                "visibility": top["visibility"],
            }
        ],
    }


print(answer_with_sources("退款规则是什么？", {"public"}))
print(answer_with_sources("客服核验流程是什么？", {"public", "internal"}))
```

### 5.1 为什么“来源返回”是作品级项目的亮点？

因为它让系统不只是“给你一个答案”，  
而是还能回答：

- 这答案从哪来
- 为什么信它

这会显著提高项目可信度。

### 5.2 为什么企业场景比普通问答更需要来源？

因为企业用户经常会真的拿答案去执行流程。  
没有来源，就很难建立信任。

---

## 六、项目最该怎么评估？

### 6.1 不是只看“有没有答出来”

企业知识库项目最少要分成三层评估：

1. 召回是否相关
2. 权限是否正确
3. 引用是否可追溯

### 6.2 一个极简评估集

```python
eval_cases = [
    {
        "query": "退款规则是什么？",
        "visibility": {"public"},
        "expected_doc": "doc_001",
    },
    {
        "query": "客服核验流程是什么？",
        "visibility": {"public"},
        "expected_doc": None,
    },
    {
        "query": "客服核验流程是什么？",
        "visibility": {"public", "internal"},
        "expected_doc": "doc_003",
    },
]

for case in eval_cases:
    result = answer_with_sources(case["query"], case["visibility"])
    got = result["sources"][0]["id"] if result["sources"] else None
    print({
        "query": case["query"],
        "expected_doc": case["expected_doc"],
        "got": got,
        "match": got == case["expected_doc"],
    })
```

### 6.3 为什么这种评估很值钱？

因为它直接覆盖了企业知识库最关键的两个风险：

- 该答却没答对
- 不该看却看到了内部文档

---

## 七、这个项目怎么往作品级再推一步？

### 7.1 把规则检索升级成向量检索

### 7.2 加文档 chunking 和 rerank

### 7.3 把来源展示做成界面

最推荐展示：

- 用户问题
- 命中文档
- 最终答案
- 来源引用

### 7.4 展示几个“权限相关失败样例”

这会非常有说服力。

---

## 八、最容易踩的坑

### 8.1 只做“能答”，不做“能追溯”

### 8.2 只看语义相关，不看权限边界

### 8.3 文档单元切得太粗

切太粗时，答案和来源常常都会变得含糊。

---

## 九、小结

这节最重要的是建立一个作品级判断：

> **企业知识库问答真正像项目的地方，不是接了一个检索器，而是能把知识单元、权限边界、回答生成和来源追溯组织成一条可信闭环。**

只要这条闭环清楚了，这个项目就会非常像真实企业场景里的系统。

---

## 练习

1. 给 `kb` 再加两条“公开文档”和一条“内部文档”，让查询竞争更真实。
2. 为什么说企业知识库项目里“权限正确”有时比“答案漂亮”更重要？
3. 想一想：如果文档 chunk 切得太粗，会怎么影响回答和引用？
4. 如果你把这个项目做成作品集，首页最值得展示哪 4 块信息？
