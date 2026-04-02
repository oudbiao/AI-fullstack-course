---
title: "5.2 项目：RAG+微调综合系统"
sidebar_position: 22
description: "从为什么单独 RAG 或单独微调都不够，到如何把两者组合，设计一个更完整的领域问答系统。"
keywords: [RAG, finetuning, domain adaptation, hybrid system, LLM project]
---

# 项目：RAG+微调综合系统

:::tip 本节定位
前面你已经分别学过：

- RAG：让模型先查资料再回答
- 微调：让模型更适应某类任务或风格

这一节要解决的问题是：

> **如果一个领域系统既需要外部知识，又需要特定表达风格和任务能力，该怎么办？**

这时，RAG 和微调往往不是替代关系，而是组合关系。
:::

## 学习目标

- 理解为什么“只做 RAG”或“只做微调”有时都不够
- 学会把领域问答系统拆成 RAG 层和微调层
- 设计一个可解释的 RAG+微调项目方案
- 跑通一个最小的组合式项目骨架

---

## 一、为什么要把 RAG 和微调组合起来？

### 1.1 单独 RAG 的优点和局限

RAG 的优点：

- 知识可更新
- 可引用来源
- 不必重新训练模型

但它也有局限：

- 模型未必懂你的领域表达
- 检索到了也未必会答得符合业务格式
- 复杂任务时，模型的“回答习惯”未必够稳

### 1.2 单独微调的优点和局限

微调的优点：

- 能让模型更懂特定任务形式
- 输出风格更稳定
- 指令跟随更贴合业务

但它也有局限：

- 新知识更新没那么灵活
- 很难靠微调记住所有细节文档
- 成本更高

### 1.3 所以它们经常是互补关系

可以先用一句话记住：

> **RAG 负责补知识，微调负责补行为。**

这正是组合式系统的核心逻辑。

---

## 二、这个项目到底在做什么？

我们把目标定成一个领域问答助手，比如：

- 面向企业内部政策文档
- 回答时要稳定引用来源
- 输出格式必须规范
- 某些问题需要用固定业务口径回答

也就是说，这个系统既要：

- 查得到知识
- 又要答得像该领域系统

---

## 三、先画出系统结构

```mermaid
flowchart LR
    A["用户问题"] --> B["检索器"]
    B --> C["相关文档块"]
    C --> D["微调后的回答模型"]
    D --> E["规范化输出"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style B fill:#fff3e0,stroke:#e65100,color:#333
    style C fill:#f3e5f5,stroke:#6a1b9a,color:#333
    style D fill:#e8f5e9,stroke:#2e7d32,color:#333
    style E fill:#ffebee,stroke:#c62828,color:#333
```

### 3.2 这张图真正重要的地方

不是“组件多”，而是职责清楚：

- 检索器负责找资料
- 微调模型负责按业务方式组织答案

这能让系统更可解释，也更容易迭代。

---

## 四、一个最小知识库和检索器

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

kb = [
    {"id": "doc1", "text": "退款政策：购买后 7 天内且学习进度低于 20% 可退款。"},
    {"id": "doc2", "text": "证书政策：完成项目并通过测试后可获得证书。"},
    {"id": "doc3", "text": "客服处理规范：回答时需要先说明政策依据，再给出结论。"}
]

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
doc_vectors = vectorizer.fit_transform([item["text"] for item in kb])

def retrieve(query, top_k=2):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    top_idx = scores.argsort()[::-1][:top_k]
    return [kb[i] for i in top_idx]

print(retrieve("退款条件是什么"))
```

这个检索器本身不复杂，但它已经是组合系统的第一半。

---

## 五、再模拟一个“微调后的回答风格”

在真实项目里，这一步可能来自：

- 指令微调
- LoRA / QLoRA
- 监督数据集训练

为了让代码能直接运行，这里我们先用规则模拟“已经被训练过的业务输出风格”。

```python
def domain_answer_style(question, retrieved_docs):
    evidence = " ".join(doc["text"] for doc in retrieved_docs)

    if "退款" in question:
        return {
            "answer": "根据现行退款政策，购买后 7 天内且学习进度低于 20% 的用户可申请退款。",
            "reasoning_style": "先政策后结论",
            "evidence": evidence
        }

    if "证书" in question:
        return {
            "answer": "根据证书政策，完成项目并通过测试后可以获得证书。",
            "reasoning_style": "先政策后结论",
            "evidence": evidence
        }

    return {
        "answer": "当前没有找到足够匹配的业务规则。",
        "reasoning_style": "谨慎拒答",
        "evidence": evidence
    }
```

### 5.2 为什么这个模拟是有意义的？

因为它在帮你理解：

- RAG 解决的是“知道什么”
- 微调解决的是“怎么答”

---

## 六、把两部分真正串起来

```python
def rag_plus_finetune_system(question):
    docs = retrieve(question, top_k=2)
    result = domain_answer_style(question, docs)
    return {
        "question": question,
        "retrieved_docs": docs,
        **result
    }

result = rag_plus_finetune_system("退款条件是什么？")
print(result["question"])
print(result["answer"])
print("evidence:", result["evidence"])
```

### 6.2 这个系统已经说明了什么？

它已经说明：

> 组合系统不是把两种技术硬拼，而是让它们各自做最擅长的部分。 

---

## 七、真正项目里，微调通常微调什么？

### 7.1 不是为了“记住所有文档”

很多新人会误以为：

> 微调后模型就该把知识库都背下来

但更常见、更现实的目标其实是：

- 学会领域术语风格
- 学会输出格式
- 学会业务回答模板
- 学会某类任务的固定结构

### 7.2 举个例子

你可能希望模型学会：

- “先引用政策，再给结论”
- “不确定时明确拒答”
- “所有回答都输出标准字段”

这类能力就很适合靠微调或至少靠强监督式训练来强化。

---

## 八、一个真正有项目价值的拆分方式

### 8.1 RAG 层负责

- 文档切块
- 检索
- 来源引用
- 知识更新

### 8.2 微调层负责

- 回答风格
- 输出格式
- 任务模板
- 业务术语理解

这个职责拆分一清楚，项目的可维护性会好很多。

---

## 九、怎样评估这个综合系统？

### 9.1 不能只看“答得顺不顺”

你至少要看两层：

- 检索层：有没有找到对的文档
- 回答层：输出是否符合业务要求

### 9.2 一个最小评估思路

```python
eval_data = [
    {"question": "退款条件是什么", "gold_doc": "doc1", "must_contain": "7 天内"},
    {"question": "证书如何获得", "gold_doc": "doc2", "must_contain": "通过测试"}
]

for item in eval_data:
    result = rag_plus_finetune_system(item["question"])
    hit = result["retrieved_docs"][0]["id"] == item["gold_doc"]
    good_answer = item["must_contain"] in result["answer"]
    print(item["question"], "retrieval_hit=", hit, "answer_ok=", good_answer)
```

这已经比“只看看 Demo 像不像”前进很多了。

---

## 十、初学者最常踩的坑

### 10.1 用微调去解决知识更新问题

这通常会很低效。

### 10.2 用 RAG 去强行解决输出风格稳定问题

这也不总合适。

### 10.3 两层职责混乱

如果你自己都说不清“哪一层在负责什么”，系统后面会很难调。

---

## 小结

这一节最重要的不是把 RAG 和微调两个词放在一起，而是理解：

> **RAG+微调综合系统的价值，在于让“知识获取”和“回答行为”分别由最合适的机制负责。**

这才是组合式 LLM 系统真正的工程思路。

---

## 练习

1. 给知识库再加两条新文档，观察检索结果是否变化。
2. 设计一个你自己的“领域回答风格规则”，模拟微调层行为。
3. 想一想：如果系统检索总是对，但回答格式总乱，你该优先优化 RAG 还是微调？
4. 用自己的话解释：为什么说“RAG 补知识，微调补行为”？
