---
title: "7.1 项目：智能问答系统"
sidebar_position: 21
description: "从问答任务拆解、知识库构造、检索、答案生成到评估，走通一个可运行的小型问答系统项目。"
keywords: [QA system, question answering, retrieval, TF-IDF, NLP project, knowledge base]
---

# 项目：智能问答系统

:::tip 本节定位
问答系统是 NLP 学习里非常有代表性的项目，因为它把很多前面学过的东西都串起来了：

- 文本表示
- 相似度
- 检索
- 任务拆解

这一节我们不追求“超强大模型”，而是追求：

> **做出一个你能解释清楚原理的完整问答系统。**
:::

## 学习目标

- 理解一个小型问答系统的核心模块
- 学会把问答任务拆成“检索 + 回答”两步
- 用 TF-IDF + 相似度做一个最小可运行系统
- 理解问答系统该怎样评估和迭代

---

## 一、问答系统到底在做什么？

### 1.1 不同问答系统其实差很多

“问答系统”这个词很大，里面至少有几种不同路线：

- FAQ 匹配式问答
- 检索式问答
- 抽取式问答
- 生成式问答

对于我们现在的阶段，最适合先学的是：

> **检索式问答。**

也就是：

1. 先在知识库里找最相关内容
2. 再把这段内容当答案返回

### 1.2 为什么先学这个路线？

因为它：

- 最容易解释
- 最容易验证
- 最适合建立系统直觉

而且它和后面 RAG 的主线天然衔接。

---

## 二、先准备一个小型知识库

### 2.1 用最简单的问答知识库开始

```python
knowledge_base = [
    {"question": "课程多久内可以退款？", "answer": "课程购买后 7 天内且学习进度低于 20% 可申请退款。"},
    {"question": "证书怎么获得？", "answer": "完成所有必修项目并通过结课测试后，可以获得结业证书。"},
    {"question": "学习顺序是什么？", "answer": "建议先学 Python、数据分析、机器学习，再进入深度学习和大模型阶段。"},
    {"question": "前四阶段需要 GPU 吗？", "answer": "前四阶段不需要 GPU，普通电脑即可完成学习。"}
]

for item in knowledge_base:
    print(item)
```

### 2.2 为什么要显式准备知识库？

因为问答系统不是“模型自己凭空知道一切”，而是：

> 你得先明确它能回答哪些范围的问题。 

这一步其实就是给系统划边界。

---

## 三、最小问答系统：问题匹配

### 3.1 先用 TF-IDF 做检索

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

kb_questions = [item["question"] for item in knowledge_base]
vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
kb_vectors = vectorizer.fit_transform(kb_questions)

def answer_question(user_query):
    query_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vec, kb_vectors)[0]
    best_idx = scores.argmax()
    return {
        "matched_question": knowledge_base[best_idx]["question"],
        "answer": knowledge_base[best_idx]["answer"],
        "score": float(scores[best_idx])
    }

print(answer_question("退款时间是多久"))
print(answer_question("怎么拿证书"))
```

### 3.2 这段代码在教你什么？

它在教你问答系统最小闭环：

1. 把知识库问题向量化
2. 把用户问题向量化
3. 比相似度
4. 返回最像的一条答案

这已经是一个真正可运行的 FAQ 问答系统雏形。

---

## 四、把匹配结果展示得更清楚

### 4.1 不只返回答案，最好也返回匹配对象

因为这能帮助你调试：

- 系统是答对了，还是“碰巧答对”
- 它到底匹配到了哪一条

```python
queries = [
    "退款是几天内可以申请？",
    "证书怎么获取？",
    "学习路线怎么安排？"
]

for q in queries:
    result = answer_question(q)
    print("用户问题:", q)
    print("匹配问题:", result["matched_question"])
    print("答案    :", result["answer"])
    print("相似度  :", round(result["score"], 4))
    print("-" * 50)
```

---

## 五、什么时候该拒答？

### 5.1 问答系统不是一定要回答

如果用户问：

> “DeepSeek 和 OpenAI 哪个更强？”

而你的知识库只有课程 FAQ，  
这时系统最合理的做法不应该是乱答，而是：

> 承认不在知识范围内。 

### 5.2 加一个阈值

```python
def safe_answer_question(user_query, threshold=0.2):
    query_vec = vectorizer.transform([user_query])
    scores = cosine_similarity(query_vec, kb_vectors)[0]
    best_idx = scores.argmax()
    best_score = float(scores[best_idx])

    if best_score < threshold:
        return {
            "answer": "当前知识库中没有足够相关的信息。",
            "matched_question": None,
            "score": best_score
        }

    return {
        "matched_question": knowledge_base[best_idx]["question"],
        "answer": knowledge_base[best_idx]["answer"],
        "score": best_score
    }

print(safe_answer_question("DeepSeek 和 OpenAI 哪个更强？"))
```

这一步非常重要，因为它让系统从“总要说点什么”变成“知道什么时候该停”。

---

## 六、为什么这个项目已经很接近真实系统？

因为它已经包含了问答系统的几个关键模块：

- 知识库
- 检索器
- 相似度打分
- 结果返回
- 拒答策略

很多更复杂的问答系统，本质上也只是在这里继续往上加：

- 更强 embedding
- 更复杂 rerank
- 生成式回答
- 引用来源

所以这个小项目不是“玩具垃圾版”，而是一个非常重要的原型台阶。

---

## 七、怎样评估一个问答系统？

### 7.1 最小评估集

至少可以先准备：

- 用户问题
- 应答是否正确

### 7.2 一个简单评估示例

```python
eval_data = [
    ("退款时间是多久", "课程购买后 7 天内且学习进度低于 20% 可申请退款。"),
    ("证书怎么拿", "完成所有必修项目并通过结课测试后，可以获得结业证书。"),
    ("前四阶段需要显卡吗", "前四阶段不需要 GPU，普通电脑即可完成学习。")
]

correct = 0
for q, gold in eval_data:
    pred = safe_answer_question(q)["answer"]
    if pred == gold:
        correct += 1

accuracy = correct / len(eval_data)
print("accuracy =", accuracy)
```

### 7.3 真实项目里还能继续加什么？

- Top-k 命中率
- 用户满意度
- 拒答准确率
- 错误案例分析

---

## 八、问答系统最常见的升级路线

### 8.1 从 FAQ 匹配到检索问答

先做问题到问题的匹配。

### 8.2 从检索问答到检索 + 生成

后面你会在 RAG 里看到：

- 不只是返回原答案
- 而是先检索，再让模型基于证据生成回答

### 8.3 从单一知识库到企业知识库系统

后面还会继续进化到：

- 多数据源
- 元数据过滤
- 权限控制
- 日志与评估

所以这节课其实是在给后面的 RAG、企业知识库打地基。

---

## 九、初学者最常踩的坑

### 9.1 一上来就追求生成式问答

很多时候，先把检索式问答做稳更重要。

### 9.2 不做拒答

这会让系统在知识库外问题上表现得很不可信。

### 9.3 只看界面，不看匹配过程

如果你不知道系统到底匹配到了哪条知识，就很难调优。

---

## 小结

这一节最重要的不是“写出一个能回复问题的函数”，而是理解：

> **问答系统的关键，在于先把知识边界定义清楚，再用稳定的检索逻辑把用户问题和知识库连接起来。**

这就是后面更复杂问答系统和 RAG 系统的起点。

---

## 练习

1. 给知识库再加 3 条问答，测试系统能否正确匹配。
2. 调整拒答阈值 `threshold`，观察系统“更保守”或“更激进”时的变化。
3. 想一想：如果用户提问方式和知识库问题写法差很多，这个系统为什么可能会失效？
4. 用自己的话解释：为什么问答系统里“知道什么时候不答”同样重要？
