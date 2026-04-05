---
title: "7.2 项目：智能问答系统"
sidebar_position: 21
description: "从知识库设计、检索、拒答、评估到展示方式，走通一个真正可解释的小型问答系统项目。"
keywords: [QA system, retrieval QA, knowledge base, FAQ, evaluation, NLP project]
---

# 项目：智能问答系统

:::tip 本节定位
问答系统很适合作为 NLP 作品集项目，因为它天然能展示：

- 文本表示
- 相似度
- 检索
- 拒答策略

但要让它像“项目”，而不只是“能回答几句”的 demo，关键在于：

> **知识边界、检索质量、拒答机制和评估方式都要讲清楚。**
:::

## 学习目标

- 学会定义一个可解释的小型问答系统范围
- 学会设计知识库、检索器和拒答策略
- 学会用最小评估集做系统验证
- 学会把问答系统包装成作品集页面

---

## 一、项目题目怎么收窄？

一个很稳的起点是：

> **做一个课程平台 FAQ 检索式问答系统。**

它适合的原因是：

- 题目范围清楚
- 知识库容易准备
- 错误原因容易分析

---

## 二、作品级问答项目最小闭环

1. 定义知识范围
2. 准备知识库
3. 做检索基线
4. 增加拒答
5. 做评估集
6. 展示错误分析

只要这 6 步清楚，项目就已经很有说服力。

## 三、推荐推进顺序

对新人来说，更稳的顺序通常是：

1. 先把知识范围收窄
2. 再做最简单检索 baseline
3. 再补拒答机制
4. 最后再做评估和展示

这样项目会更像“可解释系统”，而不是“碰巧答对几句”的 demo。

### 3.1 为什么问答系统特别适合训练“系统边界感”？

因为它会逼你一直面对三个问题：

- 这个系统到底知道什么
- 它不知道什么
- 它什么时候应该停住不答

这正是很多真实产品系统最关键的一层判断。

---

## 四、先做一个更完整的最小系统

```python
knowledge_base = [
    {"question": "课程多久内可以退款？", "answer": "课程购买后 7 天内且学习进度低于 20% 可申请退款。"},
    {"question": "证书怎么获得？", "answer": "完成所有必修项目并通过结课测试后，可以获得结业证书。"},
    {"question": "学习顺序是什么？", "answer": "建议先学 Python、数据分析、机器学习，再进入深度学习和大模型阶段。"},
    {"question": "前四阶段需要 GPU 吗？", "answer": "前四阶段不需要 GPU，普通电脑即可完成学习。"},
]


def tokenize(text):
    return set(text.replace("？", "").replace("?", ""))


def answer_question(user_query):
    query_tokens = tokenize(user_query)
    scored = []

    for item in knowledge_base:
        score = len(query_tokens & tokenize(item["question"]))
        scored.append((score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_item = scored[0]
    return {
        "matched_question": best_item["question"],
        "answer": best_item["answer"],
        "score": best_score,
    }


print(answer_question("退款时间是多久"))
print(answer_question("怎么拿证书"))
```

### 4.1 这个例子为什么更像项目，而不只是一个函数？

因为它已经有：

- 知识库
- 匹配逻辑
- 匹配得分
- 可解释的返回结果

### 4.2 为什么 `matched_question` 很值得展示？

因为它能帮你回答：

- 系统是答对了
- 还是只是碰巧答得像

### 4.3 为什么“检索命中什么”比“回答看起来顺不顺”更值得先看？

因为问答系统很多错误并不是生成层的错误，  
而是：

- 一开始就命中了不对的知识

如果这一步没看清，  
后面你会很难判断问题到底出在哪。

---

## 五、拒答机制为什么是作品级问答系统的关键？

没有拒答时，系统很容易：

- 任何问题都硬答

这在真实项目里很危险。

```python
def safe_answer_question(user_query, threshold=2):
    result = answer_question(user_query)
    if result["score"] < threshold:
        return {
            "answer": "当前知识库中没有足够相关的信息。",
            "matched_question": None,
            "score": result["score"],
        }
    return result


print(safe_answer_question("DeepSeek 和 OpenAI 哪个更强？"))
```

### 5.1 为什么这一步特别值钱？

因为它会让系统从：

- 总想说点什么

变成：

- 知道什么时候该停

这在作品集里很加分。

---

## 六、一个最小评估集怎么设计？

```python
eval_data = [
    ("退款时间是多久", "课程购买后 7 天内且学习进度低于 20% 可申请退款。"),
    ("证书怎么拿", "完成所有必修项目并通过结课测试后，可以获得结业证书。"),
    ("前四阶段需要显卡吗", "前四阶段不需要 GPU，普通电脑即可完成学习。"),
]


correct = 0
for q, gold in eval_data:
    pred = safe_answer_question(q, threshold=1)["answer"]
    if pred == gold:
        correct += 1

accuracy = correct / len(eval_data)
print("accuracy =", accuracy)
```

### 6.1 还应该评估什么？

除了准确率，还值得看：

- 拒答是否合理
- 哪些问题最容易误匹配
- 近义表达是否稳定

### 6.2 一个很适合新人的最小评估表

你可以先只做这样一张表：

| query | matched_question | answer | should_answer | actually_answered | correct |
|---|---|---|---|---|---|

这张表已经足够帮你判断：

- 命中对不对
- 拒答稳不稳
- 最后答案靠不靠谱

---

## 七、最值得展示的失败案例

例如：

- 问题说法变化后匹配错
- 知识库没有覆盖
- 不该回答时却给出错误答案

把这些列出来，会比只展示正确样例更像项目课。

### 7.1 如果继续把项目往上做，最值得补什么？

更值得优先补的通常是：

1. 近义表达鲁棒性测试
2. 更稳的拒答策略
3. 命中结果和最终回答的并排展示

这样项目会更像真正可解释的问答系统，而不是一组 FAQ 文本拼接。

---

## 八、小结

这节最重要的是建立一个作品级判断：

> **问答系统的价值，不只是“能答对几题”，而是你能否把知识边界、检索逻辑、拒答策略和错误分析讲成一个完整闭环。**

只要这条闭环立住，这个项目就会非常适合做作品集。

## 九、项目交付时最好补上的内容

- 一张知识边界说明表
- 一张检索命中 / 拒答效果示例
- 一组典型错例
- 一段你对下一步升级路线的说明

## 十、这节最该带走什么

- 问答系统首先是“知识边界系统”，其次才是“回答系统”
- 检索命中、拒答和错误分析是项目最值得展示的三块
- 如果能把“为什么答、为什么不答、为什么答错”讲清楚，这个项目就会非常像作品级项目

---

## 练习

1. 给知识库再加 5 条课程 FAQ，看看匹配效果如何变化。
2. 为什么拒答机制会显著提升项目可信度？
3. 想一想：如果两个问题很相近但答案不同，系统最容易出什么错？
4. 如果做作品集展示，你最想给面试官看哪 3 块内容？
