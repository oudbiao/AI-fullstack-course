---
title: "7.2 项目：文本摘要系统"
sidebar_position: 22
description: "从切句、句子打分、摘要生成、评估到失败分析，走通一个真正可展示的文本摘要项目闭环。"
keywords: [text summarization, extractive summarization, TF-IDF, evaluation, NLP project]
---

# 项目：文本摘要系统

:::tip 本节定位
摘要项目很适合作为作品集，因为它逼着你回答几个非常真实的问题：

- 什么叫关键信息
- 怎样压缩长文本
- 怎样判断摘要是否真的好

这一节不会只停在“会抽几句”，而会把一个作品级项目最该展示的部分讲清楚。
:::

## 学习目标

- 学会定义一个摘要项目的最小闭环
- 学会把抽取式 baseline 做成可解释系统
- 学会设计最小评估和失败分析
- 学会把这个题材包装成一个完整的 NLP 项目

---

## 一、项目题目怎么收窄？

一个适合练手的题目可以是：

> **给课程长文介绍生成 2 句摘要。**

这类题目好在：

- 领域清晰
- 文本长度适中
- 摘要目标比较直观

---

## 二、作品级摘要项目最小闭环

1. 选文本集合
2. 切句
3. 句子打分
4. 选 top-k 句子
5. 做人工评估
6. 总结失败模式

---

## 三、先做一个更完整的抽取式摘要系统

```python
import re

article = """
人工智能课程的学习路径通常分为基础阶段和进阶阶段。
基础阶段包括 Python 编程、数据分析和机器学习。
当学习者掌握了这些内容之后，才能更稳地进入深度学习和大模型应用开发。
很多人一开始就想直接学大模型，但往往因为基础不牢而很快卡住。
如果学习目标是做 AI 应用工程，理解数据处理、模型训练和系统部署都很重要。
""".strip()


def split_sentences(text):
    parts = re.split(r"[。！？\\n]+", text)
    return [p.strip() for p in parts if p.strip()]


def sentence_score(sentence, all_sentences):
    # 极简词频打分：句子中的高频词越多，分数越高
    tokens = "".join(all_sentences)
    return sum(tokens.count(ch) for ch in sentence if ch.strip())


def summarize(text, top_k=2):
    sentences = split_sentences(text)
    scored = [
        (sentence_score(sent, sentences), idx, sent)
        for idx, sent in enumerate(sentences)
    ]
    top = sorted(sorted(scored, reverse=True)[:top_k], key=lambda x: x[1])
    return "。".join(item[2] for item in top) + "。", scored


summary, scored = summarize(article, top_k=2)
print("summary:", summary)
print("scores:", scored)
```

### 3.1 这个示例为什么更像项目？

因为它不只给你结果，  
还保留了：

- 切句结果
- 打分结果

这让你能做：

- 解释
- 调试
- 失败分析

### 3.2 为什么摘要项目特别值得展示中间分数？

因为摘要好不好本来就带主观性。  
中间打分过程能帮助别人理解：

- 你是怎样做选择的

---

## 四、一个最小人工评估表该长什么样？

```python
eval_cases = [
    {
        "text": article,
        "gold_focus": ["基础阶段", "深度学习和大模型", "系统部署"],
    }
]

for case in eval_cases:
    pred_summary, _ = summarize(case["text"], top_k=2)
    covered = [item for item in case["gold_focus"] if item in pred_summary]
    print({
        "summary": pred_summary,
        "covered_focus": covered,
        "coverage_ratio": round(len(covered) / len(case["gold_focus"]), 4),
    })
```

### 4.1 这个评估为什么简单但有用？

因为它逼你回答：

- 摘要到底保没保住主线

这比只看“读起来顺不顺”更具体。

---

## 五、摘要项目最值得展示的失败案例

例如：

- 选句重复
- 漏掉关键信息
- 句子顺序不自然

### 为什么这些很值得展示？

因为它们恰好体现了抽取式摘要的典型局限。

---

## 六、怎么把这个项目再推成作品级？

### 6.1 加一个生成式摘要对比

### 6.2 增加更多文本类型

例如：

- 新闻
- 课程介绍
- 会议纪要

### 6.3 做一页 before / after 展示

例如：

- 原文
- baseline 摘要
- 调优后摘要
- 失败分析

---

## 七、小结

这节最重要的是建立一个作品级判断：

> **摘要项目的关键，不只是能抽几句，而是你能否把“切句、打分、生成、评估和失败分析”讲成一个可解释的闭环。**

只要这个闭环清楚，文本摘要项目就会非常像一个成熟的 NLP 作品。

---

## 练习

1. 把 `top_k` 改成 1 和 3，观察摘要内容怎么变化。
2. 为什么摘要项目特别值得展示“中间打分结果”？
3. 想一想：抽取式摘要最容易出现哪类失败？
4. 如果你要把这个项目放进作品集，你会优先展示哪 4 部分？
