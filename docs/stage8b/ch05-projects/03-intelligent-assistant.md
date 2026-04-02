---
title: "5.3 项目：智能问答助手"
sidebar_position: 23
description: "从多轮对话、知识检索、工具调用到会话状态，设计一个更接近真实产品的智能问答助手。"
keywords: [assistant project, multi-turn QA, dialog state, retrieval, tool calling, LLM app]
---

# 项目：智能问答助手

:::tip 本节定位
这一节和“企业知识库问答”很像，但目标更进一步。  
企业知识库更偏“查资料并回答”，而智能问答助手更像一个真正和用户互动的系统：

- 能多轮对话
- 能记住上下文
- 能在必要时调用工具

也就是说，这一节更接近“产品雏形”。
:::

## 学习目标

- 理解一个智能问答助手和普通问答函数的差别
- 学会把检索、状态、工具调用放进同一条流程
- 做出一个最小可运行的多轮问答助手
- 理解这种项目应该怎样评估和继续扩展

---

## 一、智能问答助手和普通问答系统差在哪？

### 1.1 普通问答系统更像单轮问答

例如：

- 你问一句
- 系统答一句

它未必关心上一轮发生了什么。

### 1.2 智能问答助手更像持续协作

例如：

1. 用户问退款政策
2. 系统追问是想看时间范围还是资格条件
3. 用户继续补充“我已经学了 30%”
4. 系统再综合判断

这就要求系统具备：

- 多轮上下文
- 状态管理
- 检索能力
- 适当的工具使用

这也是为什么它更像“助手”，而不只是“答题器”。

---

## 二、先设计一个最小系统结构

我们先不追求复杂，而是先定一个最小闭环：

1. 维护会话历史
2. 检索知识库
3. 根据当前问题和上下文给回答
4. 必要时调用工具

### 2.1 最小知识库

```python
kb = [
    {"key": "退款", "text": "退款政策：购买后 7 天内且学习进度低于 20% 可退款。"},
    {"key": "证书", "text": "证书政策：完成所有项目并通过测试后可获得证书。"}
]

for item in kb:
    print(item)
```

### 2.2 一个小工具

```python
def get_user_progress(user_id):
    progress_db = {
        1: 0.15,
        2: 0.30
    }
    return progress_db.get(user_id, None)
```

这已经有了知识和工具两层。

---

## 三、先做检索函数

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

vectorizer = TfidfVectorizer(token_pattern=r"(?u)\\b\\w+\\b")
doc_vectors = vectorizer.fit_transform([item["text"] for item in kb])

def retrieve(query):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, doc_vectors)[0]
    best_idx = scores.argmax()
    return kb[best_idx], float(scores[best_idx])

print(retrieve("退款政策是什么"))
```

这个检索器不复杂，但已经足够支撑一个最小对话助手。

---

## 四、加入会话状态

### 4.1 为什么一定要有状态？

因为多轮对话里，系统不能每轮都当第一次见到用户。

例如：

- 本轮关注的是退款
- 用户的学习进度是多少
- 刚才已经追问过什么

### 4.2 一个最小状态结构

```python
def new_session():
    return {
        "history": [],
        "topic": None,
        "user_id": None
    }

session = new_session()
print(session)
```

这就是一个最小“助手状态”。

---

## 五、把检索、状态和工具串起来

### 5.1 一个最小智能助手

```python
def assistant_reply(session, user_message):
    session["history"].append({"role": "user", "content": user_message})

    if "退款" in user_message:
        session["topic"] = "退款"
        doc, score = retrieve("退款")
        answer = f"{doc['text']} 你更想确认时间范围，还是想判断自己是否符合资格？"

    elif "证书" in user_message:
        session["topic"] = "证书"
        doc, score = retrieve("证书")
        answer = doc["text"]

    elif "我学了" in user_message and session["topic"] == "退款":
        # 简化解析：演示用，不做复杂 NLP 抽取
        progress = 0.30 if "30%" in user_message else 0.15 if "15%" in user_message else None
        if progress is None:
            answer = "我还没识别出你的学习进度，请明确告诉我是 15% 还是 30% 这类形式。"
        else:
            can_refund = progress < 0.2
            answer = (
                f"你的学习进度约为 {int(progress * 100)}%。"
                + (" 根据退款政策，你仍然可以申请退款。" if can_refund else " 根据退款政策，你当前不符合退款条件。")
            )

    else:
        answer = "我可以帮你查询退款或证书相关问题。"

    session["history"].append({"role": "assistant", "content": answer})
    return answer

session = new_session()
print(assistant_reply(session, "退款政策是什么？"))
print(assistant_reply(session, "我学了 30% 还能退吗？"))
```

### 5.2 这段代码已经比普通问答强在哪？

它已经具备了：

- 会话主题
- 多轮上下文
- 检索知识
- 基于上下文继续判断

这就是“助手感”的起点。

---

## 六、加入一个更清楚的工具调用边界

### 6.1 为什么要明确哪些能力走工具？

因为真实系统里：

- 检索类问题适合查知识库
- 用户状态类问题适合查工具 / 数据库

如果全靠模型硬猜，会越来越不稳。

### 6.2 一个最小工具调用版本

```python
def assistant_reply_with_tools(session, user_message, user_id=None):
    session["history"].append({"role": "user", "content": user_message})
    if user_id is not None:
        session["user_id"] = user_id

    if "退款" in user_message:
        session["topic"] = "退款"
        doc, _ = retrieve("退款")
        answer = f"{doc['text']} 如果你告诉我学习进度，我可以进一步帮你判断。"

    elif "能退吗" in user_message and session["topic"] == "退款" and session["user_id"] is not None:
        progress = get_user_progress(session["user_id"])
        if progress is None:
            answer = "当前查不到用户进度信息。"
        else:
            answer = (
                f"系统查到你的学习进度为 {int(progress * 100)}%。"
                + (" 仍可退款。" if progress < 0.2 else " 当前不满足退款条件。")
            )
    else:
        answer = "我可以帮助你处理退款或证书相关问题。"

    session["history"].append({"role": "assistant", "content": answer})
    return answer

session = new_session()
print(assistant_reply_with_tools(session, "退款政策是什么？", user_id=2))
print(assistant_reply_with_tools(session, "那我还能退吗？"))
```

现在这已经更像真实的产品逻辑了：

- 知识问题走检索
- 用户状态问题走工具

---

## 七、一个助手项目最重要的评估维度

### 7.1 不是只看一句话顺不顺

你至少要看：

- 检索是否对
- 状态是否连续
- 工具调用是否在正确时机发生
- 多轮对话有没有跑偏

### 7.2 一个简单评估思路

```python
test_cases = [
    {
        "dialog": ["退款政策是什么？", "那我还能退吗？"],
        "user_id": 1,
        "expected_keyword": "仍可退款"
    },
    {
        "dialog": ["退款政策是什么？", "那我还能退吗？"],
        "user_id": 2,
        "expected_keyword": "不满足退款条件"
    }
]

for case in test_cases:
    session = new_session()
    assistant_reply_with_tools(session, case["dialog"][0], user_id=case["user_id"])
    final_answer = assistant_reply_with_tools(session, case["dialog"][1])
    print(final_answer, "->", case["expected_keyword"] in final_answer)
```

这个评估虽然简单，但它已经在检查：

- 多轮连续性
- 工具调用是否正确支撑回答

---

## 八、怎样把这个项目继续升级？

下一步你可以继续加：

1. 更强检索
2. 更真实的用户信息库
3. 更严格的结构化输出
4. 前端聊天界面
5. 日志与观测

如果再往后走，就会自然进入：

- RAG
- Function Calling
- Agent

这也是为什么它是一个非常好的综合型项目。

---

## 九、初学者最常踩的坑

### 9.1 只做检索，不做会话状态

这样系统很难像“助手”。

### 9.2 状态全放自然语言，不做结构化字段

一到多轮任务就容易乱。

### 9.3 工具接上了，却没有设计什么时候该调

这样系统会显得很不稳。

---

## 小结

这一节最重要的不是写出一个“会聊天的函数”，而是理解：

> **智能问答助手的核心，是把知识检索、会话状态和工具调用放进同一条连续决策链里。**

只有这样，它才更像一个真正能帮你持续做事的助手。

---

## 练习

1. 给知识库再增加一个“学习顺序”主题，让助手也能回答。
2. 把 `assistant_reply_with_tools()` 改成返回结构化对象，而不是纯字符串。
3. 想一想：如果用户先问证书，再突然转问退款，系统状态该怎么处理？
4. 用自己的话解释：为什么智能问答助手比普通 FAQ 问答函数更难做？
