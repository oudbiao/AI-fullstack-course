---
title: "8.1 项目：垂直领域微调"
sidebar_position: 27
description: "围绕一个可展示的领域助手项目，从任务边界、SFT 数据、baseline、评估和 before/after 对比，做成作品级微调课程。"
keywords: [domain finetuning, SFT, baseline, evaluation, vertical assistant, project]
---

# 项目：垂直领域微调

:::tip 本节定位
垂直领域微调项目最容易流于一句空话：

- “做一个行业专家模型”

真正有作品集价值的项目，通常更像：

> **把一个范围明确、边界清楚、评估可做的领域问答任务，做成一个 before/after 对比很清楚的系统。**

这一节会把这条线真正走清楚。
:::

## 学习目标

- 学会把“领域微调”题目收窄成一个可执行项目
- 学会把原始知识整理成 SFT 数据和评估集
- 学会做出真正有说服力的 baseline 对比
- 学会把这个项目展示成作品级页面

---

## 一、项目选题为什么必须先收窄？

### 1.1 大题目几乎都做不动

例如：

- 做一个行业专家大模型

这类题目通常边界太宽，  
很难明确：

- 输入是什么
- 输出是什么
- 什么叫答对

### 1.2 更适合作品集的题目

例如：

> **电商售后政策助手：专注退款、地址修改、发票和售后流程四类问题。**

这个题目好在：

- 范围窄
- 语义稳定
- 评估标准好设计

---

## 二、作品级微调项目最小闭环长什么样？

1. 定义任务边界
2. 整理知识与对话样本
3. 做 baseline
4. 整理 SFT 数据
5. 建评估集
6. 训练并做 before/after 对比

只要这 6 步清楚，你的项目就已经很有说服力。

---

## 三、先看一个更完整的数据与 baseline 示例

下面这个例子会同时体现：

- 原始记录
- SFT 样本
- 两个 baseline
- 评估规则

```python
raw_records = [
    {
        "intent": "refund_unshipped",
        "question": "订单还没发货，可以直接退款吗？",
        "policy_points": ["未发货可直接申请退款", "退款原路返回", "到账时间 3 到 7 个工作日"],
        "answer": "可以。如果订单尚未发货，您可以直接申请退款，款项会原路退回，通常 3 到 7 个工作日到账。",
    },
    {
        "intent": "change_address",
        "question": "收货地址填错了，还能改吗？",
        "policy_points": ["未出库可修改地址", "已出库需联系人工客服"],
        "answer": "如果订单尚未出库，您可以在订单详情页修改地址；若已经出库，请联系人工客服处理。",
    },
    {
        "intent": "invoice",
        "question": "发票什么时候可以开？",
        "policy_points": ["订单完成后可申请", "电子发票发送到邮箱"],
        "answer": "订单完成后可在发票中心申请开具，电子发票会发送到您预留的邮箱。",
    },
]


def build_sft_record(row):
    system = "你是电商售后政策助手，请用礼貌、准确、符合平台规则的方式回答用户问题。"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ],
        "intent": row["intent"],
        "policy_points": row["policy_points"],
    }


def generic_baseline(question):
    if "退款" in question:
        return "一般可以申请退款，具体以订单状态为准。"
    if "地址" in question:
        return "建议尽快联系客服处理地址问题。"
    if "发票" in question:
        return "发票通常可以申请，具体规则请查看页面说明。"
    return "请联系平台客服获取帮助。"


def retrieval_baseline(question, records):
    best = max(records, key=lambda row: overlap(question, row["question"]))
    return best["answer"]


def overlap(a, b):
    return len(set(a) & set(b))


def coverage(answer, policy_points):
    covered = sum(point[:4] in answer or point in answer for point in policy_points)
    return round(covered / len(policy_points), 3)


sft_dataset = [build_sft_record(row) for row in raw_records]
sample = raw_records[0]

generic_answer = generic_baseline(sample["question"])
retrieval_answer = retrieval_baseline(sample["question"], raw_records)

print("question:", sample["question"])
print("generic  :", generic_answer, "coverage=", coverage(generic_answer, sample["policy_points"]))
print("retrieval:", retrieval_answer, "coverage=", coverage(retrieval_answer, sample["policy_points"]))
print("sft_sample:", sft_dataset[0])
```

### 3.1 这个例子为什么比纯“项目规划对象”更值钱？

因为它已经把项目里真正最重要的四件事放出来了：

1. 原始数据长什么样
2. SFT 样本长什么样
3. baseline 结果是什么
4. 评估规则是什么

这已经是一个很像真实微调项目的核心骨架。

### 3.2 为什么要先做两个 baseline？

最少建议比较：

1. 纯 Prompt / 通用回答
2. 检索或简单领域匹配
3. 微调后系统

否则你后面很难说清：

- 微调到底提升了什么

---

## 四、微调项目最重要的评估不只是“看起来像专家”

### 4.1 结构化评估点

至少应包含：

- 是否覆盖关键政策点
- 是否有违规承诺
- 风格是否一致
- 是否答非所问

### 4.2 一个更像作品级的展示方式

最推荐展示：

- 同一组问题
- baseline 回答
- 微调后回答
- 逐条说明差异

### 4.3 失败样例非常重要

例如：

- 容易编政策
- 细节答错
- 口吻不稳定

这会比只展示成功案例更像真实项目。

---

## 五、怎么把这个项目做成作品级页面？

### 5.1 页面结构建议

1. 任务边界
2. 数据构造方式
3. baseline 对比
4. SFT 样本示例
5. before / after
6. 失败案例

### 5.2 一个很加分的点

把“政策点覆盖率”这种明确规则放出来。  
它会让项目显得非常扎实，而不是只靠主观评价。

---

## 六、最容易踩的坑

### 6.1 一上来就做大范围题目

这会让评估和数据一起发散。

### 6.2 没有 baseline

没有对比，微调项目几乎站不住。

### 6.3 只展示模型训练，不展示任务判断

项目真正值钱的地方在：

- 题目定义
- 数据组织
- 评估设计

---

## 七、小结

这节最重要的是建立一个作品级判断：

> **垂直领域微调项目真正有价值的地方，不是“我微调了一个模型”，而是你能否把任务边界、SFT 数据、baseline、评估规则和 before/after 对比讲成一条清楚闭环。**

只要这条闭环清楚，这个项目就非常适合拿来展示。

---

## 练习

1. 再给原始数据补 5 条样本，让四类意图都更均衡。
2. 想一想：如果 Retrieval baseline 已经很好，什么时候微调还值得做？
3. 为什么说“政策点覆盖率”比“感觉更像人工”更适合做项目评估？
4. 如果把这个项目做成作品集，最值得展示哪 4 个 before/after 样例？
