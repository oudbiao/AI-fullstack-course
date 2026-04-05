---
title: "10.3 项目：数据分析 Agent"
sidebar_position: 55
description: "围绕读取表格、做统计、输出图表建议和解释结论，建立一个可复核的数据分析 Agent 项目闭环。"
keywords: [data analysis agent, statistics, chart suggestion, insight generation, agent project]
---

# 项目：数据分析 Agent

:::tip 本节定位
数据分析 Agent 的真正价值不在于：

- 帮你算平均值

而在于：

> **它能否把“读数据 -> 做分析 -> 解释结论”串成一条可复核的链路。**

所以这类项目非常适合拿来展示多步工具协作和中间状态。
:::

## 学习目标

- 学会定义一个数据分析 Agent 的最小项目范围
- 学会把数据输入、统计计算和解释输出串成闭环
- 学会用最小样例做“可复核性”展示
- 学会把这个题材包装成一页很强的作品集项目

---

## 一、项目题目怎么收窄？

建议先做成：

- 读取一个小表
- 算几个核心统计量
- 根据统计量生成洞察摘要

而不是一开始做成：

- 自动 BI 平台
- 全自动报告工厂

---

## 二、先跑一个最小数据分析闭环

这个例子会做：

1. 读取一份小型销售表
2. 计算总销售额和品类均值
3. 给出一条简单分析结论

```python
sales = [
    {"category": "course", "amount": 299},
    {"category": "course", "amount": 199},
    {"category": "book", "amount": 59},
    {"category": "book", "amount": 79},
    {"category": "service", "amount": 499},
]


def summarize_sales(rows):
    total = sum(row["amount"] for row in rows)

    grouped = {}
    for row in rows:
        grouped.setdefault(row["category"], []).append(row["amount"])

    per_category_avg = {
        category: round(sum(values) / len(values), 2)
        for category, values in grouped.items()
    }

    top_category = max(per_category_avg, key=per_category_avg.get)

    return {
        "total_amount": total,
        "per_category_avg": per_category_avg,
        "insight": f"{top_category} 的客单价最高。",
    }


result = summarize_sales(sales)
print(result)
```

### 2.1 这个例子为什么已经很像项目？

因为它不只做了“计算”，  
还做了：

- 输入数据
- 中间统计
- 输出结论

这已经是最小数据分析工作流。

### 2.2 为什么“insight”特别重要？

因为用户通常不是为了看原始数字，  
而是为了得到：

- 有解释力的结论

这正是数据分析 Agent 和普通计算器的差别。

---

## 三、一个作品级数据分析 Agent 最该展示什么？

### 3.1 输入数据长什么样

最好明确：

- 字段
- 样本量
- 缺失值情况

### 3.2 中间计算结果

例如：

- 汇总统计
- 分组结果
- 趋势判断

### 3.3 最终解释

例如：

- 哪类商品表现最好
- 哪段时间波动最大

### 3.4 图表建议

即使你不直接生成图，也可以输出：

- 该用柱状图还是折线图

这会让项目更接近真实分析助手。

---

## 四、再加一个最小“图表建议器”

```python
def suggest_chart(columns):
    if "date" in columns and "amount" in columns:
        return "line_chart"
    if "category" in columns and "amount" in columns:
        return "bar_chart"
    return "table"


print(suggest_chart(["category", "amount"]))
print(suggest_chart(["date", "amount"]))
```

### 4.1 这个小模块有什么价值？

它说明项目不只是“算数”，  
而是在逐渐往：

- 分析
- 解释
- 可视化建议

推进。

---

## 五、最容易踩的坑

### 5.1 字段理解错

这是数据分析 Agent 的典型致命问题。  
如果字段含义理解错，后面全链路都可能被带偏。

### 5.2 只展示结论，不展示中间过程

这样项目会很像黑盒，难以建立信任。

### 5.3 只做 happy path

没有展示：

- 缺失值
- 异常值
- 统计口径冲突

项目会显得不够真实。

---

## 六、怎么把它打磨成作品级页面？

### 6.1 结构建议

1. 原始数据样例
2. 中间统计表
3. 洞察摘要
4. 图表建议
5. 错误案例

### 6.2 很值得补的一个亮点

把：

- 原始数据
- 中间计算
- 最终结论

做成一条 trace 展示出来。  
这会比只贴一段结果强很多。

---

## 小结

这节最重要的是建立一个作品级判断：

> **数据分析 Agent 的真正亮点，不是会不会调用 pandas，而是能否把输入数据、中间计算和最终洞察组织成可复核的分析闭环。**

只要这条闭环清楚，这个项目会非常适合展示你对多工具 Agent 的理解。

---

## 练习

1. 给示例数据再加一个 `date` 字段，把项目扩成简单时间趋势分析。
2. 想一想：为什么“可复核性”对数据分析 Agent 特别重要？
3. 如果结论和数字对不上，这个项目最可能出问题的层在哪？
4. 如果做作品集展示，你会把哪一块设计成最显眼的部分？
