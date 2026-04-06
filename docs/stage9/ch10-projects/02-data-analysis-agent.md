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

## 先建立一张地图

数据分析 Agent 更适合按“读数据 -> 算统计 -> 形成解释 -> 给出展示建议”来理解：

```mermaid
flowchart LR
    A["输入表格"] --> B["统计计算"]
    B --> C["生成洞察"]
    C --> D["图表建议 / 报告建议"]
```

所以这节真正想解决的是：

- 数据分析 Agent 为什么不只是“会调 pandas”
- 为什么可复核的中间过程会比最终一句结论更重要

---

## 一、项目题目怎么收窄？

建议先做成：

- 读取一个小表
- 算几个核心统计量
- 根据统计量生成洞察摘要

而不是一开始做成：

- 自动 BI 平台
- 全自动报告工厂

### 1.1 一个更适合新人的总类比

你可以把数据分析 Agent 理解成：

- 一个会先算、再讲、还会建议怎么画图的分析助理

它和普通计算器的差别不在于：

- 算得更快

而在于：

- 它能把数字组织成有解释力的结论

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

### 2.3 一个很适合初学者先记的项目检查表

| 环节 | 你最该先确认什么 |
|---|---|
| 输入数据 | 字段含义清不清楚 |
| 中间统计 | 计算口径是不是一致 |
| insight | 结论和数字能不能对上 |
| 图表建议 | 图表类型是不是贴数据形态 |

这个表很适合新人，因为它会把“数据分析 Agent”重新压回一条可检查的工作流。

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

### 4.2 再看一个最小“分析 trace”示例

```python
trace = {
    "input_rows": len(sales),
    "total_amount": result["total_amount"],
    "per_category_avg": result["per_category_avg"],
    "insight": result["insight"],
}

print(trace)
```

这个示例很适合初学者，因为它会帮助你看到：

- 数据分析 Agent 项目真正值钱的地方
- 往往在“过程能不能被复核”

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

### 6.3 一个新人很适合先记的评估表

| 维度 | 最值得先问什么 |
|---|---|
| 正确性 | 数字有没有算对 |
| 可复核性 | 中间过程能不能看回去 |
| 解释性 | 结论和统计有没有对上 |
| 展示性 | 图表建议和结论是否自然 |

这个表很适合新人，因为它会把“Agent 项目好不好”拆成几项更具体的判断。

---

## 小结

这节最重要的是建立一个作品级判断：

> **数据分析 Agent 的真正亮点，不是会不会调用 pandas，而是能否把输入数据、中间计算和最终洞察组织成可复核的分析闭环。**

只要这条闭环清楚，这个项目会非常适合展示你对多工具 Agent 的理解。

## 如果把它做成作品集，最值得展示什么

最值得展示的通常不是：

- 一句分析结论

而是：

1. 原始数据样例
2. 中间统计结果
3. insight 是怎么长出来的
4. 图表建议为什么这样给

这样别人会更容易看出：

- 你理解的是分析闭环
- 不只是让 Agent 说了一段话

---

## 练习

1. 给示例数据再加一个 `date` 字段，把项目扩成简单时间趋势分析。
2. 想一想：为什么“可复核性”对数据分析 Agent 特别重要？
3. 如果结论和数字对不上，这个项目最可能出问题的层在哪？
4. 如果做作品集展示，你会把哪一块设计成最显眼的部分？
