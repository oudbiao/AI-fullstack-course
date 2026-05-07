---
title: "E.F AI 产品设计思维"
sidebar_position: 6
description: "AI 产品判断的简明实操指南：先定义用户问题，再评估价值、成本、风险和体验。"
keywords: [AI 产品设计, 产品思维, 评估, 成本, UX, 产品策略]
---

# E.F AI 产品设计思维

AI 产品设计从用户问题开始，而不是从模型能力开始。一个功能是否值得做，取决于价值、成本、风险和体验能否说清楚。

## 先看决策闭环

![AI 产品决策矩阵](/img/course/elective-ai-product-decision-matrix.png)

![AI 产品实验与指标闭环](/img/course/elective-ai-product-experiment-metrics-loop.png)

第一个产品习惯是：实现之前先把取舍写出来。

## 跑一个小优先级评分

```python
ideas = [
    {"name": "AI Tutor", "value": 9, "cost": 6, "risk": 4, "ux": 8},
    {"name": "AI Customer Service", "value": 8, "cost": 5, "risk": 5, "ux": 7},
    {"name": "AI Code Review", "value": 7, "cost": 4, "risk": 6, "ux": 6},
]

def score(item):
    return round(item["value"] * 0.45 + (10 - item["cost"]) * 0.2 + (10 - item["risk"]) * 0.2 + item["ux"] * 0.15, 2)

ranked = sorted(({**item, "score": score(item)} for item in ideas), key=lambda item: item["score"], reverse=True)

for item in ranked:
    print(item["name"], item["score"])
```

预期输出：

```text
AI Tutor 7.25
AI Customer Service 6.65
AI Code Review 6.05
```

这些数字不是真理。它们的作用是逼你说清楚自己在优化什么。

## 产品检查表

| 问题 | 好答案 |
|---|---|
| 谁卡住了？ | 具体用户群和具体任务 |
| 改善什么？ | 完成率、省时、质量或成本 |
| 会出什么问题？ | 风险边界和人工兜底 |
| 怎样证明进展？ | 指标或用户测试结果 |

## 通过标准

你能给一个 AI 功能想法打分，解释取舍，定义成功指标，并说出至少一种不应该上线的情况，就算通过本选修。
