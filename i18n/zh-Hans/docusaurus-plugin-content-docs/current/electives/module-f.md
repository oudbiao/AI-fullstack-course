---
title: "E.F AI 产品设计思维"
sidebar_position: 6
description: "在开工前用价值、成本、风险、体验和上线阻断条件评估 AI 产品想法。"
keywords: [AI product design, product thinking, evaluation, cost, UX, product strategy]
---

# E.F AI 产品设计思维

AI 产品设计从用户问题开始，不是从模型能力开始。一个功能是否值得做，要能说清价值、成本、风险和用户体验。

## 先看决策闭环

![AI 产品决策矩阵](/img/course/elective-ai-product-decision-matrix.webp)

![AI 产品实验与指标闭环](/img/course/elective-ai-product-experiment-metrics-loop.webp)

第一个产品习惯是：在实现前把取舍讲清楚。

## 运行一个小型优先级评分

```python
ideas = [
    {"name": "AI Tutor", "value": 9, "cost": 6, "risk": 4, "ux": 8},
    {"name": "AI Customer Service", "value": 8, "cost": 5, "risk": 5, "ux": 7},
    {"name": "AI Code Review", "value": 7, "cost": 4, "risk": 6, "ux": 6},
    {"name": "AI Medical Diagnosis", "value": 9, "cost": 8, "risk": 9, "ux": 5},
]


def score(item):
    return round(
        item["value"] * 0.45
        + (10 - item["cost"]) * 0.2
        + (10 - item["risk"]) * 0.2
        + item["ux"] * 0.15,
        2,
    )


def decision(item):
    if item["risk"] >= 8:
        return "do_not_launch"
    return "pilot" if item["score"] >= 6 else "wait"


ranked = sorted(({**item, "score": score(item)} for item in ideas), key=lambda item: item["score"], reverse=True)

for item in ranked:
    print(item["name"], "score=", item["score"], "decision=", decision(item))
```

预期输出：

```text
AI Tutor score= 7.25 decision= pilot
AI Customer Service score= 6.65 decision= pilot
AI Code Review score= 6.05 decision= pilot
AI Medical Diagnosis score= 5.4 decision= do_not_launch
```

分数不是最终真理。它强迫你说清自己在优化什么，以及什么情况下必须阻止上线。

## 产品检查清单

| 问题 | 好答案 |
|---|---|
| 谁卡住了？ | 明确用户群和任务 |
| 改善什么？ | 完成率、节省时间、质量或成本 |
| 会出什么问题？ | 风险边界和人工兜底 |
| 如何证明进展？ | 指标或用户测试结果 |

## 通过标准

能给一个 AI 功能想法打分，解释取舍，定义成功指标，并说出一个不应上线的条件，就算通过本选修。
