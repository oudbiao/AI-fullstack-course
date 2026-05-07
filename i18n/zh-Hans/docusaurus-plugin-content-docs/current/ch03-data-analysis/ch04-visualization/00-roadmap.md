---
title: "3.4.1 可视化路线图：先选图表，再调样式"
sidebar_position: 16
description: "紧凑版可视化路线图：为趋势、对比、分布、关系和报告选择合适图表。"
keywords: [数据可视化, 图表选择, matplotlib, seaborn, plotly, 可视化路线图]
---

# 3.4.1 可视化路线图：先选图表，再调样式

可视化不是装饰。它把分析结果变成别人能快速看懂的图。

## 3.4.1.1 先看选图地图

![数据可视化路线图](/img/course/ch03-visualization-roadmap.png)

先用这个判断：

| 想表达... | 先用... |
|---|---|
| 随时间变化 | 折线图 |
| 类别对比 | 柱状图 |
| 分布 | 直方图或箱线图 |
| 两个数值的关系 | 散点图 |
| 相关矩阵 | 热力图 |

图表类型选对之后，再处理标题、坐标轴、图例、颜色和标注。

## 3.4.1.2 先跑一次图表

创建 `visual_first_loop.py`，安装 `pandas` 和 `matplotlib` 后运行。

```python
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.DataFrame(
    {
        "month": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "amount": [120, 180, 160, 220],
    }
)

ax = sales.plot(x="month", y="amount", marker="o", legend=False)
ax.set_title("Monthly sales")
ax.set_xlabel("Month")
ax.set_ylabel("Amount")
plt.tight_layout()
plt.savefig("sales_trend.png", dpi=150)

print("saved: sales_trend.png")
```

预期输出：

```text
saved: sales_trend.png
```

打开图片，只检查一件事：读者能不能在三秒内看出趋势？

## 3.4.1.3 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [3.4.2 Matplotlib 基础](./01-matplotlib.md) | Figure、Axes、折线/柱状/散点 |
| 2 | [3.4.3 Seaborn 统计可视化](./02-seaborn.md) | 更快做探索性图表 |
| 3 | [3.4.5 可视化最佳实践](./04-best-practices.md) | 选图、标签、颜色、误导性图表 |
| 4 | [3.4.4 Plotly 交互式可视化](./03-plotly.md) | 项目需要交互时再用 |

## 3.4.1.4 通过标准

能从一个数据集做出 4 张有用图表，并说清楚每张图为什么这样选，就算通过。
