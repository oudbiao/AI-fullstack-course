---
title: "6.8.4 项目：生成模型实战 [选修]"
sidebar_position: 3
description: "构建生成项目评审闭环：sample checkpoints、质量记录、多样性检查、失败样本和作品集展示。"
keywords: [generative project, GAN, VAE, generation quality, diversity, evaluation]
---

# 6.8.4 项目：生成模型实战 [选修]

:::tip 本节定位
生成项目不是产出一张好看的样本就结束。你需要展示质量、多样性、稳定性、失败样本，以及为什么保留某个 checkpoint。
:::

## 学习目标

- 解释为什么生成项目的评估不同于分类。
- 同时追踪质量和多样性。
- 构建一个小型 checkpoint review table。
- 识别 mode collapse 和模糊输出。
- 把生成样本包装成项目证据。

---

## 先看评估闭环

![生成模型项目评估闭环图](/img/course/ch06-project-generative-eval-loop.webp)

```text
train -> sample checkpoints -> review quality + diversity -> keep failures -> choose next step
```

练习项目建议选择：

- 能肉眼检查；
- 足够小，能训练或模拟；
- checkpoint 之间容易比较。

数字、图标、简单形状、小灰度图案，都比开放式照片级生成更适合第一轮项目。

## 实验：Checkpoint Review Dashboard

创建 `generative_review_dashboard.py`：

```python
checkpoints = [
    {"epoch": 1, "quality": 0.20, "diversity": 0.80, "note": "mostly noise"},
    {"epoch": 10, "quality": 0.45, "diversity": 0.72, "note": "outlines appear"},
    {"epoch": 30, "quality": 0.68, "diversity": 0.60, "note": "usable but varied"},
    {"epoch": 60, "quality": 0.75, "diversity": 0.48, "note": "possible collapse"},
]

print("generation_review")
for row in checkpoints:
    status = "candidate" if row["quality"] >= 0.6 and row["diversity"] >= 0.55 else "review"
    print(
        f"epoch={row['epoch']:03d} "
        f"quality={row['quality']:.2f} "
        f"diversity={row['diversity']:.2f} "
        f"status={status}"
    )

selected = max(
    [row for row in checkpoints if row["diversity"] >= 0.55],
    key=lambda row: row["quality"],
)
print("selected_epoch:", selected["epoch"])
```

运行：

```bash
python generative_review_dashboard.py
```

预期输出：

```text
generation_review
epoch=001 quality=0.20 diversity=0.80 status=review
epoch=010 quality=0.45 diversity=0.72 status=review
epoch=030 quality=0.68 diversity=0.60 status=candidate
epoch=060 quality=0.75 diversity=0.48 status=review
selected_epoch: 30
```

![生成模型 checkpoint 评审结果图](/img/course/ch06-generative-checkpoint-selection-result-map.webp)

为什么不选 epoch 60？因为质量更高，但多样性更低。好的生成项目不能只选最漂亮的一张。

## 要保存什么

| 证据 | 为什么 |
|---|---|
| samples by checkpoint | 展示训练进展 |
| failure samples | 诚实展示边界 |
| diversity notes | 捕捉重复输出 |
| quality notes | 解释视觉改善 |
| training logs | 展示稳定或 collapse |
| final selection rule | 让选择可复现 |

## 质量、多样性、稳定性

| 维度 | 好信号 | 警告信号 |
|---|---|---|
| Quality | 样本像目标数据 | 噪声、模糊、结构破碎 |
| Diversity | 样本有 meaningful variation | 重复输出或单一风格 |
| Stability | checkpoint 逐步改善 | 突然 collapse 或震荡 |
| Interpretability | 记录失败原因 | 只展示最好样本 |

常见取舍：

```text
最好看的单个样本 != 最好的项目 checkpoint
```

## 项目升级路线

| 版本 | 增加什么 |
|---|---|
| basic | 一个模型、固定 sampling seed、checkpoint samples |
| standard | quality/diversity table 和 failure samples |
| challenge | 比较 VAE、GAN 或 diffusion-style 输出 |
| portfolio | 清楚故事：data、model、samples、failures、next step |

## 留下的证据

生成项目至少留下这些证据：

```text
checkpoint_samples: fixed-seed samples across epochs
quality_note: what improved visually
diversity_note: whether outputs repeat
failure_sample: blurry, broken, collapsed, or unrealistic output
selection_rule: why this checkpoint was kept
next_action: data, objective, architecture, or sampling change
```

## 常见错误

| 错误 | 修复 |
|---|---|
| 只展示最好样本 | 同时展示平均样本和失败样本 |
| 忽略多样性 | 跟踪重复输出或 unique patterns |
| checkpoint 比较不公平 | 使用同一组 fixed seed |
| 一开始数据集太复杂 | 从小视觉目标开始 |
| 不解释模型选择 | 说明为什么选 VAE、GAN 或其他方法 |

## 练习

1. 加一个 epoch `90`，quality `0.80`、diversity `0.30`。应该选它吗？
2. 给每个 checkpoint 增加 `failure` 字段。
3. 为你自己的生成项目想法写一个 4 行表格。
4. 用 checkpoint table 解释 mode collapse。
5. 写一个作品集小节标题：“为什么我选择这个 checkpoint”。

## 小结

- 生成项目需要评估故事，而不只是 gallery。
- 质量和多样性必须一起读。
- 失败样本会让项目更可信。
- 清楚的 checkpoint 选择规则也是交付物的一部分。
