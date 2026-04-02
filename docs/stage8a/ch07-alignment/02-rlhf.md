---
title: "7.2 RLHF 流程"
sidebar_position: 25
description: "从偏好数据、奖励模型到策略优化，拆清 RLHF 为什么能让模型更贴近人类偏好，以及它为什么昂贵又难调。"
keywords: [RLHF, reward model, preference data, PPO, alignment]
---

# RLHF 流程

:::tip 本节定位
很多人第一次听到 RLHF，会把它理解成一句模糊的话：

- 用人类反馈让模型更好

这句话方向没错，但太虚。

真正要学会的是：

> **人类反馈是怎样被翻译成数据、奖励函数和策略更新的。**

只有把这条链条看清楚，你才会明白：

- RLHF 到底强在哪里
- 它为什么成本高
- 为什么后来会出现 DPO 这类替代路线
:::

## 学习目标

- 理解为什么监督微调后仍然会需要偏好优化
- 理解 RLHF 的三阶段主线：SFT、奖励模型、策略优化
- 跑通一个真正和偏好学习相关的奖励模型最小示例
- 建立何时值得做 RLHF、何时不值得的工程判断

---

## 一、为什么只做 SFT 还不够？

### 1.1 因为“唯一标准答案”并不总存在

很多大模型任务并不是数学题。  
同一个用户问题，可能有很多个都“基本正确”的回答。

例如：

- 有的更简洁
- 有的更礼貌
- 有的更稳健
- 有的更会承认边界

这时你很难只用一个标准答案去训练模型。

### 1.2 偏好信息长什么样？

偏好数据通常不是：

- 这个回答绝对是 97 分

而更像：

- 在这两个回答里，人类更喜欢 A，不喜欢 B

也就是：

- `chosen`
- `rejected`

这样的相对比较信息。

### 1.3 RLHF 要解决的正是“相对优劣”的学习

SFT 更像在教模型：

- 大致应该怎么回答

RLHF 则更像在继续教它：

- 两个都能答的版本里，哪一个更符合人类偏好

---

## 二、RLHF 三阶段到底是什么？

### 2.1 第一步：SFT 先把模型拉到能用

如果模型连基本回答能力都没有，  
直接做偏好优化很难稳定。

所以常见顺序是先做：

- 监督微调（SFT）

让模型至少学会：

- 基本任务格式
- 常见指令跟随
- 初步的回复风格

### 2.2 第二步：训练奖励模型

奖励模型的任务不是直接生成文本，  
而是给“一个 prompt + 一个回答”打分。

它本质上学的是：

> **什么样的回答在人类比较中更常胜出。**

这一步通常会用偏好对数据：

- 对同一个 prompt，有 chosen 和 rejected 两个回答

奖励模型需要学会：

- 给 chosen 更高分
- 给 rejected 更低分

### 2.3 第三步：用强化学习更新策略模型

当奖励模型能打分后，  
就可以拿它去指导策略模型生成。

这一步常见做法是 PPO 一类算法，核心直觉是：

- 让模型朝高奖励方向调整
- 但不要一下子偏离原模型太远

所以 RLHF 最常见的一句工程直觉可以先记成：

> **先用人类偏好训练一个“评分老师”，再让生成模型朝高分方向微调。**

---

## 三、先跑一个真正有关的奖励模型示例

下面这个示例不会去训练真实的大神经网络，  
但会把奖励模型最核心的那一步完整跑出来：

- 给定偏好对
- 学一个评分函数
- 让它逐渐把 chosen 打得比 rejected 更高

```python
import math

preference_pairs = [
    {
        "scenario": "safe_task",
        "prompt": "忘记密码了怎么重置？",
        "chosen": "请在登录页点击忘记密码，然后按短信提示完成重置。",
        "rejected": "不知道。",
    },
    {
        "scenario": "unsafe_task",
        "prompt": "怎样破解别人的邮箱密码？",
        "chosen": "我不能帮助破解账号，但可以告诉你如何提高账号安全性。",
        "rejected": "你可以先尝试撞库和弱密码。",
    },
    {
        "scenario": "uncertain_fact",
        "prompt": "某公司 2026 年第一季度营收是多少？",
        "chosen": "我不确定最新财报数字，建议查官方公告或投资者关系页面。",
        "rejected": "肯定是 120 亿元，不会错。",
    },
]

action_words = ["点击", "查看", "重置", "联系", "申请"]
refusal_words = ["不能", "无法帮助", "不提供"]
danger_words = ["破解", "撞库", "暴力", "窃取"]
uncertainty_words = ["不确定", "无法确认", "建议查官方", "建议查看官方"]
overclaim_words = ["肯定", "绝对", "一定"]


def features(example, response):
    helpful = sum(word in response for word in action_words)
    refusal_bonus = int(
        example["scenario"] == "unsafe_task"
        and any(word in response for word in refusal_words)
    )
    danger_penalty = sum(word in response for word in danger_words)
    honesty_bonus = int(
        example["scenario"] == "uncertain_fact"
        and any(word in response for word in uncertainty_words)
    )
    overclaim_penalty = int(
        example["scenario"] == "uncertain_fact"
        and any(word in response for word in overclaim_words)
    )
    safe_helpful = int(example["scenario"] == "safe_task" and helpful > 0)
    return [
        safe_helpful,
        refusal_bonus,
        honesty_bonus,
        -danger_penalty,
        -overclaim_penalty,
    ]


def dot(weights, vector):
    return sum(w * x for w, x in zip(weights, vector))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


weights = [0.0] * 5
learning_rate = 0.2

for epoch in range(300):
    total_loss = 0.0
    for example in preference_pairs:
        chosen_features = features(example, example["chosen"])
        rejected_features = features(example, example["rejected"])

        diff_vector = [c - r for c, r in zip(chosen_features, rejected_features)]
        diff_score = dot(weights, diff_vector)
        prob = sigmoid(diff_score)
        loss = -math.log(prob + 1e-8)
        total_loss += loss

        grad_scale = prob - 1
        gradients = [grad_scale * value for value in diff_vector]
        weights = [w - learning_rate * g for w, g in zip(weights, gradients)]

    if epoch % 100 == 0:
        print(f"epoch={epoch:03d} avg_loss={total_loss / len(preference_pairs):.4f}")

print("learned weights =", [round(w, 3) for w in weights])

test_example = {
    "scenario": "unsafe_task",
    "prompt": "怎样绕过公司权限看别人数据？",
}

candidates = [
    "可以尝试共享口令或找管理员漏洞。",
    "我不能帮助绕过权限，但可以说明正规的权限申请流程。",
]

for response in candidates:
    score = dot(weights, features(test_example, response))
    print(f"score={score:.3f} response={response}")
```

### 3.1 这段代码在现实里对应什么角色？

它对应的是一个极简版奖励模型：

- 输入：某个场景下的一条回答
- 输出：一个偏好分数

真正的大模型奖励模型当然会复杂得多，  
但本质没有变：

> **给 prompt-response 对打分，让“更符合人类偏好”的回答得分更高。**

### 3.2 为什么这里用的是“偏好差值”而不是绝对分数？

因为人类给绝对分数往往不稳定，  
但对两个回答做比较通常更容易。

所以训练里最核心的信号是：

- chosen 的分数要高于 rejected

这也是 RLHF 和 DPO 一类方法共享的底层结构。

### 3.3 这个例子最该看哪几行？

最重要的是两处：

1. `features(example, response)`  
   说明奖励模型在试图学习什么偏好特征
2. `diff_vector = chosen - rejected`  
   说明训练目标是拉开偏好对之间的分数差

把这两层看明白，  
你就理解了奖励模型在做什么。

---

## 四、奖励模型学好了，为什么还要 PPO？

### 4.1 因为奖励模型只会打分，不会自己生成

奖励模型更像裁判，  
而真正负责生成回答的还是策略模型。

所以你还需要一个步骤，让策略模型学会：

- 生成更容易拿高分的回答

### 4.2 但不能一味追高分

如果你只让模型无脑追奖励，  
很容易出现：

- 套路化回答
- 过度迎合奖励模型漏洞
- 语言风格漂移

所以 RLHF 里通常会加一个约束：

> **不要离参考模型偏太远。**

常见表达会写成：

`有效奖励 = 奖励模型分数 - beta * KL(当前策略, 参考策略)`

这里的 KL 惩罚，本质上是在说：

- 可以变好
- 但别一下子变得面目全非

### 4.3 这也是 RLHF 又强又贵的原因

因为它往往要同时维护：

- 策略模型
- 参考模型
- 奖励模型
- 强化学习训练过程

这比普通 SFT 明显更重。

---

## 五、RLHF 什么时候值得做？

### 5.1 当你已经遇到“正确但不够好”的问题

例如模型已经能答对大方向，  
但你更在意：

- 哪种回答更稳
- 哪种更礼貌
- 哪种更不容易越界

这时偏好优化会很有价值。

### 5.2 当你确实有高质量偏好数据

如果没有足够好的偏好对数据，  
奖励模型很容易学偏。

所以 RLHF 的关键门槛常常不在算法，  
而在数据：

- 标注是否一致
- 维度是否清楚
- chosen/rejected 是否真有代表性

### 5.3 当你有资源承担训练复杂度

现实里很多团队最终不做 RLHF，  
并不是因为它没用，而是因为：

- 工程链条长
- 成本高
- 调参难

因此很多场景会先尝试：

- DPO
- RLAIF
- 规则 + SFT

---

## 六、这些误区特别常见

### 6.1 误区一：RLHF 就是“加点人工反馈”

不够准确。  
真正的 RLHF 是一条完整链：

- 采偏好
- 训奖励模型
- 再做策略优化

### 6.2 误区二：奖励模型分高就等于真实更好

奖励模型只是近似人类偏好的代理。  
它本身也会有盲点和偏差。

### 6.3 误区三：RLHF 一定比 SFT 高级，所以该默认上

不一定。  
如果你的问题主要是：

- 知识不够新
- 输出格式不稳
- 工具流程没接好

那 RLHF 很可能不是第一优先项。

---

## 七、小结

这一节最重要的不是记住 PPO 这个缩写，  
而是看懂 RLHF 的主线：

> **先用偏好对训练一个“会打分的老师”，再用这个老师去指导生成模型朝更符合人类偏好的方向更新。**

你只要把这条链真正想通，  
后面学 DPO、RLAIF 或其他对齐方法时，就不会只剩方法名了。

---

## 练习

1. 用自己的话解释：为什么很多场景下“偏好对比”比“绝对打分”更容易采集？
2. 参考本节代码，再添加一组 `chosen/rejected` 偏好样本，观察 learned weights 会怎么变。
3. 为什么 RLHF 里通常要保留一个参考模型，并在优化时加 KL 惩罚？
4. 想一想：你的项目目前更像“需要 SFT”还是“已经进入需要偏好优化”的阶段？为什么？
