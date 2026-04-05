---
title: "6.4 其他 PEFT 方法【选修】"
sidebar_position: 21
description: "从 Prompt Tuning、Prefix Tuning、Adapter 到 IA3，理解除了 LoRA 之外，参数高效微调还可以把可训练部分放在哪里。"
keywords: [PEFT, prompt tuning, prefix tuning, adapter, IA3, finetuning]
---

# 其他 PEFT 方法【选修】

:::tip 本节定位
前一节我们已经知道了 LoRA 和 QLoRA 的主线：  
不重训整个大模型，而是只训练少量增量参数。

但 PEFT 不是只有 LoRA 一条路。真正该问的问题其实是：

> **我们到底想把“可训练能力”放在模型的哪里？**

- 放在输入侧，可以变成 Prompt Tuning
- 放在每一层的上下文前缀，可以变成 Prefix Tuning
- 放在层与层之间的小模块，可以变成 Adapter
- 放在中间激活的缩放系数上，可以变成 IA3

这一节就是把这几条支线整理成一张能真正拿来选型的地图。
:::

## 学习目标

- 理解 Prompt Tuning、Prefix Tuning、Adapter、IA3 分别在改哪里
- 知道这些方法和 LoRA 的核心差异
- 跑通一个真正与 PEFT 主题相关的 Adapter 最小训练示例
- 建立多任务、低显存、快速切换场景下的选型直觉

---

## 一、为什么 LoRA 不是唯一答案？

### 1.1 PEFT 真正想解决的不是“发明缩写”

PEFT 的根问题很朴素：

> **冻结大模型主体，只训练很小一部分参数，还能不能把模型拉向新任务？**

只要这个目标不变，“训练哪一小部分参数”就会自然衍生出很多变体。

所以这些方法之间最大的区别，不是名字，而是：

- 可训练参数放在哪里
- 它会影响模型的哪一段信息流
- 训练成本、推理成本、可复用性分别怎样

### 1.2 一个类比：给同一台电脑做轻量改造

你可以把基础模型想成一台已经装好的电脑：

- Prompt Tuning 像是给开机桌面多放几张“隐藏便签”
- Prefix Tuning 像是给每个软件启动前都预先塞一点上下文
- Adapter 像是在主板上插一个很小的扩展卡
- IA3 像是在几个关键旋钮上加可调节的增益控制

它们都不是重做整台电脑，  
而是在不同位置加一层可调结构。

### 1.3 为什么现实项目里会需要这些分支？

因为工程上的约束并不完全一样：

- 有的团队最在意显存
- 有的团队最在意多任务热切换
- 有的团队最在意推理时不要额外拖慢
- 有的团队想让同一个底座挂很多领域适配器

同样是 PEFT，最优方法未必相同。

---

## 二、先把 PEFT 家族地图理清楚

### 2.1 Prompt Tuning：把可训练部分放在输入前面

Prompt Tuning 的直觉是：

> **不改模型层内部结构，而是在输入 embedding 前面接上一小段可训练“软提示”。**

这里的 prompt 不是你手写的自然语言，而是一组可训练向量。

它的优点是：

- 参数极少
- 实现概念清晰
- 适合任务数量很多、每个任务适配都要很轻的时候

它的局限是：

- 对复杂任务的改造力度有限
- 主要影响输入端，不像层内改造那样深入

### 2.2 Prefix Tuning：给每一层加“前缀上下文”

Prefix Tuning 比 Prompt Tuning 更进一步。

它不是只在输入最前面加向量，而是：

> **给 Transformer 每一层的注意力模块额外准备一段可训练的 key/value 前缀。**

你可以把它理解成：

- Prompt Tuning 更像只在开头塞一句“任务说明”
- Prefix Tuning 则像是每一层在做注意力时，都能看到一段额外的上下文提示

因此它通常比 Prompt Tuning 更有表达力。

### 2.3 Adapter：在层与层之间插小模块

Adapter 的思路很适合新手理解，因为它最像“明确加了一个插件”。

常见结构是：

1. 原始隐藏状态先经过一个降维层
2. 中间做非线性变换
3. 再升回原来的维度
4. 通过残差连接加回主干

也就是：

> **主干冻结，旁边多插一个很小的可训练旁路。**

它的工程优点很明显：

- 不用大改原模型主体
- 各个任务可以挂不同 adapter
- 多任务切换时只换小模块即可

### 2.4 IA3：不学大矩阵，而学“缩放系数”

IA3 的思路更节制：

> **不插小网络，也不学大增量，而是只学习少量的逐通道缩放向量。**

例如在注意力输出或前馈层激活上做：

- 某些维度放大
- 某些维度压低

这意味着：

- 参数更少
- 训练更轻
- 但可表达能力也相对更克制

### 2.5 四种方法放在一起看

| 方法 | 可训练部分放在哪里 | 直觉 | 常见优点 | 常见局限 |
|---|---|---|---|---|
| Prompt Tuning | 输入 embedding 前 | 给模型塞一段软提示 | 参数极少 | 改造力度有限 |
| Prefix Tuning | 每层注意力的 KV 前缀 | 每层都能看到额外上下文 | 表达力比软提示更强 | 实现复杂度更高 |
| Adapter | 层间小瓶颈模块 | 插一个轻量插件 | 多任务切换方便 | 推理会多一小段计算 |
| IA3 | 激活缩放向量 | 调关键通道的增益 | 参数极少、实现轻 | 对复杂变化的表达较弱 |

---

## 三、先跑一个真正和 PEFT 相关的 Adapter 示例

下面这个例子会做一件非常具体的事：

- 构造一个很小的文本分类任务
- 冻结基础编码器
- 只训练 Adapter 和分类头

这样你能直接看到：

- 主模型没动
- 少量参数照样能把任务学起来

:::info 运行提示
```bash
pip install torch
```
:::

```python
import torch
import torch.nn as nn

torch.manual_seed(42)

samples = [
    ("refund my order", 0),
    ("need a refund", 0),
    ("cancel and refund", 0),
    ("login failed again", 1),
    ("cannot login account", 1),
    ("password login problem", 1),
]
label_names = ["refund", "login"]

vocab = {"<pad>": 0}
for text, _ in samples:
    for token in text.split():
        if token not in vocab:
            vocab[token] = len(vocab)

max_len = max(len(text.split()) for text, _ in samples)


def encode(text):
    ids = [vocab[token] for token in text.split()]
    ids += [0] * (max_len - len(ids))
    return ids


x = torch.tensor([encode(text) for text, _ in samples], dtype=torch.long)
y = torch.tensor([label for _, label in samples], dtype=torch.long)


class FrozenBaseEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, token_ids):
        emb = self.embedding(token_ids)
        mask = (token_ids != 0).unsqueeze(-1)
        pooled = (emb * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        hidden = torch.tanh(self.proj(pooled))
        return hidden


class AdapterClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=16, bottleneck_dim=4, num_labels=2):
        super().__init__()
        self.base = FrozenBaseEncoder(vocab_size, hidden_dim)
        self.adapter_down = nn.Linear(hidden_dim, bottleneck_dim)
        self.adapter_up = nn.Linear(bottleneck_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, token_ids):
        hidden = self.base(token_ids)
        adapted = hidden + self.adapter_up(torch.tanh(self.adapter_down(hidden)))
        logits = self.classifier(adapted)
        return logits


model = AdapterClassifier(vocab_size=len(vocab))
optimizer = torch.optim.Adam(
    [p for p in model.parameters() if p.requires_grad],
    lr=0.05,
)
criterion = nn.CrossEntropyLoss()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("total params     =", total_params)
print("trainable params =", trainable_params)

for step in range(201):
    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 50 == 0:
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean().item()
        print(f"step={step:03d} loss={loss.item():.4f} acc={acc:.2f}")

with torch.no_grad():
    preds = model(x).argmax(dim=-1)
    for text, pred in zip([text for text, _ in samples], preds.tolist()):
        print(f"{text:22s} -> {label_names[pred]}")
```

### 3.1 这段代码到底在教什么？

它不是在教你“怎么做一个完整生产级微调”，而是在刻意把重点钉在 Adapter 本身：

- `FrozenBaseEncoder` 全部冻结
- `adapter_down` 和 `adapter_up` 是新增小模块
- `classifier` 负责把适配后的表示映射到标签

真正关键的是这一句：

```python
adapted = hidden + self.adapter_up(torch.tanh(self.adapter_down(hidden)))
```

这就是典型的 Adapter 思路：

- 主干表示先保留
- 旁边走一个小瓶颈分支
- 再以残差方式加回去

### 3.2 为什么这比“只打印方法名”强得多？

因为你现在能直接观察三件事：

1. 可训练参数只占很小一部分
2. 主模型不动，任务仍然能拟合
3. 新增能力来自插入的小模块，而不是重训整网

这三点正是 Adapter 的本体。

---

## 四、再看三个更短的结构示意

### 4.1 Prompt Tuning：输入前面拼接软提示

```python
import torch

token_embeddings = torch.randn(1, 5, 8)
soft_prompt = torch.randn(1, 3, 8, requires_grad=True)

combined = torch.cat([soft_prompt, token_embeddings], dim=1)
print("原始长度:", token_embeddings.shape[1])
print("拼接后长度:", combined.shape[1])
```

这里最该记住的是：

- soft prompt 本身不是可读文本
- 它是一组训练出来的向量
- 模型看到的是“额外输入 token 的 embedding”

### 4.2 Prefix Tuning：不是改输入长度，而是改每层注意力上下文

```python
import torch

layer_keys = torch.randn(1, 4, 8)
prefix_keys = torch.randn(1, 2, 8, requires_grad=True)

all_keys = torch.cat([prefix_keys, layer_keys], dim=1)
print("注意力原始 key 数量:", layer_keys.shape[1])
print("加入 prefix 后 key 数量:", all_keys.shape[1])
```

这个示意对应的直觉是：

- 普通注意力只看原序列
- Prefix Tuning 让每层注意力额外看到一段可训练前缀

### 4.3 IA3：不是加模块，而是给关键通道乘缩放因子

```python
import torch

hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
gate = torch.tensor([0.5, 1.0, 1.5, 2.0], requires_grad=True)

scaled = hidden * gate
print("before:", hidden)
print("after :", scaled)
```

IA3 的核心不是“变复杂”，而是“只在最关键的位置做轻量调节”。

---

## 五、到底该怎么选？

### 5.1 如果你最在意任务切换和模块化

优先想到：

- Adapter

因为它天然适合：

- 一个底座模型
- 挂很多小适配器
- 按任务切换加载

### 5.2 如果你最在意参数再少一点

可以先关注：

- Prompt Tuning
- IA3

这类方法非常轻，但要注意：

- 参数更少不等于效果一定更好
- 任务复杂时，表达能力可能不够

### 5.3 如果你希望干预更深入一些

可以看：

- Prefix Tuning

因为它影响的不只是输入最前面，而是每一层注意力读上下文的方式。

### 5.4 如果你想要一个“默认优先尝试”的工业方案

现实里很多团队还是会先尝试：

- LoRA / QLoRA

原因很简单：

- 生态成熟
- 工具链丰富
- 社区经验多

所以这节不是要你抛弃 LoRA，  
而是让你知道：

> **LoRA 只是 PEFT 地图里最常用的一块，不是全部。**

---

## 六、这些误区特别常见

### 6.1 误区一：参数越少越高级

不一定。  
参数少意味着：

- 训练便宜

但也可能意味着：

- 表达力更受限制

### 6.2 误区二：方法名越多越说明自己懂了

真正要会的是：

- 它改哪里
- 影响哪段信息流
- 为什么适合当前任务

### 6.3 误区三：把“可训练模块”当成唯一重点

别忘了任务成败还强依赖：

- 数据质量
- 模板格式
- 评估方式
- 是否真的需要微调

---

## 小结

这一节最该带走的不是四个名词，而是一条主线：

> **PEFT 的本质，是在尽量不动大模型主体的前提下，选择一个合适的位置放入少量可训练能力。**

你以后再遇到新的 PEFT 变体时，也可以先用同样的问题去拆：

1. 它把可训练参数放在哪？
2. 它会影响输入、层内还是层间？
3. 它换来了什么工程收益，又牺牲了什么？

只要这三件事看清楚，方法名就不再会显得神秘。

---

## 练习

1. 用自己的话解释 Prompt Tuning、Prefix Tuning、Adapter、IA3 分别改的是模型的哪一部分。
2. 如果你要给一个底座模型同时适配 20 个不同业务任务，为什么 Adapter 往往很有吸引力？
3. 把本节 Adapter 代码中的 `bottleneck_dim` 改大或改小，观察可训练参数数量怎么变化。
4. 想一想：如果你的硬件很紧张，但任务又比较复杂，你会优先尝试哪种 PEFT 方法？为什么？
