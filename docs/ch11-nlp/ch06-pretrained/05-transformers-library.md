---
title: "6.6 Transformers 库实战"
sidebar_position: 20
description: "从 tokenizer、config、model 到最小 pipeline，真正学会怎样离线使用 HuggingFace Transformers 的核心接口。"
keywords: [transformers, HuggingFace, tokenizer, AutoModel, pipeline, config]
---

# Transformers 库实战

:::tip 本节定位
学预训练模型只停留在概念层很容易“会说不会用”。  
这一节要解决的是另一个很现实的问题：

> **面对 `transformers` 这个库，我到底该从哪里下手？**

我们会尽量用不依赖外网下载的方式，把最核心接口走通。
:::

## 学习目标

- 理解 `transformers` 库里最常见的几个对象分别干什么
- 学会区分 tokenizer、config、model、pipeline 的角色
- 离线跑通一个 tokenizer + model 的最小示例
- 理解实际项目里怎样从“能跑”走向“可维护”

---

## 一、先把库里的几个主角分清楚

### 1.1 `Tokenizer`

负责把文本变成模型能吃的数字序列。

### 1.2 `Config`

负责描述模型结构参数，比如：

- hidden size
- 层数
- 头数

### 1.3 `Model`

真正负责前向计算。

### 1.4 `Pipeline`

是更高层的封装，帮你把：

- 分词
- 前向
- 后处理

串成一个更方便调用的接口。

一句话记：

> tokenizer 负责进门，model 负责计算，pipeline 负责把整件事串起来。 

---

## 二、为什么很多人第一次用 `transformers` 会晕？

因为它有两层世界：

### 2.1 概念层

你知道：

- BERT 是 encoder-only
- GPT 是 decoder-only

### 2.2 工具层

你又会遇到：

- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `pipeline`
- `from_pretrained`

初学者经常会卡在：

> 名字太多，但不知道每个接口到底解决什么问题。 

所以这一节的核心不是背接口，而是建立调用顺序的地图。

---

## 三、先离线构造一个最小 tokenizer

### 3.1 为什么不直接下载现成模型？

因为教程要尽量保证你在没有外网时也能跑通。  
所以这里我们手工准备一个超小 `vocab.txt`，让你真正理解 tokenizer 在做什么。

### 3.2 可运行示例

```python
from pathlib import Path
from transformers import BertTokenizer

vocab_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "我", "爱", "自", "然", "语", "言", "处", "理", "北", "京"
]

vocab_path = Path("mini_vocab.txt")
vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

tokenizer = BertTokenizer(vocab_file=str(vocab_path))

encoded = tokenizer("我爱自然语言处理", return_tensors="pt")

print(encoded)
print("tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
```

### 3.3 这段代码在教你什么？

它在教你：

- tokenizer 不是魔法黑盒
- 它本质上是“词表 + 切分规则 + 编码规则”
- 输出里最关键的是 `input_ids` 和 `attention_mask`

---

## 四、再离线构造一个最小 BERT 模型

### 4.1 为什么用随机初始化模型？

因为我们现在的目标不是追求效果，而是：

> 真正看懂 `transformers` 库里 model 对象是怎么接输入、吐输出的。 

### 4.2 可运行示例

```python
import torch
from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=15,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64
)

model = BertModel(config)

input_ids = torch.tensor([[2, 5, 6, 7, 8, 9, 10, 11, 12, 3]])
attention_mask = torch.ones_like(input_ids)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("last_hidden_state shape:", outputs.last_hidden_state.shape)
print("pooler_output shape    :", outputs.pooler_output.shape)
```

### 4.3 你真正该看懂什么？

- `input_ids` 是 token 编号
- `attention_mask` 告诉模型哪些位置有效
- `last_hidden_state` 是每个位置的上下文化表示
- `pooler_output` 更像句级表示之一

这几样东西看懂了，后面你再接分类头、匹配头、生成头就会轻松很多。

---

## 五、把 tokenizer 和 model 串起来

### 5.1 可运行示例

```python
from pathlib import Path
import torch
from transformers import BertTokenizer, BertConfig, BertModel

vocab_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "我", "爱", "自", "然", "语", "言", "处", "理"
]

vocab_path = Path("mini_vocab_2.txt")
vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

tokenizer = BertTokenizer(vocab_file=str(vocab_path))

config = BertConfig(
    vocab_size=len(vocab_tokens),
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64
)
model = BertModel(config)

batch = tokenizer(["我爱自然语言处理", "我爱语言"], padding=True, return_tensors="pt")
outputs = model(**batch)

print("input_ids shape        :", batch["input_ids"].shape)
print("attention_mask shape   :", batch["attention_mask"].shape)
print("last_hidden_state shape:", outputs.last_hidden_state.shape)
```

### 5.2 这就是最基础的真实调用链

真正项目里，最常见的底层流程就是：

1. 文本 -> tokenizer
2. tokenizer -> tensor
3. tensor -> model
4. model 输出 -> 后处理或任务头

你现在已经把这条链跑通了。

---

## 六、`Auto*` 系列接口是干什么的？

### 6.1 为什么库里这么多 `AutoModel`？

因为 `transformers` 想让你不用手写一堆模型类型判断。

例如：

- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `AutoModelForCausalLM`

它们的设计目标是：

> 给一个模型名字或配置，就自动选合适的类。 

### 6.2 一个离线的 `AutoModel.from_config` 示例

```python
from transformers import AutoModel, BertConfig

config = BertConfig(
    vocab_size=20,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    intermediate_size=32
)

model = AutoModel.from_config(config)
print(type(model))
```

这说明：

- `AutoModel` 不一定非要联网下载
- 它本质上是“根据配置自动实例化正确模型”

---

## 七、`pipeline` 到底值不值得学？

### 7.1 值得，但你要知道它适合什么阶段

`pipeline` 的优点：

- 上手快
- Demo 快
- 少写样板代码

它更适合：

- 学习
- 快速验证
- 小实验

### 7.2 但工程里不能只靠它

因为真实项目往往还要控制：

- batch
- device
- 输出格式
- 日志
- 错误处理

所以真正成熟的做法通常是：

- 学会 `pipeline`
- 但也要理解底层 tokenizer + model 调用链

---

## 八、Transformers 库最常见的任务头

粗略记几个高频接口：

| 接口 | 适合什么 |
|---|---|
| `AutoModel` | 只拿基础表示 |
| `AutoModelForSequenceClassification` | 文本分类 |
| `AutoModelForTokenClassification` | 序列标注 |
| `AutoModelForQuestionAnswering` | 抽取式问答 |
| `AutoModelForCausalLM` | 生成任务 |

这背后的逻辑其实很简单：

> 相同骨干模型 + 不同任务头。 

---

## 九、初学者最常踩的坑

### 9.1 只会 `pipeline`，不会底层调用

这样一到工程场景就容易卡住。

### 9.2 不理解 tokenizer 输出字段

至少要看懂：

- `input_ids`
- `attention_mask`

### 9.3 把“模型概念”和“库接口”混在一起

你要能分清：

- BERT / GPT 是模型路线
- `AutoModel` / `pipeline` 是库接口

---

## 小结

这一节最重要的不是会不会背 API，而是搞清楚：

> **Transformers 库的核心调用链，就是 tokenizer 把文本编码成张量，model 对张量做前向计算，再由任务头或后处理得到结果。**

一旦这条链清楚了，后面无论你做分类、抽取、生成还是微调，思路都会稳定很多。

---

## 练习

1. 修改本节的 mini vocab，加入你自己的几个词，再看 tokenizer 输出有什么变化。
2. 把 `BertConfig` 的 `hidden_size` 改成 64，看看输出 shape 怎么变。
3. 用自己的话解释：为什么学 `transformers` 库时，不能只会 `pipeline`？
4. 想一想：如果你要做文本分类，应该优先找 `AutoModel` 还是 `AutoModelForSequenceClassification`？
