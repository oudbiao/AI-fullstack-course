---
title: "2.3 大模型核心概念"
sidebar_position: 6
description: "用新人能读懂的方式理解 token、上下文、注意力、采样温度、预训练和指令跟随等核心概念。"
keywords: [token, 上下文窗口, attention, temperature, sampling, pretraining, LLM]
---

# 大模型核心概念

## 学习目标

完成本节后，你将能够：

- 理解 token、上下文窗口、next-token prediction 的含义
- 理解 embedding、logits、temperature 的直觉
- 看懂一个极简的注意力计算例子
- 分清预训练、微调、提示词驱动这几类能力来源

---

## 一、大模型到底在做什么？

先用最不容易误解的话说：

> **大语言模型本质上是在做“给定上下文，预测下一个 token”。**

这听起来朴素，但能力就是从这里长出来的。

比如你看到：

> “北京是中国的”

你很可能会接：

> “首都”

模型做的事，本质上也是类似的，只不过它是在超大规模语料上学会这种预测。

---

## 二、Token：模型真正看到的不是“句子”，而是切分后的单位

很多新人以为模型是按“字”或“词”看文本，其实不一定。

更准确地说：

> 模型看到的是 token。

token 可以是：

- 一个字
- 一个词
- 一个词的一部分
- 一个标点

### 一个玩具版 tokenizer

```python
text = "AI fullstack course"

# 这里只是最简单的空格切分，真实大模型 tokenizer 更复杂
tokens = text.split()

print("原文:", text)
print("tokens:", tokens)
print("token 数量:", len(tokens))
```

真实大模型通常会把文本切得更细，因为这样更利于处理生僻词和不同语言。

---

## 三、上下文窗口：模型一次能“看多远”

上下文窗口（context window）可以理解成模型的“当前工作台”。

工作台越大：

- 一次能放下的信息越多
- 模型越可能利用更长的历史内容

但它不是无限的。  
所以很多长文任务、RAG 任务都绕不开“上下文怎么塞进去”这个问题。

类比一下：

> 你在桌上做题，桌面越大，能摊开的参考资料越多。

---

## 四、Embedding：先把 token 变成向量

模型不能直接吃 token 字符串，所以要先变成向量。  
这个过程可以先理解成：

> **给每个 token 分配一个高维坐标。**

语义相近的 token，在好的表示空间里也会更接近。

虽然真实 embedding 很复杂，但我们先用一个小例子体会“文本转向量”的思想：

```python
import numpy as np

embedding_table = {
    "cat": np.array([0.9, 0.1, 0.2]),
    "dog": np.array([0.85, 0.15, 0.25]),
    "car": np.array([0.1, 0.8, 0.3])
}

print("cat embedding:", embedding_table["cat"])
print("dog embedding:", embedding_table["dog"])
print("car embedding:", embedding_table["car"])
```

这里只是玩具示意，但你已经能看出：

- `cat` 和 `dog` 更接近
- `car` 更远

---

## 五、模型为什么叫“自回归”？

因为它经常是这样生成文本的：

1. 看已有上下文
2. 预测下一个 token
3. 把这个新 token 拼回上下文
4. 再预测下一个

所以生成是一点点往后滚的。

就像你玩接龙游戏：

- 先说一个词
- 再根据前面的话继续往下接

---

## 六、logits、概率和 temperature

模型内部先算出来的通常不是“最终概率”，而是一组分数，常叫 `logits`。

然后经过 softmax，变成概率分布。

### 一个温度采样的可运行例子

```python
import numpy as np

tokens = ["北京", "上海", "广州"]
logits = np.array([3.0, 1.5, 0.5])

def softmax_with_temperature(logits, temperature=1.0):
    scaled = logits / temperature
    exp_values = np.exp(scaled - scaled.max())
    return exp_values / exp_values.sum()

for temp in [0.5, 1.0, 2.0]:
    probs = softmax_with_temperature(logits, temperature=temp)
    print(f"temperature={temp}")
    for token, prob in zip(tokens, probs):
        print(f"  {token}: {prob:.4f}")
```

### 怎么理解 temperature？

- 温度低：更保守，更偏向最高分选项
- 温度高：更发散，更容易尝试次优选项

类比一下：

- 低温像“特别谨慎的答题”
- 高温像“更敢发散联想”

---

## 七、注意力（Attention）：为什么它这么关键？

注意力的核心直觉是：

> 当前 token 在计算表示时，不必平均看所有词，而是可以“更关注和自己相关的词”。

比如句子：

> “小王把球给了小李，因为他接得很稳。”

这里“他”到底指谁，需要看上下文关系。  
注意力机制就是在做这种“相关性分配”。

### 一个极简注意力示例

```python
import numpy as np

# 假设有 3 个 token 的向量表示
X = np.array([
    [1.0, 0.0],   # token1
    [0.0, 1.0],   # token2
    [1.0, 1.0]    # token3
])

# 这里为了演示，直接把 Q K V 都设成 X
Q = X
K = X
V = X

scores = Q @ K.T
scaled_scores = scores / np.sqrt(K.shape[1])

def softmax(row):
    e = np.exp(row - row.max())
    return e / e.sum()

attention_weights = np.apply_along_axis(softmax, 1, scaled_scores)
output = attention_weights @ V

print("注意力分数:\n", np.round(scaled_scores, 3))
print("注意力权重:\n", np.round(attention_weights, 3))
print("输出表示:\n", np.round(output, 3))
```

你不需要现在就完全吃透公式，但要先抓住直觉：

- 先比较“谁和谁相关”
- 再按相关性做加权汇总

---

## 八、预训练、微调、提示词，分别在干什么？

### 1. 预训练

让模型在海量文本上学语言规律。

### 2. 微调

在特定任务或风格上进一步训练，让模型更适应某场景。

### 3. 提示词（Prompting）

不改模型参数，只通过输入方式引导模型按某种方式工作。

类比一下：

| 方式 | 类比 |
|---|---|
| 预训练 | 通读大量书籍 |
| 微调 | 岗前专项培训 |
| Prompting | 临时给清晰任务说明 |

---

## 九、为什么大模型会“看起来像会思考”？

因为当模型规模、数据量和训练质量足够高时，它会学到很多复杂模式：

- 语言规律
- 常识关联
- 指令跟随
- 多步生成结构

但要注意：

> 模型“像会思考”，不等于它和人类思维方式完全一样。

作为工程师，我们更关心的是：

- 它输入输出规律是什么
- 它什么时候可靠
- 它什么时候容易出错

---

## 十、初学者常见误区

### 1. 以为大模型是“直接记住答案”

不完全是。  
它更像是在学习大规模语言分布和模式。

### 2. 以为 temperature 越高越聪明

不是。  
高温更发散，未必更准确。

### 3. 看到 attention 公式就放弃

没必要。  
先抓住“相关性加权”这个直觉，再逐步看公式。

---

## 小结

这节课最核心的几句话是：

1. 大模型通过预测下一个 token 学习语言
2. token 会先变成向量，再进入模型计算
3. 注意力让模型能根据相关性利用上下文
4. 预训练、微调、Prompt 分别贡献不同层面的能力

理解了这些，后面你看 RAG、Agent、工具调用时，就不会只觉得它们是“黑盒魔法”。

---

## 练习

1. 修改温度采样例子里的 `logits`，看看概率分布怎么变化。
2. 试着把注意力示例中的 `X` 改掉，观察注意力权重变化。
3. 用自己的话解释：为什么上下文窗口会直接影响 RAG 效果？
