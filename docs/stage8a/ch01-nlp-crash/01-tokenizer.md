---
title: "1.1 分词与 Tokenizer"
sidebar_position: 1
description: "从“模型为什么不能直接读文字”讲起，理解词级、字级、子词级分词的取舍，以及 padding、truncation、special tokens 在工程里为什么重要。"
keywords: [tokenizer, tokenization, subword, BPE, wordpiece, padding, truncation]
---

# 分词与 Tokenizer

:::tip 本节定位
很多人第一次学大模型时，会把注意力全放在模型结构上。  
但真正把文本送进模型之前，还有一道绕不过去的门：

> **文字到底要被切成什么单位，模型才能处理？**

这就是 tokenizer。

如果这一步没想明白，后面你看到的：

- `input_ids`
- `attention_mask`
- context length
- token 成本

都会像一堆零碎术语。

这节课的目标，就是把 tokenizer 从“工具黑盒”拉回到它最本质的位置。
:::

## 学习目标

- 理解为什么模型不能直接吃原始字符串
- 区分字级、词级、子词级分词的核心差异
- 理解 special tokens、padding、truncation 在工程中的作用
- 通过可运行示例看懂 tokenizer 是怎样把文本变成 `input_ids` 的

---

## 一、为什么模型不能直接读文字？

### 1.1 模型最终处理的是数字，不是字符本身

神经网络本质上只能处理数值张量。  
而人类输入给模型的，通常是：

- 中文句子
- 英文段落
- 代码
- 混合标点和表情

模型并不认识“退款”“password”“hello”这些词的肉眼形状。  
它需要先经过两步：

1. 把文本切成一个个 token
2. 把 token 映射成整数 id

所以 tokenizer 做的不是“简单切词”，  
而是：

> **把人类语言变成模型可处理离散符号序列的第一层接口。**

### 1.2 一个类比：把文章翻译成机器能编号的积木

你可以把 tokenizer 想成仓库管理员。

原始文本像一大段还没整理的货物。  
tokenizer 要先决定：

- 每块积木的大小是什么
- 每块积木编号是多少

之后模型看到的就不再是“文章”，而是：

- `[101, 2057, 2024, 2172, 102]`

如果积木切得太碎，会变得很长；  
如果切得太粗，又会有很多词不认识。

---

## 二、最常见的三种切法

### 2.1 字级 / 字符级：最稳，但序列会变长

最简单的思路是：

- 一个字或一个字符算一个 token

优点是：

- 几乎不会有 OOV 问题
- 不认识的词也能拆开表示

缺点是：

- 序列很长
- 语义粒度太细
- 模型需要自己花更多层去组合词义

例如中文里：

- “退款规则” -> `退 / 款 / 规 / 则`

### 2.2 词级：语义直观，但 OOV 会严重

另一种思路是：

- 一个完整单词算一个 token

优点是：

- 粒度自然
- 词义直观

缺点是：

- 新词、拼写变体、专有名词很多
- 词表会非常大

例如英文里：

- `refund` 很常见
- 但 `refundability`、`refund-processing` 可能就很容易变成未知词

### 2.3 子词级：现实里最常见的折中

现代大模型里最常见的是：

- subword tokenizer

也就是把词拆成“高频片段”。

例如：

- `transformers` -> `transform` + `ers`
- `tokenization` -> `token` + `ization`

这种方法的好处是：

- 词表不必无限大
- 新词能由已有子词拼出来
- 序列长度和 OOV 问题之间取得平衡

这也是为什么 BPE、WordPiece、SentencePiece 这类方法会这么重要。

---

## 三、先跑一个真正说明问题的 tokenizer 示例

下面这段代码不会复刻完整工业 tokenizer，  
但它会非常清楚地演示三件事：

1. 词级切分
2. 子词级切分
3. token 到 id 的映射、padding 和 truncation

```python
import re

vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "refund": 4,
    "policy": 5,
    "reset": 6,
    "password": 7,
    "transform": 8,
    "##er": 9,
    "##s": 10,
    "token": 11,
    "##ization": 12,
    "please": 13,
    "help": 14,
}


def word_tokenize(text):
    return re.findall(r"[A-Za-z]+", text.lower())


def subword_tokenize(word, vocab):
    if word in vocab:
        return [word]

    tokens = []
    start = 0
    while start < len(word):
        matched = None
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                matched = piece
                tokens.append(piece)
                start = end
                break
        if matched is None:
            return ["[UNK]"]
    return tokens


def encode(text, vocab, max_length=8):
    words = word_tokenize(text)
    tokens = ["[CLS]"]
    for word in words:
        tokens.extend(subword_tokenize(word, vocab))
    tokens.append("[SEP]")

    token_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    token_ids = token_ids[:max_length]
    attention_mask = [1] * len(token_ids)

    if len(token_ids) < max_length:
        pad_count = max_length - len(token_ids)
        token_ids += [vocab["[PAD]"]] * pad_count
        attention_mask += [0] * pad_count

    return tokens, token_ids, attention_mask


examples = [
    "Please help reset password",
    "Transformers policy",
    "Tokenization refund",
]

for text in examples:
    tokens, token_ids, attention_mask = encode(text, vocab, max_length=10)
    print("-" * 60)
    print("text          :", text)
    print("tokens        :", tokens)
    print("input_ids     :", token_ids)
    print("attention_mask:", attention_mask)
```

### 3.1 这段代码最该看哪几行？

重点看三处：

1. `word_tokenize`  
   说明原始字符串如何先被切成词
2. `subword_tokenize`  
   说明词不在词表里时，如何贪心拆成子词
3. `encode`  
   说明 special tokens、padding、truncation 是怎么加进去的

### 3.2 为什么 `Transformers` 会被拆成多个子词？

因为词表里没有完整的 `transformers`，  
但有：

- `transform`
- `##er`
- `##s`

所以它仍然能被表示出来。

这正是子词 tokenizer 的关键优势：

- 新词不一定要整个都在词表里

### 3.3 `attention_mask` 是干什么的？

因为 batch 里的句子长度通常不同。  
为了凑成统一张量，我们会在短句后面补 `[PAD]`。

但模型不能把这些 pad 位当成真实内容，  
所以要用 `attention_mask` 告诉它：

- `1` 是真实 token
- `0` 是 padding

---

## 四、为什么 tokenizer 会直接影响成本和效果？

### 4.1 同一句话切得越碎，token 数就越多

token 越多意味着：

- 上下文更容易用完
- 推理成本更高
- API 计费更贵

所以 tokenizer 不是纯理论问题，  
它也会直接影响工程成本。

### 4.2 词表太小和太大都不好

如果词表太小：

- 很多词会被切得很碎

如果词表太大：

- embedding 表会更大
- 稀有词会更多
- 训练数据利用率未必更好

现实中 tokenizer 设计就是在这些因素之间找平衡。

### 4.3 不同语言会带来不同挑战

例如：

- 英文天然有空格，分词相对容易
- 中文没有空格，切分粒度更敏感
- 代码里还会混入驼峰命名、下划线、符号

所以 tokenizer 往往会针对训练语料的语言特征做适配。

---

## 五、special tokens 为什么总在出现？

### 5.1 `[CLS]`、`[SEP]`、`[PAD]` 不只是装饰

这些特殊 token 一般承担明确功能：

- `[CLS]`：句子级表示的起点
- `[SEP]`：分隔多个片段
- `[PAD]`：对齐 batch 长度

不同模型的具体符号可能不同，  
但思想很接近。

### 5.2 Chat 模型里的 system / user / assistant 其实也是类似思路

到了聊天模型时代，你会看到更多特殊标记，例如：

- `<|system|>`
- `<|user|>`
- `<|assistant|>`

它们本质上也是在用特殊 token 告诉模型：

- 这一段是谁说的
- 对话结构怎么分隔

所以 chat template 其实也是 tokenizer 生态的一部分。

---

## 六、最容易踩的坑

### 6.1 误区一：tokenizer 只是预处理细节

不是。  
它直接影响：

- token 数量
- 词表规模
- OOV 处理
- 下游模板格式

### 6.2 误区二：只要能切开就行

真正重要的是：

- 切得是否稳定
- 是否适配语料
- 是否兼顾长度和语义粒度

### 6.3 误区三：中文就一定按“词”切最好

不一定。  
很多现代模型仍然采用：

- 字级
- 子词级
- SentencePiece 一类统一分词

关键还是看训练目标和数据分布。

---

## 小结

这节最重要的不是背下 BPE 或 WordPiece 这些名字，  
而是抓住一条主线：

> **Tokenizer 的本质，是把原始文本切成模型可处理的离散单位，并在词表大小、未知词问题和序列长度之间做工程权衡。**

只要这条主线建立起来，  
你后面再看：

- input ids
- attention mask
- context length
- prompt 模板

就不会把它们当成零碎概念了。

---

## 练习

1. 把示例里的词表删掉 `transform` 或 `##ization`，看看哪些词会退化成 `[UNK]`。
2. 为什么说子词 tokenizer 是词级和字级之间的折中？
3. 把 `max_length` 改短，观察 truncation 对输出有什么影响。
4. 想一想：如果你的语料里代码特别多，tokenizer 设计最可能先碰到什么问题？
