---
title: "7.1.1 NLP 速成路线图：文本到 token 到向量"
description: "紧凑版 NLP 速成路线图：分词、嵌入、预训练模型、Hugging Face 和 token 小实验。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "NLP 速成, tokenizer, embedding, pretrained model, Hugging Face"
---
想理解 LLM，先看文本如何变成模型能处理的形式：文本 -> token -> ID -> 向量 -> 模型输出。

## 先看流程

![NLP 速成章节流程图](/img/course/ch07-nlp-crash-chapter-flow.webp)

| 词 | 第一层意思 |
|---|---|
| token | 模型使用的一小段文本 |
| tokenizer | 切分文本并映射成 ID 的工具 |
| embedding | token 或文本的稠密向量 |
| pretrained model | 已经在大规模文本上训练过的模型 |
| Hugging Face | 模型、数据集、工具生态 |

## 跑一个极小 token 实验

```python
text = "RAG retrieves evidence before answering"
tokens = text.lower().split()
vocab = {token: index for index, token in enumerate(sorted(set(tokens)))}
ids = [vocab[token] for token in tokens]

print("tokens:", tokens)
print("ids:", ids)
print("unique_tokens:", len(vocab))
```

预期输出：

```text
tokens: ['rag', 'retrieves', 'evidence', 'before', 'answering']
ids: [3, 4, 2, 1, 0]
unique_tokens: 5
```

真实 tokenizer 更复杂，但主线一样：文本必须先变成稳定的片段和 ID，后面才能进入向量和模型。

## 按这个顺序学

| 顺序 | 阅读 | 练什么 |
|---|---|---|
| 1 | [7.1.2 Tokenizer](./01-tokenizer.md) | 文本 -> token -> ID |
| 2 | [7.1.3 Embeddings](./02-embeddings.md) | token/文本 -> 向量 |
| 3 | [7.1.4 预训练模型](./03-pretrained-models.md) | 加载并复用模型能力 |
| 4 | [7.1.5 Hugging Face 快速上手](./04-huggingface-quickstart.md) | pipeline、model card、本地运行 |
| 5 | [7.1.6 Tokenizer 与 Embedding 实验](./05-tokenizer-embedding-lab.md) | 检查 token 和向量 |

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
文本路径：原始文本 -> tokens -> ids -> embeddings
token 风险：长输入可能触及上下文或成本限制
嵌入用途：相似度可支持检索，但不是推理
模型桥接：预训练模型 = 共享基础加任务行为
下一步动作：先运行 tokenizer 和 embedding 实验室，再做 Prompt 工作
```

## 通过标准

能解释为什么原始文本需要分词、为什么 embedding 是向量、为什么预训练模型通常复用而不是从零训练，就算通过。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要说明 token、上下文、attention、prompt 和生成行为如何组成一次请求到回答的路径。
2. 证据至少包含一个可复现 prompt 或结构化输出测试，并说明输出为什么通过或失败。
3. 自检时要区分 prompt、RAG、微调和对齐：优先使用能解决已观察问题的最轻方案。

</details>
