---
title: "11.6.1 预训练模型路线图：BERT、GPT、T5"
sidebar_position: 0
description: "NLP 预训练模型的简短实操路线：理解预训练、BERT、GPT、T5、transformers pipeline 和任务迁移。"
keywords: [预训练指南, BERT, GPT, T5, transformers]
---

# 11.6.1 预训练模型路线图：BERT、GPT、T5

预训练模型让 NLP 从单任务训练进入可复用基础模型：先在大规模文本上预训练，再迁移到下游任务。

## 先看范式地图

![BERT GPT T5 对比图](/img/course/bert-gpt-t5-comparison.webp)

![预训练语言模型章节学习顺序图](/img/course/ch11-pretrained-chapter-flow.webp)

![预训练迁移微调图](/img/course/ch11-pretraining-transfer-finetune-map.webp)

BERT 偏理解，GPT 偏生成，T5 把很多任务改写成 text-to-text。

## 跑一个模型家族选择检查

```python
task = {
    "needs_generation": True,
    "needs_sentence_label": False,
    "needs_text_to_text": True,
}

if task["needs_text_to_text"]:
    family = "T5-style text-to-text"
elif task["needs_generation"]:
    family = "GPT-style autoregressive"
else:
    family = "BERT-style understanding"

print("family:", family)
print("reason:", "match model objective to task output")
```

预期输出：

```text
family: T5-style text-to-text
reason: match model objective to task output
```

不要只按模型名称选择。要匹配 tokenizer、训练目标、输出格式、成本和部署约束。

## 按这个顺序学

| 步骤 | 阅读 | 实操产出 |
|---|---|---|
| 1 | 预训练范式 | 解释 pretrain → transfer → fine-tune/infer |
| 2 | BERT | 理解 mask prediction 和双向表示 |
| 3 | GPT | 理解 next-token generation 和上下文窗口 |
| 4 | T5 | 把任务改写成 text-to-text 形式 |
| 5 | Transformers 实战 | 连接 tokenizer、model、pipeline、input、output |

## 通过标准

如果你能解释不同训练目标为什么带来不同优势，并运行或设计一个小型预训练模型对比实验，就通过了本章。

<details>
<summary>检查思路与讲解</summary>

1. 合格答案要从文本单元和输出类型说起：token、span、句子标签、序列、embedding 或生成文本。
2. 证据应包含小样本、模型或 pipeline 选择、评价指标，以及至少一个被检查过的错误案例。
3. 自检时要能区分预处理问题和模型问题，例如分词错误、标签歧义、数据不平衡或生成幻觉。

</details>


## 留下的证据

学完这一页，至少保留这张证据卡：

```text
模型选择：BERT、GPT、T5、Transformer 流水线或其他预训练基线
tokenizer 输出：id、mask、解码文本或批次形状
任务结果：分类、生成、抽取或文本到文本输出
失败检查：错误的模型家族、token 限制、领域不匹配、成本或延迟
期望产出：模型调用结果加一段简短的选择理由
```
