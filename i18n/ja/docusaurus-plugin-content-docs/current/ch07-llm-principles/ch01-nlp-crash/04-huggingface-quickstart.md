---
title: "7.1.5 Hugging Face クイックスタート"
sidebar_position: 4
description: "Hugging Face の基本フローを動かします。tokenizer 出力、config、batch tensor、model.forward、hidden states、logits、よくあるデバッグを確認します。"
keywords: [Hugging Face, transformers, tokenizer, model, config, forward, batch, logits]
---

# 7.1.5 Hugging Face クイックスタート

![Hugging Face ワークフローのオブジェクト図](/img/course/ch07-huggingface-workflow-object-map-ja.webp)

:::tip 中心の流れ
多くの Hugging Face 例は、次の 1 本の流れに戻せます。

```text
text -> tokenizer -> input_ids / attention_mask -> model.forward -> hidden states / logits / generated tokens
```

この流れが分かると、`pipeline`、`Trainer`、`DataCollator`、`AutoModel...` は不思議な API ではなく、便利なラッパーとして読めます。
:::

## 4 つのオブジェクト

| オブジェクト | 役割 | よく見るフィールド |
|---|---|---|
| tokenizer | テキスト前処理と token から ID への変換 | `input_ids`, `attention_mask` |
| config | モデル構造の設計図 | `hidden_size`, `num_hidden_layers`, `vocab_size` |
| model | ニューラルネットワーク計算 | `last_hidden_state`, `logits`, generated IDs |
| batch | 同じ shape に積まれた tensor 群 | `[batch, seq_len]` 入力 |

大事な習慣は shape を確認することです。shape が違う場合、モデルはまだ「AI らしい処理」に入る前で止まっています。

## 実験 1：重みをダウンロードせずに流れを動かす

依存関係を入れます。

```bash
python -m pip install torch transformers
```

この例では `BertConfig` から小さなランダム BERT を作ります。言語能力はありませんが、pretrained weights をダウンロードせずに基本の呼び出し経路を確認できます。

```python
import torch
from transformers import BertConfig, BertModel

vocab = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "reset": 4,
    "password": 5,
    "refund": 6,
    "order": 7,
    "please": 8,
    "help": 9,
}


def encode(text, max_length=6):
    tokens = ["[CLS]"] + text.lower().split() + ["[SEP]"]
    input_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens][:max_length]
    attention_mask = [1] * len(input_ids)

    while len(input_ids) < max_length:
        input_ids.append(vocab["[PAD]"])
        attention_mask.append(0)

    return input_ids, attention_mask


texts = ["please help reset password", "refund order"]
encoded = [encode(text) for text in texts]

input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.long)

config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
)

model = BertModel(config)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("input_ids shape        :", tuple(input_ids.shape))
print("attention_mask shape   :", tuple(attention_mask.shape))
print("last_hidden_state shape:", tuple(outputs.last_hidden_state.shape))
print("pooler_output shape    :", tuple(outputs.pooler_output.shape))
```

期待される出力：

```text
input_ids shape        : (2, 6)
attention_mask shape   : (2, 6)
last_hidden_state shape: (2, 6, 32)
pooler_output shape    : (2, 32)
```

shape はこう読みます。

- `2` は batch に 2 文あるという意味。
- `6` は各文が長さ 6 に padding または truncation されたという意味。
- `32` は `hidden_size=32` から来る。
- `last_hidden_state` は token ごとに 1 vector を持つ。
- `pooler_output` はこの BERT 風モデルで、文ごとに 1 vector を返す。

## 実験 2：本物の pretrained model を使う

ネットワークがある場合は `from_pretrained` を使います。

```python
from transformers import AutoModel, AutoTokenizer

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

batch = tokenizer(
    ["please help reset password", "refund order"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

outputs = model(**batch)

print(batch.keys())
print(batch["input_ids"].shape)
print(outputs.last_hidden_state.shape)
```

期待される形状レベルの出力：

```text
dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
torch.Size([2, 6])
torch.Size([2, 6, 768])
```

![Hugging Face shape 出力結果図](/img/course/ch07-huggingface-batch-shape-forward-result-map-ja.webp)

ここでは pretrained weights を持つモデルを使います。流れは同じですが、tokenizer、config、weights は Hub から来ており、互いに一致している必要があります。

## 実コードを読むためのオブジェクト地図

![Hugging Face 用語マップ](/img/course/ch07-huggingface-terms-map-ja.webp)

リポジトリを読むときは、知らない名前を中心の流れに戻します。

| 名前 | どう考えるか |
|---|---|
| `pipeline` | tokenizer + model の高レベル demo wrapper |
| `AutoTokenizer` | model repo に合う tokenizer class を読み込む |
| `AutoModel` | task head なしの base model を読み込む |
| `AutoModelForSequenceClassification` | base model + classification head |
| `AutoModelForCausalLM` | next-token generation 用の decoder 系 model |
| `DataCollator` | sample を padding し、batch に積む |
| `Trainer` | training loop、evaluation、checkpoint、logging を包む |
| `logits` | softmax や token 選択前の生スコア |

## デバッグチェックリスト

- Tokenizer と model は同じ model repo から読み込む。
- model を呼ぶ前に `batch.keys()` と tensor shapes を出す。
- padding を使うなら、通常は `attention_mask` が必要。
- ランダムな `BertModel(config)` は interface 学習用で、pretrained model ではない。
- `AutoModel` は表現を返す。task-specific class は task logits を返す。
- CUDA memory error では、まず batch size、sequence length、model size を下げる。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
objects: tokenizer, model, config, pipeline or manual forward pass
offline_run: toy workflow output is saved
real_model_optional: model id and task are recorded if downloaded
shape_or_score: one output tensor shape or prediction score
debug_note: model path, device, and tokenizer/model mismatch checked
```

## 練習

1. 実験 1 の `max_length` を `6` から `4` に変える。どの token が切られるか。
2. `hidden_size=64` に変える。どの output shape が変わるか。
3. 3 つ目の文を追加し、batch 次元が `2` から `3` になることを確認する。
4. 実験 2 で `AutoModel` を `AutoModelForSequenceClassification` に替える。どの新しい field が出るか。
5. `pipeline()` が demo に便利でも、batch shape のデバッグに不十分な理由を説明する。

## まとめ

Hugging Face は tensor を追うと学びやすいです。

```text
tokenizer creates tensors -> model consumes tensors -> outputs expose states or logits
```

この経路を確認できれば、公式サンプルはかなり読みやすくなります。
