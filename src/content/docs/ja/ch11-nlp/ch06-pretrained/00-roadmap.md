---
title: "11.6.1 事前学習モデルロードマップ：BERT、GPT、T5"
description: "事前学習モデル章を短く実践的に進めるための地図です。BERT、GPT、T5 の目的の違いと、タスクに合わせた選び方を先に見ます。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "事前学習ガイド, BERT, GPT, T5, transformers"
---
事前学習モデルでは、モデルを毎回ゼロから育てません。大規模テキストで先に言語のパターンを学び、あとから分類、生成、抽出、検索などのタスクに使います。

![BERT、GPT、T5 の違い](/img/course/bert-gpt-t5-comparison-ja.webp)

## 先に全体像を見る

![事前学習モデル章の進め方](/img/course/ch11-pretrained-chapter-flow-ja.webp)

| モデル群 | 得意な方向 | 代表的な用途 |
|---|---|---|
| BERT | 理解 | 分類、抽出、照合 |
| GPT | 生成 | チャット、文章生成、ツール呼び出し |
| T5 | text-to-text | 翻訳、要約、QA、分類の統一 |

## タスクからモデルの型を選ぶ

![事前学習から転移・微調整への流れ](/img/course/ch11-pretraining-transfer-finetune-map-ja.webp)

モデル名を丸暗記するより、まず出力形式を見ます。

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

期待される出力：

```text
family: T5-style text-to-text
reason: match model objective to task output
```

操作のコツ：`autoregressive` は、前の token を見ながら次の token を生成する方式です。GPT 系の生成直感を理解するための重要語です。

## transformers を学ぶときの見方

`transformers` は、tokenizer、model、pipeline などを同じ考え方で扱えるライブラリです。初心者はまず次の 3 点だけ意識します。

| 部品 | 役割 |
|---|---|
| tokenizer | 文字列を token ID に変える |
| model | ID を受け取り、予測や生成を行う |
| pipeline | よくある処理を短いコードで実行する |

## 通過条件

| チェック | 合格ライン |
|---|---|
| 事前学習 | 大規模テキストで先に学び、下流タスクへ使う流れを説明できる |
| BERT / GPT / T5 | 理解、生成、text-to-text の違いを言える |
| タスク選定 | 出力形式からモデルの型を選べる |
| 次章とのつながり | RAG、Prompt、Agent で tokenizer、embedding、生成が再登場すると説明できる |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
モデル選択：BERT、GPT、T5、Transformers のパイプライン、または他の事前学習ベースライン
tokenizer 出力：ids、masks、デコード済みテキスト、またはバッチ形状
タスク結果：classification、generation、extraction、または text-to-text 出力
失敗確認: 間違ったモデルファミリー、トークン上限、ドメイン不一致、コスト、またはレイテンシ
期待される成果: モデル呼び出し結果と短い選択理由
```
