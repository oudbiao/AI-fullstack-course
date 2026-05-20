---
title: "7.0 学習チェックリスト：大モデル原理、Prompt、微調整"
sidebar_position: 1
description: "第 7 章のコンパクトなチェックリスト。LLM 原理、Prompt 実験、構造化出力、RAG/微調整判断、ポートフォリオ証拠を確認する。"
keywords: [LLM 学習チェックリスト, Prompt 評価, Transformer, 微調整, RLHF]
---

# 7.0 学習チェックリスト：大モデル原理、Prompt、微調整

このページは印刷用チェックリストとして使います。詳しい説明が必要なときは、[第 7 章入口ページ](./index.md) に戻ってください。

![LLM 学習ガイド進化パス](/img/course/ch07-study-guide-evolution-line-ja.webp)

## 2時間の初回通読

| 時間 | やること | ここまで言えたら止める |
|---|---|---|
| 20 分 | 入口ページの「Token から回答まで」の図を見る | 「文章は token、ベクトル、文脈になり、次の token 予測へ進む。」 |
| 25 分 | 7.1 をざっと読み、tokenizer の例を1つ動かす | 「Token 数はコストとコンテキスト制限に影響する。」 |
| 25 分 | 7.2 と LLM 発展史をざっと読む | 「規模、データ、Transformer、整合がモデル能力を変えた。」 |
| 30 分 | 入口ページの Prompt テストスクリプトを動かす | 「固定ケースで Prompt 版を比較できる。」 |
| 20 分 | 手法選択表を読む | 「Prompt、RAG、ツール、検証を確認する前に微調整へ急がない。」 |

## 必ず残す証拠

| 証拠 | 最小版 |
|---|---|
| `prompts/` | 1つのタスクに対する3つの Prompt 版 |
| `prompt_eval_cases.csv` | 少なくとも5つの固定入力と簡単なスコア列 |
| `structured_output_schema.json` | 必須フィールドと許可する値の型 |
| `failure_cases.md` | 少なくとも3つの失敗出力と推定原因 |
| `llm_stage_workshop_output.txt` | [7.8.4 実践：第 7 章フルワークショップ](./ch08-projects/03-stage-hands-on-workshop.md) の出力 |
| `README.md` | 実行方法、通った点、失敗した点、次に試すこと |

## 品質ゲート

| ゲート | 合格条件 |
|---|---|
| Prompt 比較 | 同じケース、変更する変数は1つ、出力とスコアを保存する。 |
| 構造化出力 | parser が欠落フィールドや型違いを拒否する。 |
| 失敗分析 | 各失敗に、指示、入力、スキーマ、知識不足、安全性のいずれかの推定原因がある。 |
| 手法選択 | 決定表が、Prompt、RAG、微調整、ツール、Agent のどれを先に使うか説明している。 |

期待される結果：第 7 章のフォルダに、Prompt 版、固定評価ケース、parser/schema チェック、失敗メモ、ワークショップ出力、手法選択を説明する README がそろっている状態です。

## 章を出る前の質問

- token、embedding、attention、コンテキストウィンドウ、事前学習、Prompt、微調整、整合を、定義の丸写しではなく自分の言葉で説明できますか？
- 毎回1つの Prompt 変数だけを変え、同じ入力ケースで結果を比べられますか？
- JSON らしく見える文章を信じるのではなく、JSON 出力を検証できますか？
- 情報不足のとき、長い Prompt ではなく RAG が必要になる場面を説明できますか？
- 繰り返す振る舞いの適応が、微調整を検討する理由になる場面を説明できますか？

答えがすべて「はい」なら、第 8 章へ進みます。第 8 章では、この考え方を実際の LLM アプリと RAG システムにつなげます。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
prompt_versions: at least three versions for one task
eval_cases: fixed inputs with scores and failure notes
schema_check: structured output is parsed and validated
method_choice: Prompt/RAG/fine-tuning/tools decision is written down
exit_proof: workshop output plus README notes
```
