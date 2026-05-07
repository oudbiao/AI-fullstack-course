---
title: "12.0 学習チェックリスト：AIGC とマルチモーダル"
sidebar_position: 1
description: "第 12 章のコンパクトなチェックリスト。multimodal inputs、structured records、generation versions、safety review、export、ポートフォリオ証拠を確認する。"
keywords: [AIGC チェックリスト, マルチモーダル チェックリスト, 画像生成, マルチモーダル RAG, 創作ワークフロー]
---

# 12.0 学習チェックリスト：AIGC とマルチモーダル

このページは印刷用チェックリストとして使います。詳しい説明が必要なときは、[第 12 章入口ページ](./index.md) に戻ってください。

![マルチモーダルポートフォリオ証拠パック](/img/course/ch12-multimodal-evidence-pack-ja.svg)

## 12.0.1 2時間の初回通読

| 時間 | やること | ここまで言えたら止める |
|---|---|---|
| 20 分 | 入口ページのワークフローループを見る | 「マルチモーダル作業は出典を保った入力から始まる。」 |
| 25 分 | 視覚記録スクリプトを動かす | 「視覚内容を確認可能な構造化記録にできる。」 |
| 25 分 | マルチモーダル基礎と画像生成をざっと読む | 「理解と生成には Prompt、model、output、review が必要。」 |
| 25 分 | 倫理とコンプライアンスをざっと読む | 「外部利用には copyright、portrait、sensitive、factual checks が必要。」 |
| 25 分 | RAG/Agent ブリッジを読む | 「マルチモーダルは RAG、Agent、卒業プロジェクトを拡張する。」 |

## 12.0.2 必ず残す証拠

| 証拠 | 最小版 |
|---|---|
| `multimodal_pipeline.md` | input、parsing、generation/understanding、review、export |
| `visual_records.jsonl` | source、page/region/time reference、visible text、objects、uncertainty |
| `prompts/` | Prompt 版、reference assets、negative requirements、selection notes |
| `outputs/` | candidate outputs、selected output、rejected output、reason |
| `safety_review.md` | copyright、portrait rights、sensitive content、factuality、usage boundary |
| `README.md` | goal、run command、source materials、sample output、limitations |

## 12.0.3 章を出る前の質問

- screenshot、PDF、image、audio、video の出典参照を保持できますか？
- 非テキスト入力を RAG や Agent が使える構造化記録にできますか？
- Prompt 版とレビュー記録で生成出力を比較できますか？
- 外部公開前に何を確認すべきか説明できますか？
- 結果を最終ポートフォリオまたは卒業 Demo としてまとめられますか？

答えがすべて「はい」なら、このコースは基礎、データ、モデル、LLM アプリ、Agent、マルチモーダル製品ワークフローまでの端から端までの道になります。
