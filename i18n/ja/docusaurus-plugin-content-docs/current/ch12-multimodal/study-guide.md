---
title: "12.0 学習チェックリスト：AIGC とマルチモーダル"
sidebar_position: 1
description: "第 12 章のコンパクトなチェックリスト。multimodal inputs、structured records、generation versions、safety review、export、ポートフォリオ証拠を確認する。"
keywords: [AIGC チェックリスト, マルチモーダル チェックリスト, 画像生成, マルチモーダル RAG, 創作ワークフロー]
---

# 12.0 学習チェックリスト：AIGC とマルチモーダル

このページは印刷用チェックリストとして使います。詳しい説明が必要なときは、[第 12 章入口ページ](./index.md) に戻ってください。

![マルチモーダルポートフォリオ証拠パック](/img/course/ch12-multimodal-evidence-pack-ja.webp)

## 2時間の初回通読

| 時間 | やること | ここまで言えたら止める |
|---|---|---|
| 20 分 | 入口ページのワークフローループを見る | 「マルチモーダル作業は出典を保った入力から始まる。」 |
| 25 分 | 視覚記録スクリプトを動かす | 「視覚内容を確認可能な構造化記録にできる。」 |
| 25 分 | マルチモーダル基礎と画像生成をざっと読む | 「理解と生成には Prompt、model、output、review が必要。」 |
| 25 分 | 倫理とコンプライアンスをざっと読む | 「外部利用には copyright、portrait、sensitive、factual checks が必要。」 |
| 25 分 | RAG/Agent ブリッジを読む | 「マルチモーダルは RAG、Agent、卒業プロジェクトを拡張する。」 |

## 必ず残す証拠

| 証拠 | 最小版 |
|---|---|
| `multimodal_pipeline.md` | input、parsing、generation/understanding、review、export |
| `visual_records.jsonl` | source、page/region/time reference、visible text、objects、uncertainty |
| `prompts/` | Prompt 版、reference assets、negative requirements、selection notes |
| `outputs/` | candidate outputs、selected output、rejected output、reason |
| `safety_review.md` | copyright、portrait rights、sensitive content、factuality、usage boundary |
| `README.md` | goal、run command、source materials、sample output、limitations |

## 品質ゲート

| ゲート | 合格条件 |
|---|---|
| Source trace | すべての input/output が source、owner/license、version、必要なら page/region/time reference を保持している。 |
| Prompt/version | candidate outputs が Prompt、model/tool、reference assets、selection reason に結びついている。 |
| Review | copyright、portrait/voice、sensitive content、factuality、accessibility、export scope が確認されている。 |
| Export | README、manifest、selected outputs、rejected outputs、limits、next fix を他者が確認できる。 |

## 章を出る前の質問

- screenshot、PDF、image、audio、video の出典参照を保持できますか？
- 非テキスト入力を RAG や Agent が使える構造化記録にできますか？
- Prompt 版とレビュー記録で生成出力を比較できますか？
- 外部公開前に何を確認すべきか説明できますか？
- 結果を最終ポートフォリオまたは卒業デモとしてまとめられますか？

答えがすべて「はい」なら、このコースは基礎、データ、モデル、LLM アプリ、Agent、マルチモーダル製品ワークフローまでの端から端までの道になります。

<details>
<summary>参考解答と解説</summary>

- 「できる」とは、各 non-text input に source、owner、version、review status があり、最終ファイルだけが残っている状態ではない、という意味です。
- よい structured record には、抽出内容、modality metadata、confidence または review notes、source artifact へ戻る安定した link が含まれます。
- 生成出力は、prompt versions、candidate ids、selected/rejected decisions、reviewer notes と結びつけます。そうすると iteration を説明できます。
- 外部公開前には、factual grounding、consent and rights、privacy、sensitive content、safety policy、高リスク素材の human approval を確認します。
- portfolio-ready な package には、brief、manifest、prompts、selected assets、rejected cases、review checklist、final export、workflow を説明する README が含まれます。

</details>
## 残す証拠

このページを終えたら、この evidence card を残します。

```text
brief: user goal, audience, assets, constraints, and export format
artifacts: source files, prompts, generated candidates, selected output, and rejected versions
review: factual check, copyright/portrait/sensitive-content check, and human decision
integration: RAG record, Agent trace, creative package, storyboard, or export preview
Expected_output: reproducible asset package with README, review checklist, and failure notes
```
