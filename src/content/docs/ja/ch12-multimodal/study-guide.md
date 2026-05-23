---
title: "12.0 学習チェックリスト：AIGC とマルチモーダル"
description: "第 12 章のコンパクトなチェックリスト。multimodal inputs、structured records、generation versions、safety review、export、ポートフォリオ証拠を確認する。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIGC チェックリスト, マルチモーダル チェックリスト, 画像生成, マルチモーダル RAG, 創作ワークフロー"
---
このページは印刷用チェックリストとして使います。詳しい説明が必要なときは、[第 12 章入口ページ](/ja/ch12-multimodal/) に戻ってください。

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

答えがすべて「はい」なら、マルチモーダル配信の道はできています。プロジェクトにオープンソースモデルのホスティング、ランタイム所有、ファインチューニング判断が必要になったら、第13章へ進みます。

<details>
<summary>確認の考え方と解説</summary>

- 「できる」とは、各 non-text input に source、owner、version、review status があり、最終ファイルだけが残っている状態ではない、という意味です。
- よい structured record には、抽出内容、modality metadata、confidence または review notes、source artifact へ戻る安定した link が含まれます。
- 生成出力は、prompt versions、candidate ids、selected/rejected decisions、reviewer notes と結びつけます。そうすると iteration を説明できます。
- 外部公開前には、factual grounding、consent and rights、privacy、sensitive content、safety policy、高リスク素材の human approval を確認します。
- portfolio-ready な package には、brief、manifest、prompts、selected assets、rejected cases、review checklist、final export、workflow を説明する README が含まれます。

</details>
## 残す証拠

このページを終えたら、この evidence card を残します。

```text
要約：ユーザーの目的、対象読者、素材、制約、出力形式
成果物: ソースファイル、プロンプト、生成候補、選択出力、却下版
レビュー: 事実確認、著作権・肖像権・機微情報チェック、人の判断
統合：RAG レコード、Agent トレース、クリエイティブパッケージ、ストーリーボード、またはエクスポートプレビュー
期待される成果: README、レビュー用チェックリスト、失敗メモを含む再現可能なアセットパッケージ
```
