---
title: "7.8.1 プロジェクトロードマップ：Prompt、RAG、微調整を選ぶ"
description: "第 7 章キャップストーンの実践ロードマップ：ドメインタスクを定義し、Prompt ベースラインを作り、適切な手法を選び、証拠を示す。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "LLM project guide, domain fine-tuning, Prompt, RAG, LLM evaluation"
---
このキャップストーンは、第 7 章を 1 つのエンジニアリング判断にまとめます。問題はタスク表現、知識不足、形式の不安定さ、安全境界、評価不足のどれでしょうか。

## まずプロジェクトルートを見る

![大規模モデル総合プロジェクトのロードマップ](/img/course/ch07-projects-route-map-ja.webp)

![大規模モデルプロジェクトの手法選択ループ図](/img/course/ch07-project-method-choice-loop-ja.webp)

![ポートフォリオ証拠パック図](/img/course/ch07-hands-on-portfolio-evidence-pack-ja.webp)

最強モデルや複雑なフレームワークから始めないでください。小さなドメインタスク、Prompt ベースライン、固定例、失敗ログから始めます。

## 証拠パックチェックを動かす

レポートを書く前に、この小さなプロジェクトログを使います。ベースライン、改善幅、次のルート、今すぐ微調整が必要かを示すためです。

```python
project = {
    "task": "classify course questions",
    "baseline_pass_rate": 0.62,
    "prompt_v2_pass_rate": 0.78,
    "rag_needed": True,
    "finetune_needed": False,
}

improvement = project["prompt_v2_pass_rate"] - project["baseline_pass_rate"]

print("task:", project["task"])
print("improvement:", round(improvement, 2))
print("next_route:", "RAG" if project["rag_needed"] else "Prompt")
print("fine_tune_now:", project["finetune_needed"])
```

期待される出力：

```text
task: classify course questions
improvement: 0.16
next_route: RAG
fine_tune_now: False
```

この項目を埋められないなら、プロジェクトをさらに小さくします。大きいが検証できないデモより、明確な比較のほうが価値があります。

## この順番で学ぶ

| 手順 | 作業 | 証拠 |
|---|---|---|
| 1 | ドメインタスクを 1 つ選ぶ | 1 文のタスク定義と 10 件の固定例 |
| 2 | Prompt ベースラインを作る | Prompt バージョン、出力、合否メモ |
| 3 | 失敗タイプを分類する | タスク表現、知識不足、形式のずれ、安全境界 |
| 4 | 次の手法を選ぶ | Prompt 反復、RAG、微調整の判断メモ |
| 5 | 結果をまとめる | README、実行コマンド、スクリーンショット、失敗例、次の一手 |

先にガイド付きで動かしたい場合は、自分のドメインプロジェクトを設計する前に [7.8.4 実践：第 7 章フルワークショップ](/ja/ch07-llm-principles/ch08-projects/03-stage-hands-on-workshop/) を実行してください。

## 判断ルール：手法を選ぶ前に失敗を名づける

卒業プロジェクトでは、「RAG」や「fine-tuning」が高度に見えるから使う、という形にしません。まず主要な失敗を言語化します。

| 主な失敗 | 最初に試すルート | 必要な証拠 |
|---|---|---|
| モデルが答えの出典を知らない | RAG | 検索文書が回答を支えている |
| 出力形式が崩れる | 構造化出力 + 検証 | parser の通過率が改善する |
| 指示が曖昧 | Prompt の反復改善 | 同じケースが 1 回の prompt 変更で改善する |
| 多くのケースで同じ振る舞いの誤りが繰り返される | fine-tuning / LoRA 候補 | 十分なラベル付き例と hold-out 評価ケース |
| タスクが外部アクションを必要とする | ツール / Agent ルート | ツール呼び出しの追跡記録と回復動作 |

この表はプロジェクトを手法の見本市ではなく、エンジニアリング判断に変えます。

## プロジェクト成果物基準

| 成果物 | 最低基準 | 強いポートフォリオ版 |
|---|---|---|
| README | 目的、実行コマンド、モデルまたは API 選択、入出力例 | 手法のトレードオフ、コストメモ、評価、振り返りを追加 |
| 例 | 10 件以上の固定ケース | Prompt、RAG、微調整、ルールベース版を比較 |
| 評価 | 明確な合否ルール | スコア、失敗タイプ統計、回帰メモを追加 |
| Prompt/データ記録 | Prompt バージョンまたはサンプル形式を保存 | スキーマ 検証、データ品質チェック、安全メモを追加 |
| 発表素材 | 動作を証明するスクリーンショットまたは短い GIF | なぜ現在のルートが代替案より良いか説明 |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
プロジェクト選択：Prompt、RAG、fine-tuning、またはハイブリッド経路
ベースライン: まずは最も簡単に動く方法
評価：固定ケースと採点ルール
成果物：README、prompts、outputs、failures、decision log
橋渡し：第8章ではこれを検索バック付きアプリケーションに変える
```

## 合格ライン

固定した評価セットを使って、「ここで微調整しない理由」「ここで RAG が必要な理由」「この Prompt 変更が効いた理由」を説明できれば、この章は合格です。

最終プロジェクトは基本版で十分です。1 つのドメインタスクで 2 つの Prompt バージョンを比較します。強い版では RAG や小さな微調整実験を追加できますが、必ずベースラインと失敗ログで必要性を示してから行います。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、token、context、attention、prompt、生成挙動が1回の request-response path でどうつながるかを説明します。
2. 証拠には、再現できる prompt または structured-output test を1つ残し、出力が通った理由または失敗した理由を書きます。
3. prompt 設計、RAG、fine-tuning、alignment を切り分け、観察した問題を直す最も軽い方法を選べれば十分です。

</details>
