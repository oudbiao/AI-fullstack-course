---
sidebar_position: 16
title: "失敗と評価の索引"
description: "AIプロジェクトごとに残すべき失敗サンプル、評価セット、ログ、振り返りファイルを短く整理します。"
keywords: [失敗ケース, 評価テンプレート, テストケース, AIプロジェクト評価, ポートフォリオ]
---

# 失敗と評価の索引

![AIプロジェクト デバッグ索引図](/img/course/appendix-quick-ref-debug-index-map-ja.png)

成功画面だけでは足りません。よい AI プロジェクトには、再現できる失敗と、繰り返し実行できる評価ケースがあります。

## 1. まず失敗の層を決める

| 症状 | 可能性のある層 | 残す証拠 |
|---|---|---|
| コマンド、import、パスのエラー | 環境または Python | コマンド、完全なエラー、バージョン情報 |
| 図や結論がおかしい | データ | データ例、クリーニング記録、前後比較 |
| モデルスコアが異常に高い | 機械学習評価 | 分割ルール、baseline、リーク確認 |
| Loss が下がらない | 深層学習 | 設定、曲線、テンソル形状メモ |
| JSON 項目がずれる | Prompt | Prompt 版、固定入力、出力差分 |
| RAG が違う出典を引用する | 検索または引用 | chunks、top-k ログ、引用比較 |
| Agent が違うツールを選ぶ | ツール schema または計画 | trace、ツール入出力、停止条件 |
| ローカルでは動き本番で失敗する | デプロイ | 環境変数、ログ、起動コマンド |

## 2. 最小ファイル

```text
reports/
├── failure_cases.md
├── improvement_record.md
└── demo_notes.md

evals/
├── eval_questions.csv
├── prompt_cases.csv
└── agent_tasks.jsonl

logs/
├── llm_calls.jsonl
├── retrieval_logs.jsonl
└── agent_traces.jsonl
```

## 3. 失敗サンプル形式

```md
## 失敗タイトル

- 入力：
- 期待：
- 実際：
- 層：
- 証拠：
- 原因の仮説：
- 修正：
- 回帰テスト：
```

失敗メモは短くて構いません。ただし再現できることが大切です。再生できる失敗は、恥ではなくエンジニアリングの証拠です。
