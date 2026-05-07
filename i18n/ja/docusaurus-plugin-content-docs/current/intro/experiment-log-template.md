---
sidebar_position: 13
title: "実験ログと README テンプレート"
description: "実践をポートフォリオ証拠に変えるための、コピー可能な README、実験ログ、失敗サンプルのテンプレートです。"
keywords: [実験ログテンプレート, READMEテンプレート, AIプロジェクト振り返り, ポートフォリオ]
---

# 実験ログと README テンプレート

![AI プロダクト実験指標ループ](/img/course/elective-ai-product-experiment-metrics-loop-ja.png)

実際のコマンド、出力、指標、失敗があるときに使ってください。テンプレートは短くします。誰も埋めないテンプレートは雑音になります。

## 最小 README テンプレート

````md
# プロジェクト名

## 目標
何の問題を解くか？誰のためか？

## 実行方法
```bash
python main.py
```

## サンプル入力と出力
入力：

出力：

## 評価または確認
結果が許容できると、どう判断するか？

## 失敗サンプル
何が失敗し、なぜ失敗し、どう修正を確認するか？

## 次の一歩
次バージョンで何を変えるか？
````

## 実験ログテンプレート

| 項目 | 書くこと |
| --- | --- |
| `experiment_id` | `rag_exp_003` |
| 目標 | 今回検証したいこと |
| baseline | 何と比較するか |
| 変更 | 今回変えた主要変数 1 つ |
| 設定 | モデル、Prompt、検索、Agent、学習設定 |
| 指標 | Accuracy、Hit@k、citation_ok、遅延、コスト、手動スコア |
| 結果 | 良くなった点と悪くなった点 |
| 判断 | 採用、却下、修正して再試行 |

## 失敗サンプルテンプレート

| 項目 | 書くこと |
| --- | --- |
| 入力 | 失敗した正確な入力 |
| 期待 | 本来起きるべきこと |
| 実際 | 実際に起きたこと |
| 層 | environment / data / model / prompt / RAG / Agent / deployment |
| 原因 | 現時点で最もありそうな説明 |
| 修正 | 何を変えたか |
| 回帰チェック | 再発しないことをどう確認するか |

## 推奨ファイル

```text
reports/
  failure_cases.md
  improvement_record.md
evals/
  eval_questions.csv
  citation_check.csv
logs/
  llm_calls.jsonl
  retrieval_logs.jsonl
  agent_traces.jsonl
```

記録は事務作業ではありません。成功 Demo を見せるだけでなく、評価し、調査し、改善できることを示す証拠です。
