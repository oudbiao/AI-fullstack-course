---
title: "9.0 学習チェックリスト：AI Agent とエージェントシステム"
sidebar_position: 1
description: "第 9 章のコンパクトなチェックリスト。Agent ループ、tool schema、trace、安全境界、評価、ポートフォリオ証拠を確認する。"
keywords: [Agent チェックリスト, AI Agent 学習, ReAct, MCP, ツール呼び出し, Agent 評価]
---

# 9.0 学習チェックリスト：AI Agent とエージェントシステム

このページは印刷用チェックリストとして使います。詳しい説明が必要なときは、[第 9 章入口ページ](./index.md) に戻ってください。

![Agent トレース 証拠パック](/img/course/ch09-agent-trace-pack-ja.webp)

## 2時間の初回通読

| 時間 | やること | ここまで言えたら止める |
|---|---|---|
| 20 分 | 入口ページの実行ループを見る | 「Agent は goal-state-tool-observation ループである。」 |
| 25 分 | トレース スクリプトを動かす | 「すべての action と observation を再生できる。」 |
| 25 分 | 9.1 と 9.2 をざっと読む | 「Agent、ワークフロー、RAG、ReAct、Plan-and-Execute を分けられる。」 |
| 25 分 | 9.3 のツール安全をざっと読む | 「ツールスキーマ と権限は、巧妙な Prompt より重要。」 |
| 25 分 | 境界選択図を読む | 「Agent を使わない方がよい場面が分かる。」 |

## 必ず残す証拠

| 証拠 | 最小版 |
|---|---|
| `tools_schema.md` | 1～2個のツール。名前、目的、パラメータ、戻り値、エラー、リスクレベルを記載 |
| `agent_traces.jsonl` | 少なくとも3回の実行。目標、ステップ、行動、入力、観察、結果を記録 |
| `safety_boundary.md` | 最大ステップ、ツールホワイトリスト、ブロック action、人間承認ルール |
| `failure_cases.md` | 少なくとも3つの失敗。誤ツール、悪い引数、ループ、権限ブロック、未対応回答 |
| `eval_tasks.csv` | 3～5個の固定タスク。期待結果と成功基準を含む |
| `README.md` | 実行コマンド、トレース 例、安全例、評価結果、制限 |

## 品質ゲート

| ゲート | 合格条件 |
|---|---|
| ツールスキーマ | 各ツールに目的、パラメータ、戻り値、エラー、リスクレベルがある。 |
| トレース再生 | レビュー担当者が、すべてのツール呼び出しの理由を再現できる。 |
| 安全境界 | ホワイトリスト外または危険な action がブロックされるか、人間承認へ回る。 |
| 停止制御 | 最大ステップと停止条件が、ループとコスト急増を防ぐ。 |

期待される結果：第 9 章のプロジェクトフォルダに、ツールスキーマ、再生可能なトレース、安全境界、固定評価タスク、失敗メモ、ループが信頼できるまで単一 Agent に留める理由を書いた README がそろっている状態です。

## 章を出る前の質問

- Agent と普通の LLM アプリの違いを説明できますか？
- トレース を示し、各ツール呼び出しがなぜ起きたか説明できますか？
- 高リスクまたはホワイトリスト外のツールをブロックできますか？
- 停止条件と最大ステップ数を定義できますか？
- 多 Agent は単一 Agent の信頼性のあとに進むべき理由を説明できますか？

答えがすべて「はい」なら、次の方向へ進みます。配置、マルチモーダル Agent、またはコース最終プロジェクトです。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
single_agent_trace: one complete goal-plan-action-observation loop
tool_contract: schema, permission, error behavior, and observation
memory_note: what is written, retrieved, forgotten, or updated
eval_note: success score, safety check, and failure reason
project_readme: run command, trace, limitations, and next action
```
