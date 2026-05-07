---
title: "9.0 学習チェックリスト：AI Agent とエージェントシステム"
sidebar_position: 1
description: "第 9 章のコンパクトなチェックリスト。Agent ループ、tool schema、trace、安全境界、評価、ポートフォリオ証拠を確認する。"
keywords: [Agent チェックリスト, AI Agent 学習, ReAct, MCP, ツール呼び出し, Agent 評価]
---

# 9.0 学習チェックリスト：AI Agent とエージェントシステム

このページは印刷用チェックリストとして使います。詳しい説明が必要なときは、[第 9 章入口ページ](./index.md) に戻ってください。

![Agent trace 証拠パック](/img/course/ch09-agent-trace-pack-ja.svg)

## 9.0.1 2時間の初回通読

| 時間 | やること | ここまで言えたら止める |
|---|---|---|
| 20 分 | 入口ページの実行ループを見る | 「Agent は goal-state-tool-observation ループである。」 |
| 25 分 | trace スクリプトを動かす | 「すべての action と observation を再生できる。」 |
| 25 分 | 9.1 と 9.2 をざっと読む | 「Agent、workflow、RAG、ReAct、Plan-and-Execute を分けられる。」 |
| 25 分 | 9.3 のツール安全をざっと読む | 「tool schema と権限は、巧妙な Prompt より重要。」 |
| 25 分 | 境界選択図を読む | 「Agent を使わない方がよい場面が分かる。」 |

## 9.0.2 必ず残す証拠

| 証拠 | 最小版 |
|---|---|
| `tools_schema.md` | 1～2個のツール。name、purpose、parameters、return value、errors、risk level を記載 |
| `agent_traces.jsonl` | 少なくとも3回の実行。goal、step、action、input、observation、result を記録 |
| `safety_boundary.md` | 最大ステップ、ツールホワイトリスト、ブロック action、人間承認ルール |
| `failure_cases.md` | 少なくとも3つの失敗。誤ツール、悪い引数、ループ、権限ブロック、未対応回答 |
| `eval_tasks.csv` | 3～5個の固定タスク。期待結果と成功基準を含む |
| `README.md` | 実行コマンド、trace 例、安全例、評価結果、制限 |

## 9.0.3 章を出る前の質問

- Agent と普通の LLM アプリの違いを説明できますか？
- trace を示し、各ツール呼び出しがなぜ起きたか説明できますか？
- 高リスクまたはホワイトリスト外のツールをブロックできますか？
- 停止条件と最大ステップ数を定義できますか？
- 多 Agent は単一 Agent の信頼性のあとに進むべき理由を説明できますか？

答えがすべて「はい」なら、次の方向へ進みます。配置、マルチモーダル Agent、またはコース最終プロジェクトです。
