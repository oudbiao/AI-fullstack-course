---
title: "9.10.1 プロジェクトロードマップ：追跡可能な Agent を作る"
sidebar_position: 0
description: "第 9 章プロジェクトの短い実践ロードマップ：目標、計画、ツール、メモリ、trace、安全、評価、デプロイ証拠を備えた Agent ポートフォリオを作る。"
keywords: [Agent プロジェクトガイド, リサーチアシスタント, データ分析 Agent, Multi-Agent プロジェクト, Agent ポートフォリオ]
---

# 9.10.1 プロジェクトロードマップ：追跡可能な Agent を作る

Agent のポートフォリオでは、最終回答だけでなく、追跡可能な実行ループを見せるべきです。

## まずプロジェクトの流れを見る

![Agent 総合プロジェクトロードマップ](/img/course/ch09-projects-route-map-ja.webp)

![Agent プロジェクト学習順序図](/img/course/ch09-project-learning-order-map-ja.webp)

![Agent プロジェクト提出ループ図](/img/course/ch09-project-delivery-loop-ja.webp)

このループは、目標、計画、ツール呼び出し、観察、状態更新、失敗処理、停止判断、最終出力、評価で構成されます。

## Agent の証拠チェックを動かす

ポートフォリオに載せられる状態と呼ぶ前に、このチェックを使います。

```python
project = {
    "goal_defined": True,
    "trace_saved": True,
    "tool_logs": True,
    "failure_case": True,
    "eval_tasks": 10,
}

ready = (
    project["goal_defined"]
    and project["trace_saved"]
    and project["tool_logs"]
    and project["failure_case"]
    and project["eval_tasks"] >= 5
)

print("portfolio_ready:", ready)
print("evidence:", "goal, trace, tools, failure, eval")
```

期待される出力：

```text
portfolio_ready: True
evidence: goal, trace, tools, failure, eval
```

ここが `False` なら、Agent の役割を増やす前に証拠を改善します。

## この順番で学ぶ

| 手順 | プロジェクト | 本当に鍛える力 |
|---|---|---|
| 1 | リサーチアシスタント | 検索、引用、要約、信頼できる出力 |
| 2 | データ分析 Agent | Python ツール呼び出し、表分析、チャート、解釈 |
| 3 | Multi-Agent 開発チーム | 役割分担、handoff、レビューループ、merge 所有権 |
| 4 | ハンズオンワークショップ | 最小の追跡可能な単一 Agent ベースライン |

プロジェクトを広げる前に、[9.10.5 実践：追跡可能な単一 Agent アシスタントを作る](./04-stage-hands-on-workshop.md) を実行します。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
project_goal: what the agent should accomplish and what it must not do
baseline: single-agent loop before adding advanced features
trace_pack: goal, plan, tool calls, observations, memory, evaluation
failure_log: one failed or unsafe run with root cause
成果物：README、実行コマンド、trace スクリーンショット/ログ、次の一手
```

## プロジェクト成果物基準

| 成果物 | 最低要件 | 強いポートフォリオ版 |
|---|---|---|
| README | 目標、実行コマンド、依存関係、例 | アーキテクチャ、トレードオフ、コスト、安全性、ふりかえりを追加 |
| アーキテクチャ | モデル、ツール、記憶、状態、評価、安全性 | 配置境界と人への引き継ぎを追加 |
| ツール一覧 | 呼び出せるツール、入出力スキーマ、失敗 | 権限ルールとサンドボックスメモを追加 |
| 実行追跡 | 計画、行動、観察、再計画、停止 | 再生可能な JSONL ログを追加 |
| 失敗ケース | 1 件以上の実際の失敗 | 3 件の原因、修正、回帰チェックを追加 |
| 評価セット | 固定タスクと合否ルール | ベースライン、メトリクス、比較実験を追加 |
| デプロイメモ | ローカル実行方法 | API エントリ、環境変数、監視、ロールバックを追加 |

## 合格ライン

別の開発者が Agent run を replay し、各 tool call と observation を inspect し、なぜ stop したか理解し、少なくとも 1 件の failure analysis を見られれば、この章は合格です。

基本版は単一 Agent プロジェクトで十分です。memory、MCP、Multi-Agent 協調、デプロイは、trace と評価ループが固まってから追加します。

<details>
<summary>参考解答と解説</summary>

1. 合格レベルの答えでは、agent loop を goal、plan、tool call、observation、memory/state update、stop condition として説明します。
2. 証拠には、最終回答だけでなく、別の開発者が確認できる trace を残します。
3. tool schema、permission boundary、retry、evaluation case、人間レビューなど、安全性または信頼性の制御を1つ説明できれば十分です。

</details>
