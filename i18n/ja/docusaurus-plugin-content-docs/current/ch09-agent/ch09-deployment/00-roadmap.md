---
title: "9.9.1 デプロイロードマップ：実行時、永続化、復旧"
sidebar_position: 0
description: "Agent のデプロイと運用の短い実践ロードマップ：API を公開し、状態を永続化し、trace を記録し、コストを制御し、失敗から復旧する。"
keywords: [Agent deployment guide, Agent operations, cost optimization, runtime, observability]
---

# 9.9.1 デプロイロードマップ：実行時、永続化、復旧

Agent のデプロイは、コードをサーバーに置くことだけではありません。モデル呼び出し、ツールサービス、キュー、状態保存、trace、権限、コスト制限、ロールバック経路が必要です。

## まず実行時ループを見る

![Agent 本番実行時アーキテクチャ図](/img/course/ch09-production-runtime-map-ja.webp)

![Agent デプロイと運用の学習フロー図](/img/course/ch09-deployment-chapter-flow-ja.webp)

![Agent デプロイの観測性と復旧ループ図](/img/course/ch09-deployment-observability-loop-ja.webp)

本番運用の問いは「1 回動いたか」ではありません。「動き続け、安全に失敗し、回復できるか」です。

## デプロイ準備チェックを動かす

このチェックは、足りない本番運用の基礎を見つけます。

```python
service = {
    "api_entry": True,
    "state_store": True,
    "trace_log": True,
    "cost_limit": True,
    "rollback": False,
}

missing = [name for name, ok in service.items() if not ok]

print("ready:", not missing)
print("missing:", missing)
```

期待される出力：

```text
ready: False
missing: ['rollback']
```

ロールバックや復旧ができないシステムを、本番準備済みと呼ばないでください。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | デプロイアーキテクチャ | frontend、backend、model service、tool service、storage を描く |
| 2 | 実行時管理 | 同期、非同期、長時間タスク、キュー、中断を扱う |
| 3 | 永続化と復旧 | タスク状態、memory、トレース、中間結果を保存する |
| 4 | コスト最適化 | モデル呼び出し、ツール呼び出し、caching、batching、routing を追跡する |
| 5 | 本番運用 | monitoring、alerts、canary release、rollback、permissions を追加する |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
runtime: queues, workers, state store, tool services, and model endpoint
persistence: checkpoints, event log, memory store, and recovery path
ops_signal: latency, cost, error rate, trace coverage, and saturation
failure_check: stuck run, duplicate action, partial failure, or runaway cost
recovery_action: resume, rollback, cancel, human handoff, or degrade gracefully
```

## 合格ライン

ローカル Agent デモを、API 入口、状態永続化、trace ログ、エラー応答、コスト記録、デプロイ手順を持つ小さなサービスにできれば、この章は合格です。
