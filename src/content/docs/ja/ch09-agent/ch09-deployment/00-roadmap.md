---
title: "9.9.1 デプロイロードマップ：実行時、永続化、復旧"
description: "Agent のデプロイと運用の短い実践ロードマップ：API を公開し、状態を永続化し、trace を記録し、コストを制御し、失敗から復旧する。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Agent deployment guide, Agent operations, cost optimization, runtime, observability"
---
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
ランタイム: キュー、ワーカー、状態ストア、ツールサービス、モデルエンドポイント
永続化：チェックポイント、イベントログ、メモリストア、復旧パス
運用シグナル：レイテンシ、コスト、エラー率、追跡カバレッジ、飽和度
失敗確認: 停止した実行、重複アクション、部分失敗、またはコスト暴走
復旧アクション：再開、ロールバック、中止、人間への引き継ぎ、または安全に劣化
```

## 合格ライン

ローカル Agent デモを、API 入口、状態永続化、trace ログ、エラー応答、コスト記録、デプロイ手順を持つ小さなサービスにできれば、この章は合格です。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、agent loop を goal、plan、tool call、observation、memory/state update、stop condition として説明します。
2. 証拠には、最終回答だけでなく、別の開発者が確認できる trace を残します。
3. tool schema、permission boundary、retry、evaluation case、人間レビューなど、安全性または信頼性の制御を1つ説明できれば十分です。

</details>
