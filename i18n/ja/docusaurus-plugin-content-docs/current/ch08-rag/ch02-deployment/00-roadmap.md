---
title: "8.2.1 デプロイロードマップ：ローカルモデル、サービス、統一 API"
sidebar_position: 0
description: "モデルデプロイの短い実践ロードマップ：モデルをどこで動かすか選び、サービスとして公開し、アプリケーションは安定した API 契約を呼ぶ。"
keywords: [モデルデプロイガイド, ローカルモデル, 推論サービス, 統一 API]
---

# 8.2.1 デプロイロードマップ：ローカルモデル、サービス、統一 API

デプロイは、モデルを Notebook 実験から再利用できる能力に変えます。モデル、プロバイダー、ハードウェア、コスト方針が変わっても、アプリケーションは安定した interface を呼べるべきです。

## まず serving 判断を見る

![モデルデプロイ章の学習フローチャート](/img/course/ch08-deployment-chapter-flow-ja.webp)

![モデル serving 選択の意思決定マップ](/img/course/ch08-model-serving-decision-map-ja.webp)

![統一 API プロバイダーゲートウェイ図](/img/course/ch08-unified-api-provider-gateway-map-ja.webp)

デプロイ判断では、品質、遅延、コスト、プライバシー、運用複雑度をバランスさせます。最強モデルが常に呼ぶべきモデルとは限りません。

## モデルルートチェックを動かす

実際の serving ツールを設定する前に、この判断形式を使います。デプロイを明示的な routing 判断にします。

```python
request = {
    "privacy": "high",
    "latency_ms": 800,
    "quality_need": "medium",
    "budget": "low",
}

if request["privacy"] == "high":
    route = "local model or private endpoint"
elif request["quality_need"] == "high":
    route = "frontier cloud model"
else:
    route = "small hosted model"

print("route:", route)
print("contract:", "/v1/chat/completions")
print("watch:", "latency, cost, errors")
```

期待される出力：

```text
route: local model or private endpoint
contract: /v1/chat/completions
watch: latency, cost, errors
```

ルートは変わっても、アプリケーション契約は安定させます。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | ローカルモデル | 1 つのローカル/ private モデルを読み込み、制約を記録する |
| 2 | 推論サービス | モデル呼び出しを service endpoint として公開する |
| 3 | 統一 API | 複数 provider に対して 1 つのアプリ interface を保つ |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
runtime_choice: local model, inference server, or unified API
request_contract: endpoint, payload, output format, and error shape
latency_or_cost: one measured or estimated number
failure_check: timeout, memory pressure, model mismatch, or version drift
rollback_plan: fallback model, retry policy, or traffic switch
```

## 合格ライン

モデルがどこで動くか、アプリがどう呼ぶか、どこで失敗するか、遅延・コスト・エラー・rate limit・fallback をどう見るか説明できれば、この章は合格です。

出口ミニプロジェクトは、小さなモデル gateway メモまたはスクリプトです。1 つのリクエストを選んだモデル endpoint に routing し、判断理由を記録します。
