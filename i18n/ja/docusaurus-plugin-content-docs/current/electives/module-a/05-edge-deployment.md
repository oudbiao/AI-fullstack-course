---
title: "E.A.5 エッジデバイスへのデプロイ"
sidebar_position: 5
description: "メモリ、消費電力、レイテンシ、オフライン要件を見て、モデルがエッジデバイスで安定して動くか判断する。"
keywords: [edge deployment, Jetson, Raspberry Pi, memory budget, latency, offline inference]
---

# E.A.5 エッジデバイスへのデプロイ

![エッジデプロイの制約と判断フロー](/img/course/elective-edge-deployment-constraint-map-ja.png)

エッジデプロイとは、モデルをユーザー、カメラ、機械、センサーの近くで動かすことです。最初の問題は、精度だけではありません。そのデバイスがシステムを長時間安定して動かせるかです。

## 準備するもの

- Python 3.10+
- 外部パッケージ不要
- カメラ分類、工場検査、オフライン帳票認識などの対象シナリオ

## 4つの確認点

- **メモリ**：モデル、ランタイム、入力バッファ、サービス本体がすべて RAM を使う。
- **消費電力**：一度動くことと、長時間熱やスロットリングなしで動くことは違う。
- **レイテンシ**：即時応答が必要なタスクもあれば、少し待てるタスクもある。
- **オフラインモード**：ネットワークが不安定でも、ローカルの代替手段が必要。

## 互換性フィルターを動かす

`edge_fit.py` を作成します。

```python
devices = [
    {"name": "edge-a", "memory_mb": 512, "power_w": 8, "offline": True},
    {"name": "edge-b", "memory_mb": 2048, "power_w": 15, "offline": False},
    {"name": "edge-c", "memory_mb": 4096, "power_w": 25, "offline": True},
]

model = {
    "name": "int8-small-classifier",
    "memory_mb": 700,
    "power_w": 10,
    "latency_ms": 65,
    "requires_offline": True,
}

for device in devices:
    reasons = []

    if device["memory_mb"] < model["memory_mb"]:
        reasons.append("memory")
    if device["power_w"] < model["power_w"]:
        reasons.append("power")
    if model["requires_offline"] and not device["offline"]:
        reasons.append("offline")

    status = "FIT" if not reasons else "CHECK " + ",".join(reasons)
    print(device["name"], status)
```

実行します。

```bash
python edge_fit.py
```

期待される出力：

```text
edge-a CHECK memory,power
edge-b CHECK offline
edge-c FIT
```

結果は左から順に読みます。`edge-c` が必ず最速・最安という意味ではありませんが、今回の制約を満たす唯一のデバイスです。

## もう少し実践に近づける

`model["memory_mb"]` を `700` から `350` に変えて、もう一度実行します。`edge-a` はまだ失敗します。理由は消費電力です。エッジデプロイが複数制約の問題であることが分かります。

## エッジデプロイのチェックリスト

デバイスを「使える」と言う前に、最低限これを確認します。

1. コールドブートから正常に起動する。
2. 30分以上動かして、明らかなメモリ増加がない。
3. ネットワーク断でも最低限動ける。
4. リモート調査に必要なログを残せる。
5. 簡単なロールバックまたは交換手順がある。

## よくある間違い

- 先にモデルを選び、小さなデバイスへ無理に載せる。
- 1回の推論だけを試し、長時間運転を確認しない。
- デバイスが常にオンラインだと思い込む。
- ログ、キャッシュ、入力画像もメモリを使うことを忘れる。

## 練習

各デバイスに `price_usd` を追加し、すべてのチェックを通る最安デバイスを選んでください。さらに2つ目のモデルを追加し、両方を動かせるデバイスを比べます。
