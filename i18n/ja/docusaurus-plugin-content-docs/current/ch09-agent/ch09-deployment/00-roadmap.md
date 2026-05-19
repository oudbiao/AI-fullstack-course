---
title: "9.9.1 Deployment ロードマップ：Runtime、Persistence、Recovery"
sidebar_position: 0
description: "Agent deployment and operations の短い実践ロードマップ：API を公開し、state を永続化し、traces を記録し、cost を制御し、failure から回復する。"
keywords: [Agent deployment guide, Agent operations, cost optimization, runtime, observability]
---

# 9.9.1 Deployment ロードマップ：Runtime、Persistence、Recovery

Agent の deploy は、code を server に置くことだけではありません。model calls、tool services、queues、state storage、traces、permissions、cost limits、rollback paths が必要です。

## まず runtime loop を見る

![Agent production runtime architecture diagram](/img/course/ch09-production-runtime-map-ja.webp)

![Agent deployment and operations 章の学習フロー図](/img/course/ch09-deployment-chapter-flow-ja.webp)

![Agent deployment observability and recovery loop](/img/course/ch09-deployment-observability-loop-ja.webp)

production の問いは「1 回動いたか」ではありません。「動き続け、安全に失敗し、回復できるか」です。

## Deployment readiness check を動かす

このチェックは、足りない production basics を見つけます。

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

rollback や recovery ができない system を production-ready と呼ばないでください。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Deployment architecture | frontend、backend、model service、tool service、storage を描く |
| 2 | Runtime management | sync、async、long-running tasks、queues、interruption を扱う |
| 3 | Persistence and recovery | task state、memory、traces、intermediate results を保存する |
| 4 | Cost optimization | model calls、tool calls、caching、batching、routing を追跡する |
| 5 | Production practices | monitoring、alerts、canary release、rollback、permissions を追加する |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
runtime: queues, workers, state store, tool services, and model endpoint
persistence: checkpoints, event log, memory store, and recovery path
ops_signal: latency, cost, error rate, trace coverage, and saturation
failure_check: stuck run, duplicate action, partial failure, or runaway cost
recovery_action: resume, rollback, cancel, human handoff, or degrade gracefully
```

## 合格ライン

local Agent demo を、API entry、state persistence、trace logs、error responses、cost records、deployment instructions を持つ小さな service にできれば、この章は合格です。
