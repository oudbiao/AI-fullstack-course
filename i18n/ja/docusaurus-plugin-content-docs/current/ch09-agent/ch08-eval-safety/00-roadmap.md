---
title: "9.8.1 評価と安全ロードマップ：採点、防御、トレース"
sidebar_position: 0
description: "Agent 評価と安全の短い実践ロードマップ：結果と過程を評価し、ガードレールを追加し、trace を記録し、リスクをレビューする。"
keywords: [Agent Evaluation Guide, Agent Safety Guide, Guardrails, Observability, Agent Risk]
---

# 9.8.1 評価と安全ロードマップ：採点、防御、トレース

Agent は動くだけでは不十分です。成功したか、過程は安全だったか、失敗がどこで起きたかを知る必要があります。

## まずガードレールの層を見る

![Agent ガードレール層の図](/img/course/agent-guardrails-layers-ja.webp)

![Agent evaluation and safety 章の学習フロー](/img/course/ch09-eval-safety-chapter-flow-ja.webp)

![Agent リスクデバッグ閉ループ図](/img/course/ch09-agent-risk-debug-loop-ja.webp)

評価はシステムが有効かを示します。安全はシステムが何をしてよいかを決めます。観測性はどこで壊れたかを示します。

## リリース用スコアカードチェックを動かす

最終出力と実行過程の両方を評価します。

```python
run = {
    "task_success": True,
    "tool_error": False,
    "permission_confirmed": True,
    "trace_saved": True,
    "cost_usd": 0.08,
}

launch_ok = (
    run["task_success"]
    and not run["tool_error"]
    and run["permission_confirmed"]
    and run["trace_saved"]
    and run["cost_usd"] < 0.10
)

print("launch_ok:", launch_ok)
print("scorecard:", "task, tools, safety, trace, cost")
```

期待される出力：

```text
launch_ok: True
scorecard: task, tools, safety, trace, cost
```

滑らかな最終回答だけでは十分な証拠ではありません。再生可能なタスクと過程 trace を残します。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | 評価方法 | 結果評価と過程評価を分ける |
| 2 | ベンチマーク | 公開ベンチマークは参考として使い、製品評価の代替にしない |
| 3 | 安全とアラインメント | prompt injection、過剰権限、漏えい、hallucination を識別する |
| 4 | ガードレール | 入力フィルター、出力検証、権限、人間による確認を追加する |
| 5 | 観測性 | ログ、トレース、エラー、レイテンシ、コスト、失敗理由を保存する |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
eval_cases: fixed tasks and expected safe behavior
scorecard: task success, tool correctness, trace quality, safety
guardrail: policy, permission, validation, or human confirmation
failure_check: unsafe tool use, prompt injection, hidden state, or unobserved action
next_action: add case, guardrail, log, rollback, or refusal path
```

## 合格ライン

すべての Agent 実行を、目標、計画、ツール呼び出し、観察、最終回答、安全ルール、コスト、失敗理由からレビューできれば、この章は合格です。

出口ミニプロジェクトは、10〜20 件のタスクを含む評価セットと、少なくとも 3 つの安全ルールです。

<details>
<summary>参考解答と解説</summary>

1. 合格レベルの答えでは、agent loop を goal、plan、tool call、observation、memory/state update、stop condition として説明します。
2. 証拠には、最終回答だけでなく、別の開発者が確認できる trace を残します。
3. tool schema、permission boundary、retry、evaluation case、人間レビューなど、安全性または信頼性の制御を1つ説明できれば十分です。

</details>
