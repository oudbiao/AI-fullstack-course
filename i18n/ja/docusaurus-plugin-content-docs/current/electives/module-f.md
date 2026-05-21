---
title: "E.F AI プロダクト設計思考"
sidebar_position: 6
description: "作り始める前に、価値、コスト、リスク、UX、ローンチ阻止条件で AI プロダクト案を評価する。"
keywords: [AI product design, product thinking, evaluation, cost, UX, product strategy]
---

# E.F AI プロダクト設計思考

AI プロダクト設計は、モデルの能力ではなくユーザーの問題から始めます。機能を作る価値があるのは、価値、コスト、リスク、ユーザー体験を説明できるときです。

## まず意思決定ループを見る

![AI プロダクト意思決定マトリクス](/img/course/elective-ai-product-decision-matrix-ja.webp)

![AI プロダクト実験と指標ループ](/img/course/elective-ai-product-experiment-metrics-loop-ja.webp)

最初のプロダクト習慣は、実装前にトレードオフを明確にすることです。

## 小さな優先度スコアを動かす

```python
ideas = [
    {"name": "AI Tutor", "value": 9, "cost": 6, "risk": 4, "ux": 8},
    {"name": "AI Customer Service", "value": 8, "cost": 5, "risk": 5, "ux": 7},
    {"name": "AI Code Review", "value": 7, "cost": 4, "risk": 6, "ux": 6},
    {"name": "AI Medical Diagnosis", "value": 9, "cost": 8, "risk": 9, "ux": 5},
]


def score(item):
    return round(
        item["value"] * 0.45
        + (10 - item["cost"]) * 0.2
        + (10 - item["risk"]) * 0.2
        + item["ux"] * 0.15,
        2,
    )


def decision(item):
    if item["risk"] >= 8:
        return "do_not_launch"
    return "pilot" if item["score"] >= 6 else "wait"


ranked = sorted(({**item, "score": score(item)} for item in ideas), key=lambda item: item["score"], reverse=True)

for item in ranked:
    print(item["name"], "score=", item["score"], "decision=", decision(item))
```

期待される出力：

```text
AI Tutor score= 7.25 decision= pilot
AI Customer Service score= 6.65 decision= pilot
AI Code Review score= 6.05 decision= pilot
AI Medical Diagnosis score= 5.4 decision= do_not_launch
```

数値は最終的な真実ではありません。何を最適化しているのか、どこでローンチを止めるべきかを明確にするための道具です。

## プロダクトチェックリスト

| 質問 | 良い答え |
|---|---|
| 誰が困っているか？ | 具体的なユーザー群とタスク |
| 何が改善するか？ | 完了率、時間短縮、品質、コスト |
| 何が失敗しうるか？ | リスク境界と人間のフォールバック |
| 進展をどう証明するか？ | 指標またはユーザーテスト結果 |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
脅威モデル：prompt injection、data leak、tool misuse、unsafe output、または model abuse
制御: 検証、権限、サンドボックス、監査、レッドチームテスト、またはインシデント対応
テストケース：1 つの攻撃または失敗サンプルと、期待される安全な挙動
失敗確認: モデルの文を信じる、ログ不足、広すぎる権限、または回帰テストなし
期待される成果: セキュリティチェックリストと1件の再現可能なレッドチーム事例
```

## 合格チェック

AI 機能案を1つスコア化し、トレードオフを説明し、成功指標を定義し、ローンチすべきでない条件を1つ言えれば合格です。

<details>
<summary>確認の考え方と解説</summary>

合格する答えは、具体的な機能案、トレードオフ、成功指標、そしてローンチ阻止条件を示します。証拠は、漠然としたビジョンではなく、実行可能な決定ノートとして残してください。

リスク境界が明確でなければ、スコアが高くてもそのまま出すべきではありません。

</details>
