---
title: "E.F AI プロダクトデザイン思考"
sidebar_position: 6
description: "AI プロダクト判断の短い実践ガイド。作る前に、ユーザー課題、価値、コスト、リスク、UX を採点します。"
keywords: [AI product design, product thinking, evaluation, cost, UX, product strategy]
---

# E.F AI プロダクトデザイン思考

AI プロダクト設計は、モデルの能力ではなくユーザーの課題から始まります。価値、コスト、リスク、ユーザー体験を説明できて初めて、作る意味があります。

## まず意思決定ループを見る

![AI プロダクト意思決定マトリクス](/img/course/elective-ai-product-decision-matrix-ja.png)

![AI プロダクト実験とメトリクスのループ](/img/course/elective-ai-product-experiment-metrics-loop-ja.png)

最初のプロダクト習慣は、実装前にトレードオフを見える形にすることです。

## 小さな優先順位スコアを動かす

```python
ideas = [
    {"name": "AI Tutor", "value": 9, "cost": 6, "risk": 4, "ux": 8},
    {"name": "AI Customer Service", "value": 8, "cost": 5, "risk": 5, "ux": 7},
    {"name": "AI Code Review", "value": 7, "cost": 4, "risk": 6, "ux": 6},
]

def score(item):
    return round(item["value"] * 0.45 + (10 - item["cost"]) * 0.2 + (10 - item["risk"]) * 0.2 + item["ux"] * 0.15, 2)

ranked = sorted(({**item, "score": score(item)} for item in ideas), key=lambda item: item["score"], reverse=True)

for item in ranked:
    print(item["name"], item["score"])
```

期待される出力:

```text
AI Tutor 7.25
AI Customer Service 6.65
AI Code Review 6.05
```

数字は最終的な真実ではありません。何を最適化しているのかを言葉にするための道具です。

## プロダクトチェックリスト

| Question | Good Answer |
|---|---|
| 誰が困っているか | 具体的なユーザー層と作業 |
| 何が改善するか | 完了率、短縮時間、品質、コスト |
| 何が起こると危ないか | リスク境界と人間への引き継ぎ |
| 進歩を何で証明するか | メトリクスまたはユーザーテスト結果 |

## 合格チェック

AI 機能案を 1 つ採点し、トレードオフを説明し、成功指標を定義し、リリースすべきでない条件を 1 つ言えれば合格です。
