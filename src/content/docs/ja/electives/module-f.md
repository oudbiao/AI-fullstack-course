---
title: "E.F AI プロダクト設計思考"
description: "作り始める前に、価値、コスト、リスク、UX、ローンチ阻止条件で AI プロダクト案を評価する。"
sidebar:
  order: 6
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AI product design, product thinking, evaluation, cost, UX, product strategy"
---
AI プロダクト設計は、モデルの能力ではなくユーザーの問題から始めます。機能を作る価値があるのは、価値、コスト、リスク、ユーザー体験を説明できるときです。

## まず意思決定ループを見る

![AI プロダクト意思決定マトリクス](/img/course/elective-ai-product-decision-matrix-ja.webp)

![AI プロダクト実験と指標ループ](/img/course/elective-ai-product-experiment-metrics-loop-ja.webp)

最初のプロダクト習慣は、実装前にトレードオフを明確にすることです。

1 枚目の図はローンチゲートとして読みます。価値が高いだけでは足りず、リスク、コスト、ユーザー信頼を管理できる必要があります。2 枚目の図はローンチ後の運用リズムです。仮説、prototype、指標、ユーザーフィードバック、意思決定を回し続けます。

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

## スコアを意思決定メモにする

スクリプトが順位を出したら、最高点を見るだけで終わらせないでください。小さな意思決定メモを書きます。

意思決定メモの例:

- 機能: AI Tutor
- 決定: pilot
- 成功指標: 演習完了率を 15% 改善する。
- 主なリスク: 自信ありげだが間違った説明を返す。
- ローンチ阻止条件: 高リスク助言に review path がない。
- 次のテスト: 実際の学習者質問 10 件で試し、失敗を記録する。

このメモはプロダクト判断とエンジニアリング作業をつなぎます。指標、リスク、阻止条件、次の実験が明確なので、実装に移りやすくなります。

ローンチ阻止条件は数値スコアより重要です。AI 医療診断アシスタントはユーザー価値が高くても、臨床レビュー、監査ログ、エスカレーション、責任境界を用意できなければ止めるべきです。

## 最小限の有効なテストを設計する

完全な機能を作る前に、意思決定を変えられる小さなテストを定義します。

| アイデア | 最小限の有効なテスト |
|---|---|
| AI Tutor | 実際の学習者質問 10 件で試し、正しさ、トーン、次の一歩の分かりやすさを記録する。 |
| AI Customer Service | 過去チケット 30 件で試し、自動解決率と危険な回答率を測る。 |
| AI Code Review | 5 つの PR で AI コメントと人間の review を比較し、実行可能な指摘を数える。 |

テストの前に意思決定ルールを書きます。たとえば、10 件中少なくとも 8 件が正しく、高リスク回答が未レビューでなく、学習者が次の一歩を理解できた場合だけ pilot に進めます。

## プロダクトチェックリスト

| 質問 | 良い答え |
|---|---|
| 誰が困っているか？ | 具体的なユーザー群とタスク |
| 何が改善するか？ | 完了率、時間短縮、品質、コスト |
| 何が失敗しうるか？ | リスク境界と人間のフォールバック |
| 進展をどう証明するか？ | 指標またはユーザーテスト結果 |
| 何がローンチを止めるか？ | 漠然とした不安ではなく、具体的な阻止条件 |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
プロダクト課題：ユーザー問題、ワークフロー、価値指標、リスク境界
実験：仮説、最小テスト、指標、意思決定ルール
成果物: 機能仕様、prototype メモ、user story、または評価結果
失敗確認：価値を測らずにデモだけ作る、またはユーザーワークフローを無視する
期待される成果：実装を導ける AI プロダクト意思決定メモ
```

## 合格チェック

AI 機能案を1つスコア化し、トレードオフを説明し、成功指標を定義し、ローンチすべきでない条件を1つ言えれば合格です。

<details>
<summary>確認の考え方と解説</summary>

合格する答えは、具体的な機能案、トレードオフ、成功指標、そしてローンチ阻止条件を示します。証拠は、漠然としたビジョンではなく、実行可能な決定ノートとして残してください。

リスク境界が明確でなければ、スコアが高くてもそのまま出すべきではありません。

</details>
