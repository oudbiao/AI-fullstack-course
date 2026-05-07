---
title: "6.8.1 深層学習プロジェクトロードマップ：学習、確認、パッケージ化"
sidebar_position: 0
description: "短い深層学習プロジェクトロードマップです。画像分類、感情分析、生成実践、学習証拠、ポートフォリオ化を扱います。"
keywords: [深層学習プロジェクトガイド, 画像分類, 感情分析, 生成実践, PyTorch ポートフォリオ]
---

# 6.8.1 深層学習プロジェクトロードマップ：学習、確認、パッケージ化

この小章は第6章の出口です。深層学習プロジェクトは学習スクリプトだけではありません。データ証拠、shape 確認、loss ログ、予測サンプル、失敗例、README が必要です。

## まずプロジェクトループを見る

![深層学習プロジェクトポートフォリオロードマップ](/img/course/ch06-projects-portfolio-loop-ja.png)

![深層学習プロジェクト学習レビューループ](/img/course/ch06-deep-learning-project-cycle-ja.png)

```text
データセット -> モデル -> 学習ログ -> 評価 -> 失敗例 -> パッケージ化
```

## 証拠記録を1つ残す

`dl_project_evidence_first_loop.py` を作ります。

```python
evidence = {
    "task": "image classification",
    "baseline_accuracy": 0.71,
    "current_accuracy": 0.82,
    "failure_case_count": 5,
    "next_step": "inspect confused classes and add augmentation",
}

print("task:", evidence["task"])
print("improvement:", round(evidence["current_accuracy"] - evidence["baseline_accuracy"], 3))
print("failure_case_count:", evidence["failure_case_count"])
print("next_step:", evidence["next_step"])
```

出力：

```text
task: image classification
improvement: 0.11
failure_case_count: 5
next_step: inspect confused classes and add augmentation
```

これがプロジェクト習慣です。改善には baseline、指標、失敗証拠、次の一手が必要です。

## この順番で学ぶ

| 順番 | 読む | 提出するもの |
|---|---|---|
| 1 | [6.8.2 画像分類](./01-image-classification.md) | データセット、CNN/転移 baseline、予測サンプル |
| 2 | [6.8.3 感情分析](./02-sentiment-analysis.md) | テキスト処理、学習ログ、エラー例 |
| 3 | [6.8.4 生成実践](./03-generative-practice.md) | 生成サンプルとレビュー記録 |
| 4 | [6.8.5 DL 実践ワークショップ](./04-hands-on-dl-workshop.md) | 再現可能な PyTorch 証拠パック |

## プロジェクト成果物基準

少なくとも1つのプロジェクトで、`README.md`、実行コマンド、データセットメモ、モデル概要、loss 曲線またはログ、指標表、予測サンプル、失敗例、次の計画を残します。

## 合格ライン

別の学習者がプロジェクトを実行し、学習証拠を確認し、成功例と失敗例を見て、次に何を改善するか理解できれば合格です。
