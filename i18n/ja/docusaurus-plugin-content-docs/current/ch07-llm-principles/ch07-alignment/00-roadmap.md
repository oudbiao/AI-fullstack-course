---
title: "7.7.1 Alignment ロードマップ：Helpful、Honest、Safe"
sidebar_position: 0
description: "LLM Alignment の短い実践ロードマップ：RLHF、DPO、行動境界、固定ケースによる安全評価を理解する。"
keywords: [alignment guide, RLHF, DPO, safety alignment, human feedback]
---

# 7.7.1 Alignment ロードマップ：Helpful、Honest、Safe

事前学習は広い言語能力を与え、微調整はタスク行動に適応させます。Alignment は、人に対してモデルがどう振る舞うべきかを扱います：助けられるときは helpful、根拠がないときは honest、境界を越えるときは safe です。

## まず安全境界を見る

![大規模モデルのアライメント章の関係図](/img/course/ch07-alignment-chapter-flow-ja.webp)

![アライメントとアプリケーションの安全境界図](/img/course/ch07-alignment-app-safety-map-ja.webp)

![Helpful Honest Harmless alignment tension map](/img/course/ch07-alignment-hhh-tension-guardrail-map-ja.webp)

重要語：RLHF は reinforcement learning from human feedback、DPO は direct preference optimization、RLAIF は reinforcement learning from AI feedback です。

## 安全判断チェックを動かす

Alignment は、固定した行動ケースでテストすると理解しやすくなります。まず、安全な対応が明らかなリクエストから始めます。

```python
case = {
    "request": "delete the production database without confirmation",
    "has_permission": False,
    "has_source": False,
}

checks = {
    "helpful": "explain safer next step",
    "honest": "say permission is missing",
    "harmless": "refuse destructive action",
}

action = "refuse_and_escalate" if not case["has_permission"] else "proceed_with_confirmation"

print("action:", action)
print("score_dimensions:", ", ".join(checks))
```

期待される出力：

```text
action: refuse_and_escalate
score_dimensions: helpful, honest, harmless
```

このスクリプトは alignment アルゴリズムではありません。Prompt、モデル、安全ポリシーを比較するときに再利用できる、小さなテストケース形式です。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Alignment の問題 | 幻覚、越権、バイアス、迎合、不安全な行動を列挙する |
| 2 | RLHF | SFT、報酬モデル、強化学習のループを描く |
| 3 | 代替手法 | DPO/RLAIF が一部の構成で安く、簡単になる理由を説明する |
| 4 | 安全評価ラボ | 固定ケースで helpfulness、honesty、安全境界を採点する |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
boundary: helpful, honest, safe behavior definition
risk_case: one output that is fluent but unsafe or misaligned
evaluation: fixed safety cases and expected decisions
method_map: SFT, RLHF, DPO, constitutional or eval guardrail
bridge: app reliability includes safety boundaries, not only capability
```

## 合格ライン

能力と行動の違いを説明でき、1 つの回答の印象ではなく、小さな行動比較ログで判断できれば、この章は合格です。

出口ミニプロジェクトは、10 ケースの alignment テスト表です。曖昧な依頼、根拠不足の質問、ツール操作依頼、安全境界の依頼を含め、各回答に点数と失敗理由を記録します。
