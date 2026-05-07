---
sidebar_position: 1
title: "AI フルスタック能力マップ"
description: "AI フルスタック学習の 7 つの能力レイヤーを、コンパクトな図で理解します。"
keywords: [AI フルスタック, 能力マップ, AI 学習ルート, LLM アプリケーション, RAG, AI Agent]
---

# AI フルスタック能力マップ

![AI フルスタック能力全体マップ](/img/course/intro-ai-fullstack-capability-map-ja.png)

このページは暗記リストではなく、地図として使います。コースの主線は 1 文です。

```text
tools -> data -> models -> LLMs -> applications -> Agents -> engineering
```

## 7 つのレイヤー

| レイヤー | なぜ重要か | 残すべき成果 |
| --- | --- | --- |
| Tools | 安定して書き、実行し、保存する場所が必要 | 実行可能フォルダ、README、Git commit |
| Data | AI 作業は確認できるデータから始まる | データレポート、グラフ、 cleaned file |
| Models | モデルがどう学び、どう失敗するかを知る | baseline、metrics、error samples |
| LLMs | Prompt、Embedding、Transformer、context が見えやすくなる | Prompt tests、説明メモ |
| Applications | モデル能力を使える機能にする | chat、document tool、knowledge-base Q&A |
| Agents | AI が手順を考え、ツールを呼び、trace を残す | tool logs、task trace、permission rule |
| Engineering | 実プロジェクトには deploy、evaluation、cost、safety が必要 | Demo、monitoring notes、evaluation report |

## この地図の読み方

上から下へ一度眺めたら、そこで止めます。最初から全分岐を理解する必要はありません。

迷ったときは、この問いに戻ります。

> 今のプロジェクトは、どのレイヤーで止まっているか？

コードが動かないなら Tools。答えに根拠がないなら Data または RAG。Agent が不安定なら trace、権限、評価に戻ります。
