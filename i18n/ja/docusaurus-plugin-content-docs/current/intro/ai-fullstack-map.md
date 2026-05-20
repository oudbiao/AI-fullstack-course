---
sidebar_position: 2
title: "0.3 AI フルスタック能力マップ"
description: "AIフルスタック学習の7つの能力層を1枚で見るための短いマップです。"
keywords: [AIフルスタック, 能力マップ, AI学習ルート, LLMアプリ, RAG, AI Agent]
---

# 0.3 AI フルスタック能力マップ

![AI フルスタック能力マップ](/img/course/intro-ai-fullstack-capability-map-ja.webp)

まず図を見ます。コースは1本の道です。

```text
tools -> Python -> data -> models -> LLM -> RAG -> Agent -> specialization/delivery
```

今は細部を全部理解しなくて大丈夫です。

| 詰まっていること | 戻る場所 |
|---|---|
| コードが動かない | ツールと Python |
| 入力が乱れている | データ |
| 答えが信頼できない | 評価と RAG |
| 行動が制御できない | Agent trace と権限 |

## 7つの層

| 層 | 対応章 | 最初に見える証拠 | 深い問い |
|---|---|---|---|
| ツール | 1 | 再現可能なプロジェクトフォルダと Git 履歴 | 他の人が再実行できるか |
| Python | 2 | 入力と出力が明確な小さなスクリプト | 読みやすく、型があり、テストできるか |
| データ | 3 | 整った表、グラフ、メモ | どこに誤りやバイアスがあるか分かるか |
| モデル | 4-6 | 学習または点検したモデル実験 | どの指標が判断を変えるか |
| LLM | 7 | prompt、tokens、embeddings、Transformer の直感 | 振る舞いは data、decoding、context のどこから来るか |
| RAG | 8 | 検索トレースと回答評価 | 答えは正しい根拠を使ったか |
| Agent | 9 | ツールトレース、権限、記憶境界、デプロイメモ | ユーザー、ファイル、操作が本物になったらどこで失敗するか |
| 専門分野 / 成果物 | 10-12 と選択モジュール | vision/NLP/マルチモーダルデモ、書き出した成果物、デプロイメモ | ドメイン制約がプロダクト判断をどう変えるか |

この講座はトピックの山ではなく、デバッグの積み重ねです。AI アプリケーションの挙動が悪いとき、原因は見ている機能より何層も下にあることがあります。

## メインラインと拡張トラック

まず第1-9章を標準のメインラインとして進めます。第9章まで終えると、小さな LLM/RAG/Agent プロジェクトを、根拠、ログ、安全境界つきで作れる状態を目指します。

その後、第10-12章はプロダクト上の必要に合わせて選びます。

| 必要なこと | 選ぶ章 | 理由 |
|---|---|---|
| 画像、カメラ、OCR、検出、セグメンテーション | 第10章 Computer Vision | 出力がラベル、枠、マスク、文字、動画イベントなどの視覚結果になる |
| テキストラベル、抽出、要約、言語評価 | 第11章 NLP | 出力がラベル、フィールド、範囲、生成テキストなどのテキストタスクになる |
| 画像、PDF、音声、動画、クリエイティブ素材、multimodal RAG | 第12章 Multimodal/AIGC | モダリティが混ざり、出典、プロンプト、レビュー、書き出し記録が必要になる |
| デプロイ、Python 上級、古典的 ML の深掘り | 選択モジュール | メインプロジェクトに特定のエンジニアリングまたはアルゴリズムの補助スキルが必要になる |

## マップの使い方

プロジェクトを始める前に、最も危険な層を1つ印づけます。たとえば PDF 質問応答アプリは、チャット UI ではなくデータ整備と検索で先に失敗しがちです。自動化 Agent は、プロンプトの言い回しではなくツール権限、状態、評価で先に失敗しがちです。

各章では、その層が動くことを証明する成果物を1つ残します。スクリーンショットも役に立ちますが、ログ、README コマンド、小さなデータセット、指標表、失敗メモのほうが、後でデバッグしやすい証拠になります。

任意の背景：これらの能力がどのように発展してきたかを知りたい場合は、[AI 発展史 15 段階マップ](/appendix/ai-milestones)を軽く見てください。

次に学習ルートを選びます。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
capability_map: tools, Python, data, math, ML, DL, LLM, RAG, Agent, and specialization links
current_position: what you already know and what you will postpone
next_step: one concrete chapter or workshop to start next
risk_check: learning everything at once, skipping evidence, or losing the main route
Expected_output: a marked personal course map with one immediate action
```
