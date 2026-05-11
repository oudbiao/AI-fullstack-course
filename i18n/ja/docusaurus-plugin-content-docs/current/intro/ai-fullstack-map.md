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
| Tools | 1 | 再現可能な project folder と Git history | 他の人が再実行できるか |
| Python | 2 | 入力と出力が明確な小さな scripts | 読みやすく、型があり、test できるか |
| Data | 3 | 整った tables、charts、notes | どこに誤りや bias があるか分かるか |
| Models | 4-6 | training または inspection した model experiments | どの metric が判断を変えるか |
| LLM | 7 | prompt、tokens、embeddings、Transformer の直感 | 振る舞いは data、decoding、context のどこから来るか |
| RAG | 8 | retrieval trace と answer evaluation | 答えは正しい evidence を使ったか |
| Agent | 9 | tool traces、permissions、memory boundary、deployment notes | users、files、actions が本物になったらどこで失敗するか |
| Specialization / delivery | 10-12 と electives | vision/NLP/multimodal demos、exported assets、deployment notes | domain constraints が product decision をどう変えるか |

この講座は topic の山ではなく、debugging stack です。AI application の挙動が悪いとき、原因は見ている機能より何層も下にあることがあります。

## Main Line と Expansion Tracks

まず第1-9章を default main line として進めます。第9章まで終えると、小さな LLM/RAG/Agent project を、evidence、logs、safety boundary つきで作れる状態を目指します。

その後、第10-12章は product need に合わせて選びます。

| Need | Choose | Why |
|---|---|---|
| images、cameras、OCR、detection、segmentation | Chapter 10 Computer Vision | output が labels、boxes、masks、text、video events などの visual result になる |
| text labels、extraction、summaries、linguistic evaluation | Chapter 11 NLP | output が labels、fields、spans、generated text などの text task になる |
| images、PDFs、audio、video、creative assets、multimodal RAG | Chapter 12 Multimodal/AIGC | modalities が混ざり、source、prompt、review、export records が必要になる |
| deployment、advanced Python、classic ML depth | Electives | main project に特定の engineering または algorithmic side skill が必要になる |

## マップの使い方

project を始める前に、最も危険な層を1つ印づけます。たとえば PDF 質問応答 app は、chat UI ではなく data cleaning と retrieval で先に失敗しがちです。automation Agent は、prompt wording ではなく tool permissions、state、evaluation で先に失敗しがちです。

各章では、その層が動くことを証明する artifact を1つ残します。screenshots も役に立ちますが、logs、README commands、小さな datasets、metric tables、failure notes のほうが、後で debug しやすい証拠になります。

任意の背景：これらの能力がどのように発展してきたかを知りたい場合は、[AI 発展史 15 段階マップ](/appendix/ai-milestones)を軽く見てください。

次に学習ルートを選びます。
