---
sidebar_position: 3
title: "0.4 おすすめ学習ルート"
description: "まず実用的な学習ルートを1つ選び、第1章へ進みます。"
keywords: [AI学習ルート, AIフルスタックルート, LLM学習, Agent学習, RAG学習]
---

# 0.4 おすすめ学習ルート

![おすすめ学習ルート選択図](/img/course/intro-learning-path-selection-ja.webp)

迷うなら初心者ルートを選びます。**第1章 -> 第9章を順番に進め、各ステージで小さな成果を1つ残します。その後、必要なときだけ第10-12章から specialization を1つ選びます。**

| 目標 | 初回ルート |
|---|---|
| 初心者 | 第1-9章を先に進め、その後 branch する |
| すでにコードを書ける | 1-6を速く読み、7-9を重点的に読み、その後 specialization を選ぶ |
| ポートフォリオが必要 | README、画像、ログ、metrics、traces、失敗サンプルを残す |
| モデルを深く理解したい | 数学、機械学習、深層学習、Transformer に時間を使い、その後 CV/NLP/multimodal depth を選ぶ |

## ルートを1つ選ぶ

| ルート | 向いている人 | 章 | 作るもの |
|---|---|---|---|
| 初心者フルルート | 初学者またはキャリア変更中の人 | 1 -> 9、その後 10-12 から1つ選ぶ | 各 main track の後に動く mini project、さらに specialization demo |
| Builder ルート | 早く LLM app を作りたい開発者 | 1-6 を skim、7 -> 9 を重点、multimodal が必要なら 12 | RAG app、Agent trace、evaluation notes、safety boundary |
| Model ルート | ML の直感を深めたい人 | 1 -> 7、その後 data type に合わせて 10 または 11 | model experiments、metric comparison、failure analysis |
| Portfolio ルート | job focused learner | 1 -> 9、README を強化し、その後 capstone direction を1つ選ぶ | setup、screenshots、logs、metrics、traces、limits を含む公開 project story |

## Stage Exit Checks

進捗は「何ページ読んだか」ではなく、evidence で判断します。

| Stage | Chapters | Minimum evidence | Experienced learners の deeper evidence |
|---|---|---|---|
| Foundations | 1-3 | 再現可能な project folder、Python scripts、cleaned data、charts | README rerun test、edge cases、data quality notes |
| Model understanding | 4-6 | metric と failure samples つきの model experiment | bias/variance notes、ablation、training diagnosis |
| LLM applications | 7-9 | Prompt tests、RAG retrieval trace、Agent tool trace | fixed eval set、safety boundary、cost/latency notes |
| Specialization | 10-12 | vision、NLP、multimodal demo のどれか1つと保存した inputs/outputs | domain metric、review checklist、deployment constraint |

specialization chapters は「全部終えた後のごほうび」ではありません。product が images、text pipelines、multimodal assets、domain-specific evaluation を必要とするときに選ぶ deliberate branch です。

## 毎週のループ

毎週同じループを使います。

```text
短く読む -> 1つ動かす -> 条件を1つ変える -> 証拠を記録する -> 1行ふり返る
```

ふり返りは短くてかまいません。よい問いは次のようなものです。

- 最初に失敗したものは何か。
- どの入力変更が出力を最も変えたか。
- 他の developer を納得させる証拠は何か。
- これが本物の user-facing feature になったら、どこで壊れるか。

## 飛ばす時、ゆっくり進む時

推測なしで chapter check を通れる場合だけ飛ばします。出力を説明できない、コードを再実行できない、結果の良し悪しを判断できないときは、ゆっくり進みます。経験者も、demo が簡単に見えるときほど evaluation、failure modes、production constraints では速度を落としてください。

毎週ルートを変えないでください。短く読み、動かし、証拠を残して、[第1章](/ch01-tools)へ進みます。
