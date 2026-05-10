---
sidebar_position: 3
title: "0.4 おすすめ学習ルート"
description: "まず実用的な学習ルートを1つ選び、第1章へ進みます。"
keywords: [AI学習ルート, AIフルスタックルート, LLM学習, Agent学習, RAG学習]
---

# 0.4 おすすめ学習ルート

![おすすめ学習ルート選択図](/img/course/intro-learning-path-selection-ja.webp)

迷うなら初心者ルートを選びます。**第1章 -> 第9章を順番に進め、各ステージで小さな成果を1つ残します。**

| 目標 | 初回ルート |
|---|---|
| 初心者 | 第1-9章を順番に進める |
| すでにコードを書ける | 1-6を速く読み、7-9を重点的に読む |
| ポートフォリオが必要 | README、画像、ログ、失敗サンプルを残す |
| モデルを深く理解したい | 数学、機械学習、深層学習、Transformer に時間を使う |

## ルートを1つ選ぶ

| ルート | 向いている人 | 章 | 作るもの |
|---|---|---|---|
| 初心者フルルート | 初学者またはキャリア変更中の人 | 1 -> 9、その後 10-12 を選ぶ | 各 main track の後に動く mini project |
| Builder ルート | 早く LLM app を作りたい開発者 | 1-6 を skim、7 -> 9 を重点 | RAG app、Agent trace、evaluation notes |
| Model ルート | ML の直感を深めたい人 | 1 -> 7、その後 10-12 | model experiments、metric comparison、failure analysis |
| Portfolio ルート | job focused learner | 1 -> 9、README を強化 | setup、screenshots、logs、limits を含む公開 project story |

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
