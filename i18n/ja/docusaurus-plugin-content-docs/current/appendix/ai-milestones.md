---
title: "A.3 AI 発展史：15 段階と重要論文"
sidebar_position: 2
description: "図を先に見る形で AI 発展の 15 段階をつかみ、機械学習、深層学習、LLM、RAG、Agent、マルチモーダル AI で初学者が先に知っておきたい重要論文とアルゴリズムを整理します。"
keywords: [AI発展史, AI発展段階, 重要論文, Transformer論文, GPT論文, RAG, Agent, 拡散モデル論文]
---

# A.3 AI 発展史：15 段階と重要論文

![AI 15段階発展史マップ](/img/course/appendix-ai-15-stage-history-map-ja.webp)

このページは任意の背景資料です。「この概念はどこから来たのか」を知るためのもので、初回から論文名を暗記するためのページではありません。

おすすめの使い方:

1. まず 15 段階の図を見る。
2. 段階表をざっと読む。
3. 今学んでいる章に関係する段階だけを見る。
4. 後で論文名やアルゴリズム名が出てきたら戻ってくる。

## 15 段階マップ

| 段階 | 初学者向けの意味 | 対応する章 |
|---|---|---|
| 1. AI という問い | 機械は知的にふるまえるのか | 導入 |
| 2. 記号主義 AI | 人がルールを書き、機械がルールで推論する | 背景知識 |
| 3. エキスパートシステム | 専門知識をルールベースのソフトウェアにする | システム思考 |
| 4. 確率と統計 | 固定ルールだけでなく、証拠と不確実性で判断する | 第 4 章 |
| 5. 古典的機械学習 | データと特徴量からパターンを学ぶ | 第 5 章 |
| 6. 初期ニューラルネット | モデルが単純な判断境界を学び始める | 第 5-6 章 |
| 7. 誤差逆伝播 | 多層ネットワークが本格的に学習可能になる | 第 6 章 |
| 8. カーネルとアンサンブル | SVM、木、森、Boosting が ML を実用的にする | 第 5 章 |
| 9. 深層学習の突破 | データ + GPU + 深いネットワークが画像と音声を開く | 第 6、10 章 |
| 10. 埋め込みと系列モデル | テキストがベクトルになり、系列を学習できる | 第 11 章 |
| 11. Transformer と事前学習 | Attention が大規模言語モデルを実用化する | 第 6-7 章 |
| 12. LLM とアラインメント | モデルが指示に従うアシスタントらしくなる | 第 7 章 |
| 13. RAG | モデルが外部知識と引用に接続する | 第 8 章 |
| 14. Agent とツール利用 | モデルが計画し、ツールを呼び、実行履歴を残す | 第 9 章 |
| 15. マルチモーダルと AIGC | AI がテキスト、画像、音声、動画、生成を扱う | 第 12 章 |

一番大事な流れはシンプルです。各段階は前の段階の限界を解き、同時に新しいエンジニアリング課題を生みます。

## 主線をリレーとして読む

![AI 主線の駅伝全体図](/img/course/appendix-ai-main-relay-map-ja.webp)

AI の歴史は、論文名の一覧というよりリレーに近いです。

| 受け渡し | 何が変わったか |
|---|---|
| ルール -> 確率 | システムが固定ロジックから不確実な証拠へ進んだ |
| 確率 -> 機械学習 | モデルがデータからパターンを学び始めた |
| 機械学習 -> 深層学習 | 特徴量を完全に手作りするのではなく、モデルが学ぶようになった |
| 深層学習 -> Transformer | 系列モデリングを大規模化しやすくなった |
| LLM -> RAG / Agent | モデルが知識、ツール、ワークフローに接続した |
| テキスト -> マルチモーダル | AI が複数のメディアを理解し生成し始めた |

## まず覚えたい 6 つの転換点

![AI 歴史の転換点コミック](/img/course/appendix-ai-history-comic-turning-points-ja.webp)

| 転換点 | 初学者が気にする理由 |
|---|---|
| パーセプトロン | 機械がデータから学べるかもしれないという強い期待を生んだ |
| XOR の限界 | 単純な線形モデルだけでは足りないことを示した |
| 誤差逆伝播 | 多層ニューラルネットワークが実用的に学習可能になった |
| AlexNet | データ、GPU、深い CNN が深層学習を一気に押し上げた |
| Transformer | Attention が系列モデリングの主線を書き換えた |
| RAG / Agent | モデルが文章回答から知識とツール利用へ進んだ |

最初から年号を覚える必要はありません。まずは流れを覚えます。期待、挫折、修復、スケール、エンジニアリングです。

## 論文ノードの読み方

![AI 論文の問題・方法・影響チェーン](/img/course/appendix-ai-paper-problem-solution-impact-chain-ja.webp)

どんな論文やアルゴリズムでも、最初は 4 つだけ問いましょう。

| 問い | 例: `Attention Is All You Need` |
|---|---|
| 以前のボトルネックは何か | RNN は並列化しにくく、長距離依存の経路も長かった |
| 新しい方法は何か | self-attention、multi-head attention、position encoding |
| どんな能力が開いたか | 大規模化しやすい系列モデリングと、その後の大規模言語モデル |
| どんなプロジェクトに影響したか | LLM、RAG、Agent、マルチモーダルモデル |

初学者の歴史理解としては、ここまでで十分です。数式の詳細は、該当する章まで進んでからで大丈夫です。

## コース主線別の重要ノード

![プロジェクト視点で見る AI タイムライン](/img/course/appendix-ai-project-lens-map-ja.webp)

| コース主線 | 先に知っておきたいノード | なぜ重要か |
|---|---|---|
| 数学基礎 | Bayes、Shannon、最尤推定、EM | 確率、情報量、損失関数 |
| 古典的機械学習 | CART、SVM、Random Forest、AdaBoost、XGBoost | 強いベースラインと表データの実務 |
| ニューラルネット | Perceptron、XOR、Backpropagation、LSTM、AlexNet、ResNet | 深さ、勾配、データ、計算資源がなぜ重要か |
| NLP と LLM | Word2Vec、Seq2Seq、Transformer、BERT、GPT、InstructGPT | 単語ベクトルからアシスタントへ進む流れ |
| RAG と Agent | RAG、Chain-of-Thought、ReAct、Toolformer | 外部知識、推論トレース、ツール利用 |
| マルチモーダル | CLIP、DDPM、Latent Diffusion、Whisper、SAM | テキスト、画像、音声、動画、生成パイプライン |

具体的な論文もあれば、アルゴリズム群や歴史的な転換点もあります。大事なのは、「そのノードは何を簡単にしたのか」です。

## 任意の分岐図

関連する章を学んでいるときだけ見れば十分です。

![3 回のニューラルネットワーク波と 2 回の谷](/img/course/appendix-neural-network-waves-timeline-ja.webp)

![古典的機械学習の分岐図](/img/course/appendix-classic-ml-branch-map-ja.webp)

![NLP から LLM への系譜図](/img/course/appendix-nlp-llm-lineage-map-ja.webp)

![アラインメント、Agent、システム主線図](/img/course/appendix-agent-system-lineage-map-ja.webp)

![LLM から Agent へのエンジニアリング進化タイムライン](/img/course/appendix-llm-to-agent-evolution-timeline-ja.webp)

![マルチモーダルと AIGC の系譜図](/img/course/appendix-multimodal-aigc-lineage-map-ja.webp)

## 章別クイック索引

| この名前を見たら | 戻る章 |
|---|---|
| Bayes、MLE、entropy、EM | 第 4 章 数学基礎 |
| SVM、Random Forest、XGBoost | 第 5 章 機械学習 |
| Perceptron、backpropagation、CNN、LSTM、Transformer | 第 6 章 深層学習 |
| GPT、RLHF、LoRA、instruction tuning | 第 7 章 LLM 原理 |
| RAG、vector retrieval、citations | 第 8 章 RAG |
| Chain-of-Thought、ReAct、Toolformer、tool use | 第 9 章 Agent |
| AlexNet、ResNet、YOLO、SAM | 第 10 章 コンピュータビジョン |
| Word2Vec、Seq2Seq、BERT、GPT | 第 11 章 NLP |
| CLIP、diffusion、Whisper、マルチモーダル生成 | 第 12 章 マルチモーダル |

## 小さな練習

好きなノードを 3 つ選び、プロジェクトの言葉に書き換えます。

```text
ノード: Attention Is All You Need
以前のボトルネック: RNN は長い系列や並列学習に向いていなかった。
新しい方法: self-attention が系列モデリングの主線になった。
影響したプロジェクト: LLM、RAG、Agent、マルチモーダルモデル。
戻って学ぶ章: 第 6、7、8、9 章。
```

目的は歴史を暗記することではありません。歴史上のノードを、これから作る実際の能力と結びつけることです。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
timeline_anchor: stage, key idea, representative paper/system, and why it mattered
chapter_link: which course chapter this milestone helps explain
memory_hook: diagram, comic panel, or one-sentence historical turn
failure_check: memorizing names without understanding the problem each milestone solved
Expected_output: a short timeline note connected to at least one project decision
```
