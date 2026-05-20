---
title: "8.1.1 RAG ロードマップ：文書、検索、回答"
sidebar_position: 0
description: "RAG の短い実践ロードマップ：文書を検索可能なチャンクにし、根拠を検索し、引用付きで答え、失敗を評価する。"
keywords: [RAG ガイド, retrieval-augmented generation, ベクトルデータベース, 文書チャンク, reranking, RAG 評価]
---

# 8.1.1 RAG ロードマップ：文書、検索、回答

RAG は実務的な問題を解きます。モデルは最新情報、非公開情報、出典が必要な事実をすべて知っているわけではないため、アプリケーションが先に根拠を検索してから回答させます。

## まず RAG パイプラインを見る

![LLM アプリケーションにおける RAG の位置づけを示す橋渡し図](/img/course/ch08-rag-position-bridge-ja.webp)

![RAG コア章の学習順序図](/img/course/ch08-rag-core-chapter-flow-ja.webp)

![RAG で資料から回答へ進むパイプライン図](/img/course/ch08-rag-data-to-answer-pipeline-ja.webp)

基本ループは、文書読み込み、chunk 分割、metadata 付与、embedding、検索、rerank、context 組み立て、回答、出典引用、評価です。

## 小さな検索チェックを動かす

これはまだベクトルデータベースではありません。検索の習慣を小さく再現します：チャンクに点を付け、出典を表示し、根拠が質問に合うか確認します。

```python
chunks = [
    {"source": "rag.md", "text": "RAG retrieves source chunks before the model answers."},
    {"source": "eval.md", "text": "Citations let users verify whether an answer is grounded."},
    {"source": "deploy.md", "text": "Deployment exposes the model through a stable API."},
]

query = "why do RAG answers need citations"
query_terms = set(query.lower().split())

def score(chunk):
    words = set(chunk["text"].lower().replace(".", "").split())
    return len(query_terms & words)

for chunk in sorted(chunks, key=score, reverse=True)[:2]:
    print(chunk["source"], score(chunk))
```

期待される出力：

```text
rag.md 2
eval.md 1
```

先頭の出典が無関係なら、最終 Prompt を先に直さないでください。文書解析、chunk 分割、metadata、検索カバレッジを確認します。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | RAG 基礎 | 質問 → 根拠 → 回答のループを描く |
| 2 | 文書処理 | source と metadata を持つ chunks を作る |
| 3 | ベクトルデータベース | embedding、ベクトルレコード、類似検索を説明する |
| 4 | 検索戦略 | キーワード、ベクトル、ハイブリッド、filter、rerank を比較する |
| 5 | 最適化と高度な RAG | recall 不足、ranking 不良、弱い コンテキスト を調べる |
| 6 | RAG 評価 | 回答正しさ、引用根拠、no-answer 動作をテストする |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
query: one user question or test case
retrieved_chunks: chunk ids, scores, and source titles
answer: final response with citation or source note
failure_check: missing evidence, wrong chunk, stale doc, or unsupported claim
next_action: chunking, embedding, reranking, prompt, or eval change
```

## 合格ライン

10 個以上の固定質問に対して、検索チャンク、回答本文、出典引用を表示する最小知識ベース Q&A ループを作れれば、この章は合格です。

出口ミニプロジェクトは、コース知識ベース助手です。3〜5 件の Markdown 文書、top-k 検索出力、出典表示、簡単な評価表を用意します。
