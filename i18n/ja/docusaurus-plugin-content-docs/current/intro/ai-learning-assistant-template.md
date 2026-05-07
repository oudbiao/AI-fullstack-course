---
sidebar_position: 10
title: "AI 学習アシスタント リポジトリテンプレート"
description: "AI 学習アシスタント通しプロジェクトの短いリポジトリ構成、README、評価、trace テンプレートです。"
keywords: [AI学習アシスタント, プロジェクトテンプレート, ポートフォリオプロジェクト, RAGプロジェクトテンプレート, Agentプロジェクトテンプレート]
---

# AI 学習アシスタント リポジトリテンプレート

![AI 学習アシスタント リポジトリ証拠キャビネット](/img/course/intro-ai-assistant-repo-evidence-cabinet-ja.png)

このテンプレートはディレクトリの飾りではありません。コード、データ、ログ、評価、スクリーンショットで、プロジェクトが実行・確認できることを示す証拠キャビネットです。

## 最小ディレクトリ構成

```text
ai-learning-assistant/
  README.md
  requirements.txt
  .env.example
  src/
    app/
    rag/
    agent/
  data/
    raw/
    processed/
  evals/
    questions.jsonl
    results/
  logs/
    traces/
    failures/
  docs/
    screenshots/
    decisions.md
  tests/
```

最初は小さく始めます。第 1-3 章では `README.md`、`src/`、`data/`、`docs/screenshots/` だけで十分です。対応する能力に入ったら、`evals/`、`logs/`、`rag/`、`agent/` を追加します。

## 各フォルダが証明すること

| フォルダ | 証明 |
| --- | --- |
| `src/` | システムに実行可能なコードがある |
| `data/` | 入力と材料が明示されている |
| `evals/` | 結果を判断できる |
| `logs/` | 失敗と trace を確認できる |
| `docs/` | 他の人がプロジェクトを理解できる |
| `tests/` | 修正後にもう一度確認できる |

## 最小 README

````md
# AI 学習アシスタント

## 目標
このアシスタントはどの学習問題を解決するか？

## 現在のバージョン
v0.x：

## 実行方法
```bash
pip install -r requirements.txt
python -m src.app.cli
```

## 例
入力：
出力：

## 評価
どの固定質問、指標、手動チェックを使うか？

## 失敗サンプル
何が失敗し、次に何を変えるか？
````

## 最小評価と trace 例

```jsonl
{"id":"q001","question":"なぜ RAG には引用が必要ですか？","expected_sources":["ch08-rag"],"ideal_points":["grounding","evaluation","failure cases"]}
```

```json
{
  "run_id": "demo-001",
  "user_input": "RAG の復習を手伝って",
  "steps": [
    {"action": "retrieve", "sources": ["ch08-rag"]},
    {"action": "generate_plan", "status": "ok"}
  ],
  "failure": null
}
```

プロジェクトを見せるときは、リポジトリを証拠として説明します。実行コマンド、サンプルデータ、評価ケース、trace ログ、失敗メモ、スクリーンショットです。
