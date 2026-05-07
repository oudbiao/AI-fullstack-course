---
sidebar_position: 10
title: "AI 学習アシスタント リポジトリテンプレート"
description: "コース全体で育てる AI 学習アシスタントのための、短いリポジトリ構成、README、評価、trace テンプレートです。"
keywords: [AI学習アシスタント, プロジェクトテンプレート, ポートフォリオ, RAGテンプレート, Agentテンプレート]
---

# AI 学習アシスタント リポジトリテンプレート

![AI 学習アシスタント リポジトリ証拠キャビネット](/img/course/intro-ai-assistant-repo-evidence-cabinet-ja.png)

リポジトリは証拠キャビネットです。各フォルダは、動く、確認できる、評価できる、改善できる、のどれかを示します。

## 1. まずこの構成から

```text
ai-learning-assistant/
  README.md
  requirements.txt
  .env.example
  src/
  data/
  evals/
  logs/
  docs/
  tests/
```

1-3章では `README.md`、`src/`、`data/`、`docs/` だけで十分です。RAG、Agent、評価、ログを学ぶ段階で `evals/` や `logs/` を追加します。

## 2. 各フォルダが示すこと

| フォルダ | 証拠 |
|---|---|
| `src/` | 動くコードがある |
| `data/` | 入力と素材が明確 |
| `evals/` | 結果を再評価できる |
| `logs/` | 失敗と trace を確認できる |
| `docs/` | スクリーンショットと判断理由が見える |
| `tests/` | 修正を後で確認できる |

## 3. 最小 README

````md
# AI 学習アシスタント

## 目的

## 実行方法
```bash
pip install -r requirements.txt
python -m src.app
```

## 例

## 評価

## 既知の失敗

## 次の一手
````

## 4. 最初の評価と trace ファイル

```jsonl
{"id":"q001","question":"Why does RAG need citations?","expected_sources":["ch08-rag"]}
```

```json
{
  "run_id": "demo-001",
  "user_input": "Help me review RAG",
  "steps": [
    {"action": "retrieve", "sources": ["ch08-rag"]},
    {"action": "generate_plan", "status": "ok"}
  ],
  "failure": null
}
```

発表では、実行コマンド、サンプル入力、評価ケース、trace、失敗記録、スクリーンショットを見せます。
