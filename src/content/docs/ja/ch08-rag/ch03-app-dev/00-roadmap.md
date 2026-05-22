---
title: "8.3.1 アプリ開発ロードマップ：API、ツール、状態"
description: "LLM アプリ開発の短い実践ロードマップ：API 呼び出しを包み、ツール操作を検証し、対話状態を管理し、プロダクトループを作る。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "LLM アプリ開発ガイド, 対話システム, Function Calling, LangChain, 大規模モデルアプリ"
---

# 8.3.1 アプリ開発ロードマップ：API、ツール、状態

LLM アプリ開発は、入力欄とモデル API だけではありません。実際の機能では、入力検証、モデル呼び出し、ツール利用、状態保存、出力解析、エラーログ、回復可能な UX が必要です。

## まずアプリケーションループを見る

![LLM アプリケーション開発章の関係図](/img/course/ch08-app-dev-chapter-flow-ja.webp)

![LLM アプリケーション開発の学習順序図](/img/course/ch08-app-dev-learning-order-map-ja.webp)

![LLM アプリケーション能力ループ図](/img/course/ch08-llm-app-capability-loop-ja.webp)

この章では、1 回のモデル呼び出しを保守できるアプリケーションループにします：input、prompt/context、model、optional tool、validation、output、feedback です。

## ツール dispatch チェックを動かす

Function Calling では、モデルが構造化された操作引数を提案します。ただし、検証して dispatch する責任はアプリケーション側にあります。

```python
model_output = {
    "tool": "search_docs",
    "arguments": {"query": "RAG citations"},
}

allowed_tools = {
    "search_docs": {"required": ["query"]},
    "create_ticket": {"required": ["title", "priority"]},
}

tool = model_output["tool"]
required = allowed_tools[tool]["required"]
validation_ok = all(name in model_output["arguments"] for name in required)

print("validation_ok:", validation_ok)
print("dispatch:", tool if validation_ok else "block")
```

期待される出力：

```text
validation_ok: True
dispatch: search_docs
```

モデルの文章から tool call を直接実行しないでください。tool 名、argument schema、権限、失敗経路を検証します。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | LLM API 実践 | timeout と error handling 付きの安定した call wrapper を書く |
| 2 | Framework 基礎 | prompt、model、tool、memory、retrieval、parser の責務を分ける |
| 3 | 関数呼び出し | dispatch 前に構造化 tool arguments を検証する |
| 4 | Hugging Face エコシステム | hosted、local、browser-side モデルの適性を判断する |
| 5 | 対話システム | session state、slots、memory、user feedback を保存する |
| 6 | 文書とテンプレートアプリ | parsing、extraction、generation を module に分ける |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
要求: 入力、状態、tools/context、期待される出力の契約
検証済み出力：パーサー/スキーマ、または業務ルール確認の結果
追跡記録：モデル呼び出し、ツール/関数呼び出し、文書解析、または対話状態
失敗確認: フォーマット不正、必須フィールド不足、古い状態、または誤ったツール
次の行動：prompt、schema、state、API、または parsing の改善
```

## 合格ライン

1 回の API call、1 つの optional tool call、1 つの structured output、1 つの error path を持つ小さな assistant loop を作れれば、この章は合格です。

出口ミニプロジェクトは、コース Q&A と学習計画助手です。ユーザー依頼を分類し、必要なら知識を検索し、構造化された提案を返し、feedback を記録します。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、query から chunks、retrieval scores、引用 evidence、answer、fallback behavior までの流れを追跡します。
2. 証拠には、retrieved passages、source metadata、引用付き回答、空振りまたは誤検索の例を含めます。
3. 失敗原因が chunking、retrieval、ranking、prompt assembly、source 不足、根拠のない生成のどれかを説明できればよいです。

</details>
