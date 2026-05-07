---
title: "8.5.1 プロジェクトロードマップ：引用付き知識助手を作る"
sidebar_position: 0
description: "第 8 章キャップストーンの短い実践ロードマップ：引用付き RAG または LLM アプリを作り、検索ログ、失敗処理、評価、デプロイメモを残す。"
keywords: [LLM project guide, enterprise knowledge base, intelligent assistant, RAG project, courseware generation assistant]
---

# 8.5.1 プロジェクトロードマップ：引用付き知識助手を作る

このキャップストーンは、知識、モデル呼び出し、アプリケーションフロー、engineering evidence を 1 つの再現可能な LLM アプリに接続できることを示します。

## まずプロジェクト証拠ループを見る

![LLM アプリケーション総合プロジェクトのロードマップ](/img/course/ch08-projects-route-map-ja.png)

![LLM アプリケーションプロジェクトの学習順序図](/img/course/ch08-project-learning-order-map-ja.png)

![LLM アプリケーションプロジェクトのデリバリーループ図](/img/course/ch08-project-delivery-loop-ja.png)

プロジェクトは「ベクトルデータベースをつなぐ」だけではありません。文書、chunks、検索、context、回答、引用、logs、評価、改善の traceable loop です。

## プロジェクト readiness チェックを動かす

プロジェクト完了前に、この checklist を使います。

```python
project = {
    "project_type": "knowledge-base assistant",
    "documents": 5,
    "eval_questions": 10,
    "citations": True,
    "empty_retrieval_handled": True,
    "failure_cases": 3,
}

ready = (
    project["documents"] >= 3
    and project["eval_questions"] >= 10
    and project["citations"]
    and project["empty_retrieval_handled"]
    and project["failure_cases"] >= 1
)

print("ready:", ready)
print("project_type:", project["project_type"])
print("evidence:", "docs, eval, citations, failures")
```

出力：

```text
ready: True
project_type: knowledge-base assistant
evidence: docs, eval, citations, failures
```

`ready` が `False` なら、別の機能を足す前に evidence loop を完成させます。

## この順番で学ぶ

| 手順 | プロジェクト | 本当に鍛える力 |
|---|---|---|
| 1 | 企業またはコース知識ベース | 検索、権限、引用、traceable answers |
| 2 | 知的アシスタント | 検索、session state、tool calling を product feature にする |
| 3 | RAG + 微調整システム | 知識不足と振る舞い不安定を分ける |
| 4 | 教材生成助手 | 文書解析、構造化出力、template rendering |
| 5 | フル実践ワークショップ | 実 API や DB を足す前の最小再現ループ |

ガイド付き baseline が必要なら、[8.5.6 実践：第 8 章 RAG アプリ完全ワークショップ](./05-stage-hands-on-workshop.md) から始めます。

## プロジェクト成果物基準

| 成果物 | 最低要件 | 強いポートフォリオ版 |
|---|---|---|
| README | 目的、実行コマンド、依存関係、例 | アーキテクチャ図、設計 trade-off、コスト、振り返りを追加 |
| 知識ベースサンプル | raw documents、chunks、metadata、source fields | 権限ルール、document version、更新メモを追加 |
| 検索ログ | matched passages、scores、ranking | failure type statistics と before/after comparison を追加 |
| 回答引用 | 最終回答が支援ソースを表示 | citation faithfulness checks を追加 |
| 失敗ケース | 少なくとも 1 件の失敗を記録 | 3 件以上の原因、修正、regression check を追加 |
| 評価 | 固定質問と pass/fail rules | baseline、metrics、regression testing を追加 |
| デプロイメモ | 実行方法と必要な環境変数 | Docker、monitoring、fallback notes を追加 |

## 合格ライン

引用付き回答、検索ログ表示、empty retrieval 処理、評価ケース保存、少なくとも 1 件の失敗説明ができれば、この章は合格です。

最強のポートフォリオ版は、最大の版ではありません。別の開発者が実行を再現し、証拠を確認し、次の改善方針を理解できる版です。
