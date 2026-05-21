---
title: "8.5.1 プロジェクトロードマップ：引用付き知識助手を作る"
sidebar_position: 0
description: "第 8 章キャップストーンの短い実践ロードマップ：引用付き RAG または LLM アプリを作り、検索ログ、失敗処理、評価、デプロイメモを残す。"
keywords: [LLM project guide, enterprise knowledge base, intelligent assistant, RAG project, SOP document assistant]
---

# 8.5.1 プロジェクトロードマップ：引用付き知識助手を作る

このキャップストーンは、知識、モデル呼び出し、アプリケーションフロー、engineering evidence を 1 つの再現可能な LLM アプリに接続できることを示します。

## まずプロジェクト証拠ループを見る

![LLM アプリケーション総合プロジェクトのロードマップ](/img/course/ch08-projects-route-map-ja.webp)

![LLM アプリケーションプロジェクトの学習順序図](/img/course/ch08-project-learning-order-map-ja.webp)

![LLM アプリケーションプロジェクトのデリバリーループ図](/img/course/ch08-project-delivery-loop-ja.webp)

プロジェクトは「ベクトルデータベースをつなぐ」だけではありません。文書、チャンク、検索、文脈、回答、引用、ログ、評価、改善をつなぐ追跡可能なループです。

## プロジェクト準備チェックを動かす

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

期待される出力：

```text
ready: True
project_type: knowledge-base assistant
evidence: docs, eval, citations, failures
```

`ready` が `False` なら、別の機能を足す前に evidence loop を完成させます。

## この順番で学ぶ

| 手順 | プロジェクト | 本当に鍛える力 |
|---|---|---|
| 1 | 企業またはコース知識ベース | 検索、権限、引用、追跡可能な回答 |
| 2 | 知的アシスタント | 検索、セッション状態、ツール呼び出しをプロダクト機能にする |
| 3 | RAG + 微調整システム | 知識不足と振る舞い不安定を分ける |
| 4 | SOP 文書アシスタント | 文書解析、構造化出力、テンプレート描画 |
| 5 | フル実践ワークショップ | 実 API や DB を足す前の最小再現ループ |

ガイド付き baseline が必要なら、[8.5.6 実践：第 8 章 RAG アプリ完全ワークショップ](./05-stage-hands-on-workshop.md) から始めます。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
プロジェクト目標：ユーザーのタスクとビジネス境界
ベースライン: まずは最も簡単なプロンプト/RAG/app版
評価：固定ケース、検索証拠、回答品質、引用チェック
失敗ログ: 少なくとも1件の失敗ケースと原因の可能性
成果物：README、実行コマンド、スクリーンショット/ログ、次の一手
```

## プロジェクト成果物基準

| 成果物 | 最低要件 | 強いポートフォリオ版 |
|---|---|---|
| README | 目的、実行コマンド、依存関係、例 | アーキテクチャ図、設計上のトレードオフ、コスト、振り返りを追加 |
| 知識ベースサンプル | 生文書、チャンク、メタデータ、出典フィールド | 権限ルール、文書バージョン、更新メモを追加 |
| 検索ログ | 命中箇所、スコア、順位 | 失敗タイプ統計と前後比較を追加 |
| 回答引用 | 最終回答が根拠ソースを表示 | 引用の忠実性チェックを追加 |
| 失敗ケース | 少なくとも 1 件の失敗を記録 | 3 件以上の原因、修正、regression check を追加 |
| 評価 | 固定質問と合否ルール | ベースライン、メトリクス、回帰テストを追加 |
| デプロイメモ | 実行方法と必要な環境変数 | Docker、監視、フォールバックメモを追加 |

## 合格ライン

引用付き回答、検索ログ表示、empty retrieval 処理、評価ケース保存、少なくとも 1 件の失敗説明ができれば、この章は合格です。

最強のポートフォリオ版は、最大の版ではありません。別の開発者が実行を再現し、証拠を確認し、次の改善方針を理解できる版です。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、query から chunks、retrieval scores、引用 evidence、answer、fallback behavior までの流れを追跡します。
2. 証拠には、retrieved passages、source metadata、引用付き回答、空振りまたは誤検索の例を含めます。
3. 失敗原因が chunking、retrieval、ranking、prompt assembly、source 不足、根拠のない生成のどれかを説明できればよいです。

</details>
