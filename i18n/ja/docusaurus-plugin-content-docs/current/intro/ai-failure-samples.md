---
sidebar_position: 15
title: "AI アプリ失敗サンプル集"
description: "LLM API、Prompt、RAG、Agent、安全、デプロイの問題を記録する短い失敗サンプル索引です。"
keywords: [AI失敗サンプル, RAGトラブル対応, Agentトラブル対応, Promptトラブル対応, LLMアプリ]
---

# AI アプリ失敗サンプル集

![AI プロジェクト速習トラブル索引図](/img/course/appendix-quick-ref-debug-index-map-ja.png)

失敗サンプルとは、実際の入力に対してシステムが期待どおりに動かなかった記録です。調査に役立ち、同じ問題の再発を防ぎます。

## 失敗レイヤー

| 層 | よくある症状 | まず確認すること |
| --- | --- | --- |
| LLM API | timeout、rate limit、空応答、高コスト | request_id、生レスポンス、tokens、遅延 |
| Prompt/schema | JSON 不正、フィールド不足、ラベルずれ | schema、例、パーサ、固定テスト |
| RAG | 出典違い、引用が弱い、文書を拾えない | 検索チャンク、metadata、citation_ok |
| Agent/tool | ツール選択ミス、引数ミス、ループ、trace 不足 | ツール schema、最大ステップ、action/observation |
| 安全 | 権限超過、機密ログ、不安全な操作 | allowlist、人間確認、監査ログ |
| デプロイ | 自分の環境でしか動かない、秘密情報問題、実行不安定 | `.env.example`、依存バージョン、起動ログ |

## 失敗サンプルテンプレート

```md
## 失敗サンプル

ユーザー入力：
期待：
実際：
層：
関連ログ：
考えられる原因：
修正：
回帰テスト：
解決済み：
```

各ポートフォリオプロジェクトで、最低 3 件の失敗サンプルを残します。よいプロジェクトは失敗を隠すのではなく、どう見つけて修正したかを示します。
