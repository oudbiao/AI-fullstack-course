---
title: "9.5.1 MCP ロードマップ：サーバー、クライアント、機能"
description: "MCP の短い実践ロードマップ：プロトコル層、サーバー／クライアントの責務、ツール、リソース、プロンプト、安全なエコシステム統合を理解する。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "MCP ガイド, Model Context Protocol, Agent ツールエコシステム, MCP Server"
---
MCP は、ツール、リソース、プロンプトテンプレートをより標準的にモデルアプリケーションへ接続するプロトコル層です。Agent やツールを置き換えるものではなく、機能を一貫して公開・利用しやすくします。

## まず MCP の境界を見る

![MCP Host Client Server アーキテクチャ図](/img/course/mcp-host-client-server-ja.webp)

![MCP 章の学習順序図](/img/course/ch09-mcp-chapter-flow-ja.webp)

![MCP 機能アクセス橋渡し図](/img/course/ch09-mcp-capability-bridge-ja.webp)

Function Calling は構造化された呼び出しに注目します。MCP は外部機能がプロトコルを通じて発見、記述、呼び出し、管理される方法に注目します。

## 機能登録チェックを動かす

本物の MCP サーバーを実装する前に、何を公開し、クライアントが何を呼び出せるか列挙します。

```python
server = {
    "tools": ["search_docs"],
    "resources": ["course://ch09-agent"],
    "prompts": ["study_plan"],
}

client_request = "search_docs"

print("server_ready:", all(server.values()))
print("can_call:", client_request in server["tools"])
print("boundary:", "server exposes, client calls")
```

期待される出力：

```text
server_ready: True
can_call: True
boundary: server exposes, client calls
```

境界が曖昧だと、権限とデバッグも曖昧になります。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | MCP の概念 | プロトコル層が統合の混乱を減らす理由を説明する |
| 2 | MCP アーキテクチャ | Host、Client、Server、tools、resources、prompts を区別する |
| 3 | サーバー開発 | 1 つの機能を明確な入力、出力、エラーで包む |
| 4 | クライアント統合 | サーバー機能を安全に発見し、呼び出す |
| 5 | エコシステム | MCP を IDE、データベース、ブラウザ、知識ベース、Agent につなげる |
| 6 | MCP、A2A、protocol layer | Tool discovery、peer-agent handoff、authorization、trace を分けた capability card を書く |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
機能：サーバーが公開するリソース、Prompt、またはツール
契約: スキーマ、通信方式、権限、エラー形式
呼び出しトレース：探索、呼び出し、応答、失敗時の処理
protocol card: capability server、peer agent、authorization rule、handoff artifacts
失敗確認：互換性のないスキーマ、認証不足、安全でないツール、またはサーバーエラー
統合アクション：自律化を追加する前にサーバー契約を確認する
```

## 合格ライン

Host-Client-Server 関係を描き、サーバーが何を公開し、クライアントが何を呼び出し、権限がどこで確認されるか説明できれば、この章は合格です。

出口ミニプロジェクトは「course-materials MCP サーバー設計」です：1 つの検索ツール、1 つのリソース URI パターン、1 つのプロンプトテンプレート、1 つの失敗処理ルールを含めます。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、agent loop を goal、plan、tool call、observation、memory/state update、stop condition として説明します。
2. 証拠には、最終回答だけでなく、別の開発者が確認できる trace を残します。
3. tool schema、permission boundary、retry、evaluation case、人間レビューなど、安全性または信頼性の制御を1つ説明できれば十分です。

</details>
