---
title: "9.5.4 MCP サーバー開発"
description: "ツールの説明、パラメータ検証、結果の返却から最小のサーバー構成まで、MCP サーバーがどう能力を公開すべきかを理解します。"
sidebar:
  order: 27
head:
  - tag: meta
    attrs:
      name: keywords
      content: "MCP server, tool server, schema, tool exposure, server development"
---
:::tip[この節の位置づけ]
前の2節で、すでに次のことが分かりました。

- MCP が何を解決するのか
- MCP のアーキテクチャでクライアントとサーバーがそれぞれ何を担当するのか

この節からは、いよいよサーバーの視点で本格的に進めていきます。次の問いに答えます。

> **もし自分で MCP サーバーを書くなら、まずどこから始めればよいのか？**
:::
## 学習目標

- MCP サーバーの最小限の責務の範囲を理解する
- ツールの説明、パラメータ構造、呼び出し入口の定義を学ぶ
- なぜサーバー開発の重点が「能力を公開すること」であって、「業務ロジックをベタ書きすること」ではないのかを理解する
- 最小で動く Mock MCP Server を読めるようになる

---

## MCP サーバーは実際に何をしているのか？

### 「もうひとつの普通のバックエンド」ではない

普通のバックエンドは、たいてい業務 API に直接向き合います。
それに対して MCP サーバーは、次のようなものです。

> **すでにある能力を、クライアントが見つけて呼び出せるツールの集まりとして整理するもの。**

そのため、主な注目点はだいたい次の4つです。

- どんなツールがあるか
- ツールをどう説明するか
- パラメータをどう検証するか
- 結果をどう統一して返すか

### 直感的なたとえ

MCP サーバーは、受付のある道具箱の管理人のようなものです。

- クライアントが「ここにはどんな道具がありますか」と聞く
- サーバーが能力一覧を出す
- クライアントが「どれを使えばいいですか」と選ぶ
- サーバーが約束どおり実行して結果を返す

これは「業務関数をバラバラにそのまま置く」やり方とはかなり違います。

---

## まず最小のツールを定義する

### ツールには最低でも何が必要？

少なくとも次のものが必要です。

- 名前
- 説明
- パラメータの説明
- 実際の実行ロジック

### 最小のツール説明の例

```python
search_docs_tool = {
    "name": "search_docs",
    "description": "コース文書を検索して関連内容を返す",
    "parameters": {
        "query": {
            "type": "string",
            "description": "検索したいキーワード"
        }
    },
    "required": ["query"]
}

print(search_docs_tool)
```

想定出力：

```text
{'name': 'search_docs', 'description': 'コース文書を検索して関連内容を返す', 'parameters': {'query': {'type': 'string', 'description': '検索したいキーワード'}}, 'required': ['query']}
```

この構造は、次のように考えると分かりやすいです。

> ツールの対外向け説明書。

---

## ツール説明を適当に書いてはいけない理由

### よくない説明

```python
bad_tool = {
    "name": "search",
    "description": "検索する",
    "parameters": {"q": {"type": "string"}}
}

print(bad_tool)
```

想定出力：

```text
{'name': 'search', 'description': '検索する', 'parameters': {'q': {'type': 'string'}}}
```

問題点は次のとおりです。

- 名前があいまい
- 説明が短すぎる
- パラメータの意味が分かりにくい

### もっと安定した説明

```python
good_tool = {
    "name": "search_course_docs",
    "description": "コース FAQ、ポリシー、学習ロードマップの文書を検索する",
    "parameters": {
        "query": {
            "type": "string",
            "description": "ユーザーが調べたいトピック。たとえば 返金ポリシー や 証明書"
        }
    },
    "required": ["query"]
}

print(good_tool)
```

想定出力：

```text
{'name': 'search_course_docs', 'description': 'コース FAQ、ポリシー、学習ロードマップの文書を検索する', 'parameters': {'query': {'type': 'string', 'description': 'ユーザーが調べたいトピック。たとえば 返金ポリシー や 証明書'}}, 'required': ['query']}
```

こちらのほうがよい理由は次のとおりです。

- ツールの境界がはっきりしている
- パラメータの意味がはっきりしている
- クライアントが正しく使いやすい

---

## サーバーの最小2機能：ツール一覧の表示 + ツールの実行

最小限の MCP サーバーなら、少なくとも次の2つが必要です。

1. 利用可能なツールを一覧表示する
2. あるツールの呼び出しを受け付ける

### まずは最小サーバーを書く

```python
class MockMCPServer:
    def __init__(self):
        self.tool_specs = [
            {
                "name": "search_docs",
                "description": "コース文書を検索する",
                "parameters": {
                    "query": {"type": "string"}
                }
            }
        ]

    def list_tools(self):
        return self.tool_specs

server = MockMCPServer()
print(server.list_tools())
```

想定出力：

```text
[{'name': 'search_docs', 'description': 'コース文書を検索する', 'parameters': {'query': {'type': 'string'}}}]
```

### 次に実際の実行ロジックを追加する

```python
class MockMCPServer:
    def __init__(self):
        self.kb = {
            "返金": "コース購入後 7 日以内で、学習進捗が 20% 未満なら返金できます。",
            "証明書": "すべてのプロジェクトを完了し、テストに合格すると証明書を取得できます。"
        }

        self.tool_specs = [
            {
                "name": "search_docs",
                "description": "コース文書を検索する",
                "parameters": {
                    "query": {"type": "string"}
                }
            }
        ]

    def list_tools(self):
        return self.tool_specs

    def call_tool(self, name, arguments):
        if name != "search_docs":
            return {"error": "unknown_tool"}

        query = arguments.get("query", "")
        for key, value in self.kb.items():
            if key in query:
                return {"result": value}
        return {"result": "関連する文書が見つかりませんでした"}

server = MockMCPServer()
print(server.call_tool("search_docs", {"query": "返金ポリシーは何ですか"}))
```

想定出力：

```text
{'result': 'コース購入後 7 日以内で、学習進捗が 20% 未満なら返金できます。'}
```

ここまでで、かなり分かりやすい最小のサーバー骨組みになっています。

---

## なぜパラメータ検証はサーバーの責任なのか？

### クライアントやモデルは、間違ったパラメータを渡すことがあるから

たとえば次のようなものです。

```python
bad_call = {"query_text": "返金ポリシー"}
```

もしサーバーがそのまま実行すると、エラーになったり、変な動作になったりします。

### 最小の検証例

```python
def validate_search_docs(arguments):
    if "query" not in arguments:
        return False, "missing_query"
    if not isinstance(arguments["query"], str):
        return False, "query_must_be_string"
    return True, "ok"

print(validate_search_docs({"query": "返金ポリシー"}))
print(validate_search_docs({"query_text": "返金ポリシー"}))
```

想定出力：

```text
(True, 'ok')
(False, 'missing_query')
```

### この手順を省いてはいけない理由

サーバーは、能力の境界を守る門番だからです。
サーバーが検証しないと、ツール全体の安定性を保つのが難しくなります。

![MCP サーバーのツール契約図](/img/course/ch09-mcp-server-tool-contract-map-ja.webp)

:::tip[図の見方]
MCP サーバーは、ツール契約の門番だと考えてください。list_tools を公開するだけでなく、call_tool のパラメータを検証し、実際のロジックを実行し、結果を統一して返し、エラーをクライアントが理解できる形に変える役割もあります。
:::
---

## より完成度の高い最小サーバー版

```python
class BetterMCPServer:
    def __init__(self):
        self.kb = {
            "返金": "コース購入後 7 日以内で、学習進捗が 20% 未満なら返金できます。",
            "証明書": "すべてのプロジェクトを完了し、テストに合格すると証明書を取得できます。"
        }

    def list_tools(self):
        return [
            {
                "name": "search_docs",
                "description": "コース文書を検索する",
                "parameters": {
                    "query": {"type": "string"}
                }
            }
        ]

    def validate(self, name, arguments):
        if name != "search_docs":
            return False, "unknown_tool"
        if "query" not in arguments:
            return False, "missing_query"
        if not isinstance(arguments["query"], str):
            return False, "query_must_be_string"
        return True, "ok"

    def call_tool(self, name, arguments):
        ok, msg = self.validate(name, arguments)
        if not ok:
            return {"error": msg}

        query = arguments["query"]
        for key, value in self.kb.items():
            if key in query:
                return {"result": value}
        return {"result": "関連する文書が見つかりませんでした"}

server = BetterMCPServer()
print(server.list_tools())
print(server.call_tool("search_docs", {"query": "証明書はどうやって取得しますか"}))
print(server.call_tool("search_docs", {"wrong": "証明書はどうやって取得しますか"}))
```

想定出力：

```text
[{'name': 'search_docs', 'description': 'コース文書を検索する', 'parameters': {'query': {'type': 'string'}}}]
{'result': 'すべてのプロジェクトを完了し、テストに合格すると証明書を取得できます。'}
{'error': 'missing_query'}
```

![MCP サーバーの検証と返却結果図](/img/course/ch09-mcp-server-validation-result-map-ja.webp)

### この版が前の版より優れている点

次の機能がすでにそろっています。

- ツールの一覧表示
- パラメータ検証
- 統一された呼び出し入口
- 統一されたエラー返却

これは、実際の開発におけるサーバーの核心的な責務にかなり近いです。

---

## MCP サーバー開発でよくある落とし穴

### 業務ロジックとプロトコルロジックを混ぜてしまう

その結果、次のようになりがちです。

- ツールの説明が分かりにくい
- 拡張しにくい
- デバッグしにくい

### ツールの粒度が粗すぎる、または細かすぎる

- 粗すぎる: 1 つのツールが何でもやる
- 細かすぎる: クライアント側の呼び出しが複雑になりすぎる

### 返却形式が統一されていない

あるときは文字列、あるときは dict、あるときはそのまま例外を投げる、という形だと、後でつなぎにくくなります。

---

## MCP サーバーの設計が十分よいかどうかをどう判断するか？

まず、次の4つを確認してみましょう。

1. クライアントが利用できるツールを明確に把握できるか
2. パラメータの条件が明確か
3. エラー返却が統一されているか
4. 新しいツールを追加したときに構造がどんどん崩れないか

この4つにしっかり答えられれば、サーバーの設計はかなり良いと言えます。

---

## 小結

この節で大事なのは、「クラスを 1 つ書くこと」ではなく、次の考え方を理解することです。

> **MCP サーバーの本質は、実行可能な能力を、見つけやすく、検証しやすく、呼び出しやすい形で公開することにある。**

サーバーが分かりやすいほど、クライアント側は拡張しやすくなり、ツールのエコシステム全体も育てやすくなります。

---

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
機能：サーバーが公開するリソース、Prompt、またはツール
契約: スキーマ、通信方式、権限、エラー形式
呼び出しトレース：探索、呼び出し、応答、失敗時の処理
失敗確認：互換性のないスキーマ、認証不足、安全でないツール、またはサーバーエラー
統合アクション：自律化を追加する前にサーバー契約を確認する
```

## 練習

1. `BetterMCPServer` に `get_weather(city)` ツールを追加してください。
2. この新しいツールに対して、パラメータ検証ロジックを追加してください。
3. 考えてみましょう: ツールの粒度が粗すぎる場合と細かすぎる場合、それぞれどんな問題が起きるでしょうか？
4. 自分の言葉で説明してください。なぜ MCP サーバー開発の核心は「ツールを実行すること」だけでなく、「明確な境界を公開すること」でもあるのでしょうか？

<details>
<summary>参考実装と解説</summary>

1. `get_weather(city)` を説明と schema つきの tool として登録し、`{city, condition, source}` のような小さな構造化結果を返します。本文で実 API に接続していない限り、プレースホルダーのデータで十分です。
2. `city` が存在するか、空でない文字列か、異常に大きな入力でないかを検証します。不正入力では黙って推測せず、明確なエラーを返します。
3. tool が粗すぎると重要な選択が隠れ、エラー箇所も特定しにくくなります。細かすぎると Agent が小さな呼び出しを大量に計画する必要があり、遅延、ルーティングミス、文脈ノイズが増えます。
4. MCP Server 開発の核心は境界を公開することです。tool が何をするか、どんな入力を受けるか、どんな出力とエラーを返すか、どんな権限が必要かを示す必要があります。実行は契約の一部にすぎません。

</details>
