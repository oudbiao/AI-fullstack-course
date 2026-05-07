---
title: "9.5.5 MCP Client の統合"
sidebar_position: 28
description: "ツールの発見、呼び出しの振り分け、エラー処理から最小限の client 実装までを通して、クライアントが MCP Server の公開する能力を実際にどう利用するかを理解します。"
keywords: [MCP client, tool discovery, client integration, dispatch, protocol client]
---

# 9.5.5 MCP Client の統合

:::tip この節の位置づけ
ここまでで、server から見た MCP はすでに学びました。  
この節では視点を変えて、client から見てみます。

> **クライアントはどうやって MCP Server の能力を発見し、選び、呼び出すのでしょうか？**

ここはとても重要です。なぜなら、実際にツールを使うのは server ではなく client だからです。
:::

## 学習目標

- MCP Client の基本的な役割を理解する
- 「ツールを発見する」と「ツールを呼び出す」を 2 つのステップとして考えられるようになる
- 最小構成の MCP Client の呼び出しフローを読めるようになる
- client 側でも、選択戦略・失敗時の処理・キャッシュが必要な理由を理解する

---

## Client と Server の役割分担はどうなっているのか？

### Server は能力を提供する

Server は「ツール置き場の管理者」のようなものです。主な役割は：

- ツール一覧を返す
- 能力を公開する
- 呼び出しを実行する

### Client は能力を利用する

Client は「実際に用事をしに来る人」のようなものです。主な役割は：

- ツールを発見する
- どのツールを呼ぶか決める
- 引数をまとめる
- 結果を受け取る

つまり、とても大事な点は次の通りです。

> **MCP Client は受け身の中継役ではなく、通常は呼び出しの判断ロジックも持っています。**

---

## Client が最初に学ぶべきことは？ まずツールを発見すること

### なぜ固定で書いてはいけないのか？

もし client が最初からツールを全部固定で書いてしまうと、次の問題が起きます。

- server のツールが変わるたびにコード修正が必要
- 別の server に変えるたびに書き直しになる

これは MCP が解決したい問題と逆です。

### 最小限の発見の例

```python
class MockMCPServer:
    def list_tools(self):
        return [
            {"name": "search_docs", "description": "コース文書を検索する"},
            {"name": "get_weather", "description": "天気を調べる"}
        ]

server = MockMCPServer()
tools = server.list_tools()

for tool in tools:
    print(tool)
```

### ここで学ぶことは？

このコードが教えているのは次のことです。

> client はまず「何が使えるのか」を知ってから、「どう使うか」を考える。

これが発見フェーズの価値です。

---

## 発見したあと、client は何をするのか？

### ツールを選ぶ

すべてのツールを呼ぶわけではありません。  
client は通常、まず次を判断します。

- 今の問題にツールが必要か
- 必要なら、どのツールを呼ぶか

### 引数を組み立てる

正しいツールを選んでも、引数を正しく組み立てる必要があります。

### エラーを処理する

もし次のようなことが起きたら：

- server がタイムアウトする
- ツールが存在しない
- 引数の検証に失敗する

client はただ落ちるのではなく、次を判断しなければなりません。

- 再試行するか
- フォールバックするか
- 別のツールに切り替えるか

---

## 最小限の Client の例

### 実行可能なコード

```python
class MockMCPServer:
    def list_tools(self):
        return [
            {"name": "search_docs", "description": "コース文書を検索する"},
            {"name": "get_weather", "description": "天気を調べる"}
        ]

    def call_tool(self, name, arguments):
        if name == "search_docs":
            return {"result": f"検索結果: {arguments['query']}"}
        if name == "get_weather":
            return {"result": f"{arguments['city']} は現在晴れ、22 度です"}
        return {"error": "unknown_tool"}

class MockMCPClient:
    def __init__(self, server):
        self.server = server
        self.tools = []

    def discover(self):
        self.tools = self.server.list_tools()
        return self.tools

    def call(self, name, arguments):
        return self.server.call_tool(name, arguments)

server = MockMCPServer()
client = MockMCPClient(server)

print(client.discover())
print(client.call("search_docs", {"query": "返金ポリシー"}))
```

### このコードは何を示しているのか？

このコードには、client の 2 つの基本機能がすでに表れています。

1. 発見
2. 呼び出し

これが MCP Client の最小ループです。

---

## Client には「戦略層」もある

### なぜ client は単なるプロトコル呼び出し器ではないのか？

実際のシステムでは、client は次のことも判断する必要があるからです。

- 今の問題に MCP を使うべきか
- 使うなら、どの server / どのツールを優先するか
- 失敗したらどう戻るか

### 簡単なツール選択器

```python
def choose_tool(user_query, tools):
    tool_names = [t["name"] for t in tools]

    if "返金" in user_query and "search_docs" in tool_names:
        return {"name": "search_docs", "arguments": {"query": "返金ポリシー"}}

    if "天気" in user_query and "get_weather" in tool_names:
        return {"name": "get_weather", "arguments": {"city": "北京"}}

    return None

tools = client.discover()
decision = choose_tool("返金ポリシーとは？", tools)
print(decision)
print(client.call(decision["name"], decision["arguments"]))
```

これは、client が軽い振り分け役も担うことを示しています。

---

## なぜ client にとってエラー処理が特に重要なのか？

### client は「失敗を最初に感じ取る側」だから

server 側では次のようなエラーが返ることがあります。

- unknown_tool
- invalid_arguments
- timeout

そのあとどうするかを決めるのは client です。

### 最小限のエラー処理の例

```python
def safe_call(client, name, arguments):
    result = client.call(name, arguments)
    if "error" in result:
        return {"ok": False, "fallback": "現在このツールは使えません。しばらくしてから再試行してください。"}
    return {"ok": True, "data": result["result"]}

print(safe_call(client, "search_docs", {"query": "返金ポリシー"}))
print(safe_call(client, "bad_tool", {}))
```

この一歩で、システムは次の状態から：

- 「1 回エラーが出たらすぐ落ちる」

次の状態へ変わります。

- 「1 回エラーが出ても、うまく受け止められる」

---

## なぜ client にもキャッシュが必要なことがあるのか？

### とても現実的な問題

毎回 `list_tools()` を呼び直すと、無駄が増えないでしょうか？

多くの場合：

- ツール一覧はそれほど頻繁には変わらない
- 毎回の再発見は遅延を増やす

### 最小限のキャッシュの考え方

```python
class CachedMCPClient(MockMCPClient):
    def discover_once(self):
        if not self.tools:
            self.tools = self.server.list_tools()
        return self.tools

cached_client = CachedMCPClient(server)
print(cached_client.discover_once())
print(cached_client.discover_once())
```

これはとても単純ですが、次のことを表しています。

> client は単なる「中継器」ではなく、状態を持ち、最適化できる。

---

## Client 統合でよくある落とし穴

### 呼び出しはできるが、選べない

client に選択戦略がないと、次のようになりやすいです。

- ツールはあるのに、うまく使えない

### 成功だけ見て、失敗パスを見ない

server でエラーが起きると、システムの体験が急に悪くなります。

### 毎回ツールを再発見する

不要なコストがかなり増えることがあります。

---

## まとめ

この節で大事なのは、server を呼べるクラスを作ること自体ではありません。大切なのは次の理解です。

> **MCP Client の核心は、単に「リクエストを送る」ことではなく、「ツールの発見、ツールの選択、引数の組み立て、結果の処理」を 1 つの安定した利用層としてまとめることです。**

client が成熟すればするほど、server 側の能力は上位システムで実際に活用されやすくなります。

---

## 練習

1. `MockMCPServer` に `read_file` ツールを 1 つ追加し、client の選択ロジックを拡張してみましょう。
2. 考えてみましょう：なぜ毎回ツールを再発見するのに向いているシステムがある一方で、キャッシュに向いているシステムもあるのでしょうか？
3. `safe_call()` に「失敗したら 1 回だけ再試行する」ロジックを追加してみましょう。
4. 自分の言葉で説明してみましょう：なぜ MCP Client には通常「戦略層」が必要なのでしょうか？
