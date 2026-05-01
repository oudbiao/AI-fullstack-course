---
title: "5.3 MCP アーキテクチャとコア概念"
sidebar_position: 26
description: "client-server 関係、ツールの公開、メッセージの流れ、transport まで含めて、MCP システムの内部がどのように動くのかを体系的に理解する。"
keywords: [MCP architecture, client, server, transport, tools, protocol flow]
---

# MCP アーキテクチャとコア概念

![MCP Host Client Server 架構圖](/img/course/mcp-host-client-server-ja.png)

:::tip この節の位置づけ
前の節で、MCP は「ツール接続層のための統一プロトコル」だと分かりました。  
この節ではさらに一歩進めて、次の問いに答えます。

> **MCP システムは構造的に、いったいどんな姿をしているのか？**

この章で大事なのは、抽象的な合言葉ではなく、次の点です。

- メッセージはどう流れるのか
- 誰が何を担当するのか
- システムはどうやって「ツールの発見」から「実際の実行」へ進むのか
:::

## 学習目標

- MCP システムにおける中心的な役割分担を理解する
- ツールの発見から呼び出しまでの一連の流れを理解する
- transport がアーキテクチャのどこに位置するかを理解する
- 「単一の API」ではなく「プロトコルの流れ」として理解できるようになる

---

## 一、まずは全体のアーキテクチャ図を見よう

```mermaid
flowchart LR
    A["Client"] --> B["Transport"]
    B --> C["MCP Server"]
    C --> D["Tools / Resources / Prompts"]

    C --> E["ローカルのツールロジック"]
    C --> F["外部サービス"]
    C --> G["ファイルシステム / データベース"]

    style A fill:#e3f2fd,stroke:#1565c0,color:#333
    style B fill:#fff3e0,stroke:#e65100,color:#333
    style C fill:#f3e5f5,stroke:#6a1b9a,color:#333
    style D fill:#e8f5e9,stroke:#2e7d32,color:#333
    style E fill:#fffde7,stroke:#f9a825,color:#333
    style F fill:#fffde7,stroke:#f9a825,color:#333
    style G fill:#fffde7,stroke:#f9a825,color:#333
```

この図で一番大事なのは、ノード名そのものではなく、次の点です。

> **Client は下層の世界を直接操作するのではなく、MCP Server という統一された入口を通して能力を得る。**

---

## 二、Client は実際に何をするのか？

Client の役割には、通常次のようなものがあります。

- 接続を確立する
- server がどんな能力を公開しているかを見つける
- 今のタスクに応じて呼び出すかどうかを判断する
- リクエストを送って結果を受け取る

Client は「利用する側」と考えると分かりやすいです。

実際のシステムでは、たとえば次のようなものが Client になります。

- IDE プラグイン
- チャットアシスタント
- デスクトップ Agent
- ワークフローエンジン

Client の一番大きな価値は、「自分で全部やること」ではなく、

> **いつ、server にどんな能力を求めるべきかを知っていること。**

---

## 三、Server は実際に何をするのか？

Server の役割には、通常次のようなものがあります。

- 能力を説明し、公開する
- client のリクエストを受け取る
- ローカルまたは外部のツールを呼び出す
- 構造化された結果を返す

言い換えると、server は「能力の提供側」です。

外部に対して次のように伝える役割があります。

- どんなツールがあるか
- それぞれのツールをどう呼ぶか
- どんなコンテキストオブジェクトをサポートしているか

つまり server は、プロトコルを実際に動かすための中心的な実体です。

---

## 四、なぜ transport を無視してはいけないのか？

初学者は次の 2 つだけに目が行きがちです。

- client
- server

でも、両者がやり取りできるようにしているのは transport です。

### 4.1 何を解決しているのか？

簡単に言うと、transport は次の問いに答えます。

> このプロトコルメッセージは、いったいどの経路でやり取りされるのか？

たとえば次のようなものです。

- ローカルプロセス間通信
- 標準入力/標準出力
- ネットワーク接続

### 4.2 transport が重要な理由

transport は次のような点に影響します。

- レイテンシ
- 信頼性
- デプロイ形態
- デバッグ方法

そのため transport は、単なる「おまけの選択肢」ではなく、アーキテクチャの一部です。

---

## 五、MCP システムでよく使われる 3 種類の能力

みんながよく「ツール」と呼びますが、もう少し正確に見ると、公開される内容は主に次の 3 種類と考えられます。

### 5.1 Tools

実行できる能力です。

たとえば：

- 検索
- ファイルの読み取り
- 天気の確認

### 5.2 Resources

「読み取れる情報源」に近いものです。

たとえば：

- ドキュメントの内容
- 設定データ
- データテーブルのスナップショット

### 5.3 Prompts

「再利用できるプロンプトテンプレート」に近いものです。

この 3 つは完全に同じではありませんが、どれも「外部に公開する利用可能な能力」という点では共通しています。

---

## 六、一連のメッセージの流れはどうなっているのか？

### 6.1 まずツールを発見する

```python
list_request = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {}
}

list_response = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
        "tools": [
            {"name": "search_docs", "description": "コース文書を検索する"},
            {"name": "get_weather", "description": "天気を確認する"}
        ]
    }
}

print(list_request)
print(list_response)
```

### 6.2 次にツールを呼び出す

```python
call_request = {
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
        "name": "search_docs",
        "arguments": {"query": "返金ポリシー"}
    }
}

call_response = {
    "jsonrpc": "2.0",
    "id": 2,
    "result": {
        "content": [{"type": "text", "text": "コース購入後 7 日以内で、学習進捗が 20% 未満なら返金可能です。"}]
    }
}

print(call_request)
print(call_response)
```

### 6.3 この 2 ステップで何が分かるのか？

MCP は単に「関数を 1 つ呼ぶ」仕組みではなく、まず次の 2 段階があることを示しています。

1. 能力の発見
2. 能力の呼び出し

これにより、client はすべてのツールの詳細を最初から固定で知っている必要がありません。

![MCP 工具發現與呼叫訊息流圖](/img/course/ch09-mcp-host-client-server-message-flow-map-ja.png)

:::tip 図の読み方
この図はメッセージの順番で見てください。Host 内の Client がまず Server に tools/list を送り、能力一覧を受け取ったあとに tools/call を実行します。MCP の価値は、「能力を発見すること」と「能力を呼び出すこと」を統一されたプロトコルでつなぐ点にあります。
:::

---

## 七、なぜ MCP は「デカップリング層」なのか？

### 7.1 MCP がない場合

Client は通常、次のような細部まで直接知る必要があります。

- ツール名の付け方
- パラメータの書き方
- 戻り値の形

これでは client と tool provider が強く結びつきすぎてしまいます。

### 7.2 MCP がある場合

Client が主に依存するのは、次のような共通の仕組みです。

- 統一プロトコル
- 統一された発見方法
- 統一された呼び出し方法

これによって、システムは次のような分かりやすい階層になります。

- 上層はタスクのオーケストレーションを担当する
- 下層は能力の提供を担当する

したがって MCP は、次のように理解できます。

> **ツールエコシステムにおけるアダプタ層であり、デカップリング層。**

---

## 八、最小構成のアーキテクチャをシミュレートする

以下では、純粋な Python で極めて簡単な MCP 風のやり取りをシミュレーションします。

```python
class MockMCPServer:
    def __init__(self):
        self.tools = {
            "search_docs": lambda query: f"検索結果: {query}"
        }

    def list_tools(self):
        return [{"name": name} for name in self.tools]

    def call_tool(self, name, arguments):
        if name not in self.tools:
            return {"error": "unknown_tool"}
        return {"result": self.tools[name](**arguments)}

class MockMCPClient:
    def __init__(self, server):
        self.server = server

    def discover(self):
        return self.server.list_tools()

    def call(self, name, arguments):
        return self.server.call_tool(name, arguments)

server = MockMCPServer()
client = MockMCPClient(server)

print(client.discover())
print(client.call("search_docs", {"query": "返金ポリシー"}))
```

### 8.2 この例は小さいですが、学習価値はとても高いです

なぜなら、すでに次の 3 層の分担が見えるからです。

- client はリクエストを担当する
- server は能力の公開を担当する
- tool は具体的な実行を担当する

この 3 層の役割分担をしっかり理解できれば、あとでより実際的な MCP システムを見たときも、かなり安定して理解できます。

---

## 九、よくあるアーキテクチャ上の誤解

### 9.1 server をツールそのものだと思ってしまう

server はツールではありません。  
server は次のものです。

> ツールのためのプロトコル化された出口。

### 9.2 transport はなくてもよいと思ってしまう

transport はデプロイ方法や安定性に直接影響します。

### 9.3 MCP が権限やポリシーの問題を自動で解決してくれると思ってしまう

そうではありません。  
MCP が解決するのは「統一接続」であって、「自動ガバナンス」ではありません。

---

## まとめ

この節で一番大切なのは、client/server という単語を覚えることではなく、次を理解することです。

> **MCP アーキテクチャの核心は、能力の提供側と能力の利用側が、統一されたメッセージの流れと統一された境界を通じて関係を結ぶことにある。**

この流れがはっきり見えていれば、次に server 開発、client 統合、エコシステム実践を学ぶときも、迷いにくくなります。

---

## 練習

1. 自分の言葉で説明してみましょう。なぜ client と server の役割は分ける必要があるのか？
2. 考えてみましょう。transport が変わっても、上位の呼び出しロジックはできるだけ変えない方がよいのはなぜか？
3. `MockMCPServer` に `get_weather` ツールを追加してみましょう。
4. 自分の言葉で説明してみましょう。なぜ MCP は「ツールそのもの」ではなく、「デカップリング層」だと言えるのか？
