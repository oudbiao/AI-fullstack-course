---
title: "9.6.7 OpenAI Agents SDK【選択】"
description: "Agent、Tool、Runner などの高レベル抽象から出発して、OpenAI Agents SDK がなぜ統一された Agent プログラミングモデルに近いのかを理解する。"
sidebar:
  order: 35
head:
  - tag: meta
    attrs:
      name: keywords
      content: "OpenAI Agents SDK, agent runtime, tools, runner, sdk, agent abstraction"
---
:::tip[この節の位置づけ]
多くのフレームワークは、次のようなものを整理する手助けをしてくれます。

- グラフ
- チェーン
- 役割

一方で、OpenAI Agents SDK のような高レベル SDK は、むしろこう言っているようなものです。

> **Agent、Tool、実行時環境を 1 つの、より標準化された開発インターフェースにまとめよう。**

ここでのポイントは、必ずしも「最も柔軟」であることではなく、「より統一された Agent プログラミング体験」です。
:::
## 学習目標

- この種の Agents SDK が抽象化したい核心的なオブジェクトは何かを理解する
- なぜ「Runner / Runtime」がこの種の SDK の重要な価値になりやすいのかを理解する
- 最小限の高レベル抽象の例を読めるようになる
- いつこの種の SDK が向いていて、いつ必ずしも向いていないのかを判断できるようにする

---

## なぜ「Agents SDK」のような層が生まれたのか？

### 直接 Agent を手書きすると、すぐに大量の重複した定型処理が出てくるから

少しまともな Agent システムには、普通は次のようなものが必要になります。

- Tool の登録
- パラメータ検証
- 実行ループ
- 結果のラップ
- トレース
- 状態の進行

もし毎回プロジェクトごとに手書きしていたら、すぐに次のような問題が出ます。

- 構造が不統一になる
- 保守性が低い
- チームごとの書き方がバラバラになる

### SDK は本当に何をしたいのか？

SDK は、あなたの代わりに製品ロジックを作るのではありません。代わりに、次のような点を統一しようとします。

- Agent というオブジェクトをどう表現するか
- Tool をどう関連付けるか
- 1 回の実行をどう走らせるか

まずは、次の一文を覚えておくとよいです。

> **SDK の価値は「より強い」ことではなく、「より統一されている」ことです。**

---

## いちばん重要な抽象オブジェクト

### エージェント（Agent）

目標と Tool の集合を持つ、インテリジェントな実行単位です。

### ツール（Tool）

Agent が呼び出せる外部機能です。たとえば次のようなものがあります。

- 検索
- 計算
- ファイルアクセス

### ランナー / 実行時（Runner / Runtime）

ここは特に重要です。  
通常は次の役割を担います。

- Agent を実際に実行する
- 実行プロセスを管理する
- 結果を収集する

多くの場合、この種の SDK における最大のエンジニアリング上の価値は、まさにここにあります。

> **「Agent をどう動かすか」を統一していることです。**

---

## 最小限の高レベル抽象の例

以下では、純粋な Python でこの種の SDK の雰囲気を再現してみます。

```python
class Tool:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn

class Agent:
    def __init__(self, name, tools):
        self.name = name
        self.tools = {tool.name: tool for tool in tools}

class Runner:
    def run(self, agent, tool_name, **kwargs):
        if tool_name not in agent.tools:
            return {"error": "unknown_tool"}
        result = agent.tools[tool_name].fn(**kwargs)
        return {"agent": agent.name, "tool": tool_name, "result": result}

def get_weather(city):
    return f"{city} は現在、晴れで 22 度です"

weather_tool = Tool("get_weather", get_weather)
assistant = Agent("weather_assistant", [weather_tool])
runner = Runner()

print(runner.run(assistant, "get_weather", city="北京"))
```

想定出力：

```text
{'agent': 'weather_assistant', 'tool': 'get_weather', 'result': '北京 は現在、晴れで 22 度です'}
```

![OpenAI Agents SDK：Agent、Tool、Runner の分担](/img/course/ch09-openai-agents-sdk-runner-flow-ja.webp)

### このコードがなぜ「SDK っぽい」のか？

それは、次の 3 つを明確に分けているからです。

- Agent 本体
- Tool 本体
- 実行層 Runner

これは、多くの高レベル SDK が最も統一したい構造そのものです。

---

## この抽象化は具体的に何を省いてくれるのか？

### Tool 接続方法の統一

毎回プロジェクトごとに、次のようなものを定義し直す必要がありません。

- Tool をどう追加するか
- Tool をどう呼び出すか

### 実行入口の統一

システムが複雑になるほど、「誰が Agent を動かすのか」はとても重要な問題になります。  
Runner / Runtime は、この部分をより統一された形にしてくれます。

### チームの書き方を揃えやすい

なぜなら、次のような箇所が毎回バラバラに書かれなくなるからです。

- Agent の定義方法
- Tool の追加方法
- 結果の返し方

---

## なぜ Runner / Runtime が特に重要なのか？

### Agent は普通の関数ではないから

Agent は単なる

- 入力 -> 出力

ではありません。

通常は次のようなものも含みます。

- Tool の選択
- 実行プロセス
- 中間状態
- エラーの返却

そのため、「どう動かすか」自体が独立した層になります。

### 直感的な例え

Runner は次のように考えることができます。

> Agent の実行スケジューラ。

Agent は参加者、Runner はそれを実際に動かして、プロセス全体を管理する役割です。

---

## このような高レベル SDK はいつ特に便利か？

### 統一された開発体験がほしいとき

たとえば次のような場合です。

- 複数の Agent プロジェクトで同じ構造を使いたい
- チームで重複した実行ロジックを書きたくない
- Tool と Agent の表現をより統一したい

### 特に向いている場面

- 中小規模の Agent アプリ
- プロトタイプから製品化の中間段階
- 一貫した実行体験が必要なチーム開発

こうした場面では、高レベル抽象はとても手間を減らしてくれます。

---

## 限界もきちんと見ておく必要がある

### 高レベル抽象には、より多くの制約が伴う

得られるものは次の通りです。

- 統一性
- 分かりやすさ
- 定型コードの削減

一方で失う可能性があるのは、次のようなものです。

- かなり細かい低レベル制御の自由度

### システムが非常に特殊な場合

たとえば次のようなケースです。

- 非常に複雑な状態グラフがある
- とてもカスタムな実行戦略がある

この場合、高レベル SDK は最も快適な表現方法ではないかもしれません。

つまり、重要なのは「強いかどうか」ではなく、次の点です。

> **その抽象が自分のシステムに合っているかどうか。**

---

## 他のフレームワークとどう区別するか？

### LangGraph との違い

LangGraph はより次の方向に寄っています。

- グラフ
- 状態フロー
- 条件付きエッジ

Agents SDK はより次の方向に寄っています。

- Agent
- Tool
- Runner

### CrewAI との違い

CrewAI はより次の方向に寄っています。

- チームの役割と協調の表現

Agents SDK はより次の方向に寄っています。

- 統一された Agent 実行モデル

つまり、これはすべてのフレームワークと同じ土俵で正面競争しているというより、次のようなものに近いです。

> より高レベルな開発インターフェースのスタイル。

---

## 初心者がよくハマる落とし穴

### SDK 名だけを見て、抽象の境界を見ない

その結果、次のように感じやすくなります。

- 使っているうちに「しっくりこない」と思う

### 「高レベル抽象 = より上位で優れている」と思い込む

そうではありません。  
高レベルというのは、単に定型コードを減らしやすいという意味であって、いつも最適とは限りません。

### Agent 自体をまだ理解していないのに、先に SDK API を暗記する

これだと、呼び出しコードは書けても、アーキテクチャの判断ができるようになりにくいです。

---

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
問題の形：ワークフローグラフ、検索アプリ、役割チーム、または実験
フレームワーク選択：どの抽象化を追加し、何を隠すか
追跡記録：state、node、tool call、message、または run id
失敗確認：フレームワークの魔法が状態、再試行、または権限を隠す
判断: シングルエージェントのループが明確になってからフレームワークを選ぶ
```

## まとめ

この節で最も重要なのはクラス名を覚えることではなく、次を理解することです。

> **OpenAI Agents SDK のようなフレームワークの価値は、Agent、Tool、実行プロセスを、より安定した 1 つのプログラミングモデルにまとめることにある。**

「一貫した Agent 開発体験」がほしいときにはとても役立ちます。  
一方で、きわめて細かい低レベル制御が必要なときには、第一候補とは限りません。

---

## 練習

1. 自分の言葉で説明してみましょう：なぜ Runner / Runtime がこの種の SDK の重要な価値になりやすいのでしょうか？
2. 考えてみましょう：この高レベル SDK と CrewAI の「チーム協調の抽象化」にはどんな違いがありますか？
3. もし自分のシステムにすでに複雑な状態機械があるなら、このような高レベル SDK を最初に選びますか？ なぜですか？
4. 自分の言葉で説明してみましょう：SDK が本当に省いてくれるのは、どの種類の頻出する定型作業でしょうか？

<details>
<summary>解法と解説</summary>

1. Runner / Runtime が重要なのは、本番 Agent には prompt 以上のものが必要だからです。tool execution、state movement、handoff、error handling、tracing、loop を安定して回す方法が必要です。
2. 高レベル SDK は、Agent runtime を構築し観測しやすくすることに重心があります。CrewAI は role と task の team としてモデル化することに重心があります。何を明示したいかで選択が変わります。
3. すでに複雑な state machine があるなら、自動的に置き換えないほうがよいです。既存の transition、trace、failure policy と統合でき、重要な control を隠さないかを先に確認します。
4. SDK は tool registration、call execution、handoff wiring、structured outputs、tracing、共通 runtime 処理の boilerplate を減らせます。ただし task design と evaluation は残ります。

</details>
