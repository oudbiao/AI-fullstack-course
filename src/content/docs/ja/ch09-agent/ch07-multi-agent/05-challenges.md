---
title: "9.7.6 マルチ Agent の課題と解決"
description: "重複作業、通信のゆがみ、衝突の収束、コスト爆発から可観測性まで、マルチ Agent の実運用で最もよくある難しさを体系的に理解します。"
sidebar:
  order: 42
head:
  - tag: meta
    attrs:
      name: keywords
      content: "multi-agent, failure modes, coordination, observability, cost, conflict resolution"
---
:::tip[本節の位置づけ]
前の節までで、マルチ Agent は役割分担、通信、協調ができることを見てきました。  
でも、実際にシステムを作っていくと、ある現実に気づきます。

> **マルチ Agent の難しさは、「何体も Agent を動かせるか」ではなく、「システムがいつ制御不能になるか」にある。**

この節では、まさにその「制御不能になるポイント」を中心に説明します。
:::
## 学習目標

- マルチ Agent システムでよくある失敗パターンを理解する
- 問題を通信、協調、コスト、品質に分けて考えられるようにする
- 最小限の衝突解決と重複排除の例を読めるようにする
- マルチ Agent の重要ポイントは「より賢くすること」より「より制御しやすくすること」だと理解する

---

## なぜマルチ Agent システムは問題が起きやすいのか？

### 単体 Agent でよくある問題

単体 Agent でよくある問題は、たいてい次のようなものです。

- 推論を間違える
- ツールの選択を間違える
- 出力が安定しない

### マルチ Agent では、さらにシステムの複雑さが増える

それぞれの Agent 自身がミスをするだけでなく、マルチ Agent では次のような問題も新たに起こります。

- 2つの Agent が同じ作業を重複して行う
- 同じメッセージを Agent ごとに違う意味で解釈する
- 子タスクは終わっているのに、主タスクが収束しない
- コストと遅延が何層にもわたって積み重なる

つまり、

> マルチ Agent = 単体の知能の問題 + 分散協調の問題

ということです。

だからこそ、言葉としては強そうに聞こえますが、実際には不安定になりやすいのです。

---

## 最もよくある課題その1：重複作業

### なぜ重複しやすいのか？

タスクの境界が十分に明確でないと、次のようなことが起きやすくなります。

- プランナーが1回指示した
- 実行担当がさらに自分で再検索した
- レビュー担当が同じチェックをもう一度行った

### 最小例

```python
tasks_done = []

def run_task(agent, task):
    tasks_done.append((agent, task))

run_task("retriever_a", "返金ポリシーを検索する")
run_task("retriever_b", "返金ポリシーを検索する")

print(tasks_done)
```

想定出力：

```text
[('retriever_a', '返金ポリシーを検索する'), ('retriever_b', '返金ポリシーを検索する')]
```

この例はとても簡単ですが、すでに次のことを示しています。

> 重複排除の仕組みがないと、マルチ Agent は「見た目は忙しいが、実際には無駄が多い」状態になりやすい。

### 最小限の修正案

```python
assigned = set()
tasks_done = []

def run_task_once(agent, task):
    if task in assigned:
        return f"{agent}: スキップしました。タスクはすでに誰かが処理しています"
    assigned.add(task)
    tasks_done.append((agent, task))
    return f"{agent}: {task} を実行します"

print(run_task_once("retriever_a", "返金ポリシーを検索する"))
print(run_task_once("retriever_b", "返金ポリシーを検索する"))
print(tasks_done)
```

想定出力：

```text
retriever_a: 返金ポリシーを検索する を実行します
retriever_b: スキップしました。タスクはすでに誰かが処理しています
[('retriever_a', '返金ポリシーを検索する')]
```

---

## 最もよくある課題その2：メッセージのゆがみと状態の不一致

### なぜゆがむのか？

Agent 間でやり取りされるのは「現実そのもの」ではなく、次のようなものだからです。

- テキストメッセージ
- JSON メッセージ
- 中間状態

メッセージ形式が統一されていなかったり、フィールドの意味が曖昧だったりすると、システムは簡単にずれていきます。

- 私は A のことを言ったつもり
- でも相手は B だと受け取った

### 例

```python
message_a = {"task": "返金を確認する", "detail": "対外向けポリシーだけを見る"}
message_b = {"task": "返金を確認する", "detail": "社内のカスタマーサポート規定も含める"}

print(message_a)
print(message_b)
```

想定出力：

```text
{'task': '返金を確認する', 'detail': '対外向けポリシーだけを見る'}
{'task': '返金を確認する', 'detail': '社内のカスタマーサポート規定も含める'}
```

この 2 つのメッセージは少し違うだけですが、結果への影響は大きくなります。  
システムがメッセージのルールを厳しく決めていないと、後で簡単に方向がずれます。

### 実務上のコツ

システム内で次のような曖昧な意味のフィールドが出てきたら、注意が必要です。

- `task`
- `detail`
- `context`
- `notes`

こうしたフィールドを使うときは、通信設計がゆるくなっていないか確認しましょう。

---

## 最もよくある課題その3：結論の衝突をどう収束させるか？

### マルチ Agent では異なる結論が出やすい

たとえば、

- 法規担当 Agent は「可能」と判断する
- 業務ルール担当 Agent は「不可」と判断する

これは異常ではなく、むしろよくあることです。

### 最小限の衝突例

```python
results = {
    "policy_agent": {"decision": "allow", "confidence": 0.72},
    "risk_agent": {"decision": "deny", "confidence": 0.88}
}

print(results)
```

想定出力：

```text
{'policy_agent': {'decision': 'allow', 'confidence': 0.72}, 'risk_agent': {'decision': 'deny', 'confidence': 0.88}}
```

### 衝突解決には、少なくとも1つのルールが必要

もっとも単純でよく使われるルールは次のようなものです。

- confidence が高いほうを優先する
- レビュー担当が最終判断する
- 監督者が最終判断する
- 保守的に判断する（高リスクタスクでよく使う）

たとえば、保守的に判断する例は次のようになります。

前の衝突例の続きとして、同じファイルまたは同じインタプリタセッションで実行してください。`results` を再利用します。

```python
def resolve_with_safe_bias(results):
    decisions = [r["decision"] for r in results.values()]
    if "deny" in decisions:
        return "deny"
    return "allow"

print(resolve_with_safe_bias(results))
```

想定出力：

```text
deny
```

収束ルールを設計しないと、システムは次のようになってしまいます。

> どの Agent も一生懸命なのに、誰も最終判断できない。

---

## 最もよくある課題その4：コストと遅延が指数的に増えやすい

### なぜマルチ Agent は高くなりやすいのか？

Agent が1つ増えるごとに、通常は次の層も増えます。

- 推論コスト
- コンテキストの連結
- 状態の受け渡し
- ツール呼び出し

### とても分かりやすい例

```python
agents = [
    {"name": "planner", "cost": 0.002, "latency_ms": 400},
    {"name": "researcher", "cost": 0.003, "latency_ms": 700},
    {"name": "writer", "cost": 0.004, "latency_ms": 900},
    {"name": "reviewer", "cost": 0.002, "latency_ms": 500},
]

total_cost = sum(a["cost"] for a in agents)
total_latency = sum(a["latency_ms"] for a in agents)

print("total_cost =", total_cost)
print("total_latency_ms =", total_latency)
```

想定出力：

```text
total_cost = 0.011
total_latency_ms = 2500
```

これらの処理がさらに直列実行だと、全体の遅延はもっと目立ちます。

### とても重要な実務判断

多くの場合、マルチ Agent の問題は品質不足ではなく、次のような点にあります。

> 品質は 10% 上がったのに、コストと遅延は 3 倍になった。

だからこそ、意識的に次のことを考える必要があります。

- このステップは本当に必要か？
- 2つの役割をまとめられないか？
- 高リスクタスクのときだけレビュー担当を呼び出せないか？

---

## 最もよくある課題その5：システムの可観測性が低い

### なぜこれは大きな問題なのか？

マルチ Agent が失敗したとき、最終結果だけ見えていても、たいてい原因は分かりません。

- どの Agent が間違えたのか
- 間違いは通信、割り当て、ツールのどこで起きたのか
- どの時点でシステムがずれ始めたのか

### 最低限、次の情報は記録したい

- task_id
- agent_name
- action
- input summary
- output summary
- レイテンシ

最小限の trace 例は次のとおりです。

```python
trace = [
    {"task_id": "t1", "agent": "planner", "action": "decompose", "latency_ms": 120},
    {"task_id": "t1", "agent": "retriever", "action": "search_docs", "latency_ms": 350},
    {"task_id": "t1", "agent": "writer", "action": "draft", "latency_ms": 480}
]

for item in trace:
    print(item)
```

想定出力：

```text
{'task_id': 't1', 'agent': 'planner', 'action': 'decompose', 'latency_ms': 120}
{'task_id': 't1', 'agent': 'retriever', 'action': 'search_docs', 'latency_ms': 350}
{'task_id': 't1', 'agent': 'writer', 'action': 'draft', 'latency_ms': 480}
```

このような trace がないと、マルチ Agent システムのデバッグはかなり難しくなります。

---

## 最もよくある課題その6：役割の境界が少しずつずれる

### 役割の境界がずれるとは？

本来は、

- プランナーはタスクを分解する
- 執筆担当は答えを書く

という役割のはずです。

でも、だんだん次のように変わっていきます。

- プランナーも検索を始める
- 執筆担当もタスク優先度を判断し始める

最後には、どの役割も「何でもできる Agent」のようになってしまいます。

### なぜ危ないのか？

それは次の問題を引き起こすからです。

- 役割分担が曖昧になる
- デバッグが難しくなる
- 責任の境界が消える

なので、マルチ Agent システムでは、ときどき次のように確認する必要があります。

> この Agent の責任範囲は、もう越えていないか？

---

## より実践的な「課題チェックリスト」

マルチ Agent システムを作るときは、次のチェックリストがとても役立ちます。

| 問題 | よくある症状 |
|---|---|
| 重複作業 | 複数の Agent が同じことをする |
| メッセージのゆがみ | 同じタスクでも解釈が違う |
| 衝突が収束しない | 複数の結論が出ても誰も決めない |
| コストが高すぎる | 役割が多すぎる、各ステップが長すぎる |
| 状態が同期しない | 古い情報を元に作業し続ける |
| デバッグできない | 最終出力しか見えず、中間過程が見えない |

![マルチ Agent 課題の制御結果図](/img/course/ch09-multi-agent-challenge-control-result-map-ja.webp)

:::tip[まず制御信号を見る]
Agent を増やす前に、不安定さの原因が重複作業、メッセージのずれ、衝突の未収束、コスト増、trace 不足にないか確認します。
:::
---

## 解決の方向は「より複雑にすること」ではなく「より明確にすること」

問題にぶつかったとき、多くの人が最初に考えるのは次のようなことです。

- さらに調整用 Agent を追加する
- さらに判定用 Agent を追加する
- さらに要約用 Agent を追加する

でも、マルチ Agent システムを本当に安定させる方向は、しばしば「役割を増やすこと」ではありません。  
むしろ次のような点を明確にすることが大事です。

- メッセージをもっと明確にする
- 役割分担をもっと明確にする
- 終了条件をもっと明確にする
- 観測手段をもっと明確にする

つまり、

> マルチ Agent の修復は、「複雑さを足すこと」ではなく、「境界を引き直すこと」であることが多い。

---

## 小まとめ

この節で大事なのは、課題をただ並べることではなく、次の点を理解することです。

> **マルチ Agent システムの本当の難しさは、単体 Agent の能力ではなく、システム全体が収束し、観測でき、制御できるかどうかにある。**

マルチ Agent を見るときに、「重複」「衝突」「コスト」「観測」の 4 つで考えられるようになると、システムの調整はずっと分かりやすくなります。

---

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
役割: 所有者、作業者、レビュー担当、または専門担当の責務
メッセージ契約：artifact、request、response、handoff 状態
調整: ルーティング、タスク分割、衝突解決、最終責任者
失敗確認：重複作業、文脈喪失、責任者不在、またはメッセージループ
評価アクション：マルチ Agent の結果を単一 Agent のベースラインと比較する
```

## 練習

1. この節の衝突解決ロジックに、さらに「レビュー担当が最終判断する」版を設計してみましょう。
2. 考えてみましょう。あるマルチ Agent システムがいつも同じ情報を再検索してしまうとき、まず直すべきなのはタスク割り当て、通信プロトコル、それとも共有状態でしょうか？
3. 自分なりのマルチ Agent トレース 構造を設計してみましょう。少なくとも `task_id`、`agent`、`action`、`latency_ms` を含めてください。
4. 自分の言葉で説明してみましょう。なぜマルチ Agent システムで問題が起きたとき、原因は「モデルが弱い」ことではなく、「システムの境界が不明確」なことだと言えるのでしょうか？

<details>
<summary>参考実装と解説</summary>

1. reviewer が最終判断する設計では、各 Agent が conclusion、evidence、uncertainty を提出します。reviewer は criteria に照らして比較し、選択または統合し、その decision reason を記録します。
2. 同じ情報を何度も retrieval しているなら、まず shared state と communication protocol を見ます。Agent が何を取得済みか、evidence がどこに保存されているかを知らない可能性があります。
3. 有用な trace には `task_id`、`agent`、`action`、`input_ref`、`output_ref`、`latency_ms`、`status`、`error` を含められます。判断には evidence references も加えます。
4. 多くの失敗は境界の曖昧さから起きます。誰が task を持つのか、どの evidence が有効か、いつ止めるか、誰が決めるかです。強い model だけでは、悪い organization loop は補えません。

</details>
