---
title: "9.7.3 Agent 間通信"
sidebar_position: 39
description: "メッセージ形式、同期と非同期、共有状態から失敗時のリトライまで、多 Agent 間が実際にどのように通信するのかを体系的に理解します。"
keywords: [multi-agent communication, message passing, event bus, shared state, async, protocol]
---

# 9.7.3 Agent 間通信

:::tip この節の位置づけ
前の節が「これらの Agent をどう分担させるか」に答えるものだとすれば、この節は次の問いに答えます。

> **役割分担を決めたあと、それぞれがどうやって情報をやり取りするのか？**

多 Agent システムが最後にうまくいかない理由は、個々の Agent が賢くないからではなく、通信設計が弱いからであることが多いです。
:::

## 学習目標

- 多 Agent 通信が、なぜシステムの成否を左右するのかを理解する
- メッセージパッシング、共有状態、イベントバスの3つの代表的な通信方法を区別する
- 最小のイベントバスの例を読めるようになる
- 同期通信と非同期通信の実装上の違いを理解する

---

## なぜ通信が多 Agent システムの核心問題になるのか？

### 多 Agent の最大のリスクは「仕事ができない」ことではなく、「互いに認識が揃わない」こと

各 Agent がそれぞれ強力でも、通信設計が悪いとシステムはうまく動きません。

- 作業の重複
- メッセージの取りこぼし
- 情報解釈の不一致
- すでに完了したのに、まだ議論を続けてしまう

### とても直感的なたとえ

多 Agent は、小さなチームで協力するのによく似ています。

- 役割分担は最初の一歩にすぎない
- 実際に効率を決めるのは、会議、引き継ぎ、同期、フィードバックといったコミュニケーションの仕組みです

だからこそ、通信は「付属モジュール」ではなく、核心となる構造なのです。

---

## もっともよく使われる3つの通信方法

### 直接メッセージパッシング（message passing）

ある Agent が、別の Agent に明確にメッセージを送ります。

利点：

- シンプル
- 明確
- 追跡しやすい

欠点：

- Agent 間の結びつきがやや強くなる

### 共有状態（shared state / blackboard）

すべての Agent が、共有の作業領域に情報を書き込み、読み取ります。

利点：

- 毎回、明示的に1対1で送る必要がない
- 複数の Agent が同じタスク状態を見ながら協力するのに向いている

欠点：

- 乱れやすい
- 権限管理や競合制御が難しい

### イベントバス（event bus）

Agent は互いを直接知らなくても、バスにメッセージを送り、購読者が受け取ります。

利点：

- より疎結合になる
- 複雑なシステムに向いている

欠点：

- デバッグが難しくなる

---

## まずは最もシンプルな1対1のメッセージパッシングを見る

### 最小例

```python
message = {
    "from": "planner",
    "to": "worker",
    "type": "task_assignment",
    "content": "返品ポリシーの重要条件を整理してください"
}

print(message)
```

想定出力：

```text
{'from': 'planner', 'to': 'worker', 'type': 'task_assignment', 'content': '返品ポリシーの重要条件を整理してください'}
```

### なぜこれでもう重要なのか？

通信の重要な要素が、すべて明示されているからです。

- 誰が送ったか
- 誰に送ったか
- メッセージの種類
- メッセージの内容

これは「適当に自然言語を1文渡す」より、ずっと安定しています。

---

## なぜメッセージ形式を標準化する必要があるのか？

### よくないメッセージ形式

```python
bad_message = "このタスクをやってください"
print(bad_message)
```

想定出力：

```text
このタスクをやってください
```

問題は次の通りです。

- 送信者がわからない
- タスクの種類がわからない
- 文脈がわからない
- 次にどう処理すればよいかがわからない

### より安定したメッセージ構造

```python
good_message = {
    "from": "planner",
    "to": "researcher",
    "type": "search_request",
    "task_id": "task_001",
    "payload": {
        "query": "返品ポリシー"
    }
}

print(good_message)
```

想定出力：

```text
{'from': 'planner', 'to': 'researcher', 'type': 'search_request', 'task_id': 'task_001', 'payload': {'query': '返品ポリシー'}}
```

これなら、システムの処理パイプラインに載せやすいメッセージになります。

![Agent 間通信契約図](/img/course/ch09-multi-agent-communication-contract-map-ja.webp)

:::tip 図の読み方
多 Agent 通信では、単に自然言語の一文だけを送らないようにしましょう。図の各メッセージには sender、receiver、type、task_id、payload、status が必要です。そうすることで、システムは追跡、再試行、責任の切り分けができます。
:::

---

## 最小のイベントバスの例

### 実行可能なコード

```python
from collections import defaultdict

class EventBus:
    def __init__(self):
        self.handlers = defaultdict(list)

    def subscribe(self, event_type, handler):
        self.handlers[event_type].append(handler)

    def publish(self, event_type, payload):
        for handler in self.handlers[event_type]:
            handler(payload)

def planner_handler(payload):
    print("[planner] 結果を受け取りました:", payload)

def worker_handler(payload):
    print("[worker] タスクを受け取りました:", payload)
    result = {
        "task_id": payload["task_id"],
        "summary": f"{payload['query']} の検索が完了しました"
    }
    bus.publish("task_done", result)

bus = EventBus()
bus.subscribe("task_assignment", worker_handler)
bus.subscribe("task_done", planner_handler)

bus.publish("task_assignment", {
    "task_id": "task_001",
    "query": "返品ポリシー"
})
```

想定出力：

```text
[worker] タスクを受け取りました: {'task_id': 'task_001', 'query': '返品ポリシー'}
[planner] 結果を受け取りました: {'task_id': 'task_001', 'summary': '返品ポリシー の検索が完了しました'}
```

### このコードが本当に教えていること

このコードが教えているのは、次の点です。

- 通信は必ずしも1対1で密結合する必要はない
- イベントの種類によって疎結合にできる
- 完了通知と結果通知を、同じ基盤で扱える

これは、かなり実際のシステムに近い通信の基本形です。

---

## 共有状態：どんなときに向いているのか？

### とても典型的な場面

複数の Agent が同じタスクを中心に動くとします。たとえば：

- planner が計画を書く
- retriever が資料を集める
- writer が下書きを作る
- reviewer がレビューを書く

このような場合、多くの情報を共有ワークスペースに置けます。

### 最小例

```python
shared_state = {
    "goal": "返品ポリシーの要約を完成させる",
    "plan": [],
    "evidence": [],
    "draft": None,
    "review": None
}

# planner
shared_state["plan"] = ["ポリシーを確認する", "要点を整理する", "要約を出力する"]

# retriever
shared_state["evidence"].append("購入後 7 日以内かつ学習進捗が 20% 未満なら返金可能")

# writer
shared_state["draft"] = "返金条件には、期間の制限と学習進捗の制限があります。"

print(shared_state)
```

想定出力：

```text
{'goal': '返品ポリシーの要約を完成させる', 'plan': ['ポリシーを確認する', '要点を整理する', '要約を出力する'], 'evidence': ['購入後 7 日以内かつ学習進捗が 20% 未満なら返金可能'], 'draft': '返金条件には、期間の制限と学習進捗の制限があります。', 'review': None}
```

### この方法の長所と短所

利点：

- みんなが同じ黒板を見られる
- 状態が一か所にまとまる

欠点：

- 誰が何を書けるかを管理する必要がある
- 上書き競合が起きやすい

---

## 同期通信と非同期通信はどう理解すればよいか？

### 同期通信

ある Agent がリクエストを送ったあと、相手の返答を待ってから次へ進みます。

利点：

- シンプル
- 理解しやすい

欠点：

- 詰まりやすい

### 非同期通信

メッセージを送ったあと、いったん別の作業を続け、相手が完了したら結果を受け取って処理します。

利点：

- より柔軟
- 複雑なシステムや高並列処理に向いている

欠点：

- 状態管理が複雑になる

### 実務で役立つ直感

タスクの流れが短く、手順がはっきりしているなら、まずは同期で十分です。  
タスクが長く、待ち時間が安定しないなら、非同期を検討しましょう。

---

## Agent 間通信でよくある失敗点

### メッセージ形式が統一されていない

今日は `task_id`、明日は `id`、その次は `job_id` というように変わると、システムはどんどん混乱します。

### メッセージは送られたのに、誰も処理しない

これはイベントシステムでとてもよくある問題です。

- publish はされた
- でも subscriber がいない

### 複数の Agent が同じメッセージを違う意味に解釈する

たとえば：

- ある Agent は「検索依頼」だと思う
- 別の Agent は「要約依頼」だと思う

これではシステムの動きがずれてしまいます。

### タイムアウトとリトライがない

ある Agent が止まると、システム全体がずっと待ち続けてしまう可能性があります。

---

## 本番に近いシステムで通信を安定させるには？

### 統一されたメッセージプロトコルを使う

少なくとも次を統一しましょう。

- `from`
- `to`
- `type`
- `task_id`
- `payload`

### 統一された状態追跡を行う

各タスクには、できれば一意な ID を付けます。そうすると次のことがしやすくなります。

- 完全な処理経路の追跡
- 再現
- デバッグ

### 統一されたタイムアウトと失敗時の方針を決める

たとえば：

- タイムアウトしたら自動でフォールバックする
- 失敗したら人間に引き継ぐ
- 数回再試行してもだめなら終了する

---

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
roles: owner, worker, reviewer, or specialist responsibilities
message_contract: artifact, request, response, and handoff state
coordination: routing, task split, conflict resolution, and final owner
failure_check: duplicated work, lost context, no accountable owner, or message loop
eval_action: compare multi-agent result against single-agent baseline
```

## まとめ

この節で最も大切なのは、「message passing、event bus、shared state」という言葉を覚えることではなく、次を理解することです。

> **多 Agent 通信の本質は、メッセージを送ることそのものではなく、メッセージの構造を安定させ、責任の所在を明確にし、失敗を制御可能にすることです。**

通信層が安定してこそ、多 Agent システムは「組織の混乱」でモデルの能力を無駄にせずに済みます。

---

## 練習

1. イベントバスの例に `reviewer_handler` を追加し、`task_done` を購読させてください。
2. 自分用の統一メッセージプロトコルを設計し、少なくとも `type`、`task_id`、`payload` を含めてください。
3. どんなときに、1対1のメッセージより共有状態を使いたくなりますか？
4. 自分の言葉で説明してみましょう。なぜ多 Agent システムでは、通信設計がタスク分担と同じくらい重要なのでしょうか？
