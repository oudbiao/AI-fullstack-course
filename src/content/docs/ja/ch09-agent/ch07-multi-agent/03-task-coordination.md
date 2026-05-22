---
title: "9.7.4 タスク分配と協調"
description: "タスクの分解、役割の割り当て、状態の同期から競合解決まで、多 Agent システムが実際にどうやって仕事を振り分けて回収するのかを理解します。"
sidebar:
  order: 40
head:
  - tag: meta
    attrs:
      name: keywords
      content: "task coordination, task assignment, multi-agent, scheduling, conflict resolution"
---

# 9.7.4 タスク分配と協調

:::tip[この節の位置づけ]
前の節では通信について学び、「情報がどうやって伝わるか」を見ました。  
この節で解決するのは、もう一つの、さらにやっかいな問題です。

> **タスクをどう分け、どう割り当て、どう回収するのか？**

割り当てがうまくいかないと、多 Agent システムは通信ができても、非効率になったり、互いにぶつかり合ったりします。
:::
## 学習目標

- 多 Agent においてタスク分配がなぜ核心的な問題なのかを理解する
- 静的分配、動的分配、能力ルーティングという代表的な方法を区別できるようになる
- 協調でよく起きる競合と、その解決方法を理解する
- 小さなタスクスケジューリングの例を読めるようになる

---

## なぜ多 Agent では「多い」だけが難しさではないのか

### 多 Agent の最大のリスク：誰も仕事をしないのではなく、みんなが間違って仕事をすること

多 Agent システムでよくある失敗は、単に次のようなものだけではありません。

- 誰もやらない

むしろよくあるのは、次のようなケースです。

- 2人が同じことを重複してやる
- 役割に合わない Agent がタスクを受ける
- タスクの順番が間違う
- 結果が戻ってこない

だから本当に大事なのは、次のことです。

> **適切な人が、適切なタイミングで、適切なことをするようにすること。**

### 生活のたとえ

小さなプロジェクトを進めるときのようなものです。

- ある人は資料集め
- ある人はコードを書く
- ある人はレビューする

分担が乱れると、どんなに優秀でも非効率になります。

---

## もっともよく使われる3つのタスク分配方式

### 静的分配

タスクと役割をあらかじめ固定しておきます。

たとえば：

- 検索は必ず検索担当に渡す
- 執筆は必ず執筆担当に渡す

利点：

- 安定している
- デバッグしやすい

欠点：

- 柔軟性が低い

### 動的分配

システムが、その時点のタスク内容に応じて、誰に渡すかを決めます。

たとえば：

- 法律の質問は legal_agent に渡す
- 技術の質問は tech_agent に渡す

利点：

- より柔軟

欠点：

- ルーティングを間違えると、連鎖的に誤りが起きる

### 能力ルーティング

名前で分けるのではなく、能力の特徴で分けます。

- 誰が検索に向いているか？
- 誰が要約に向いているか？
- 誰がレビューに向いているか？

これは「担当できる能力に応じて仕事を振る」やり方に近いです。

---

## 最小のタスク分配例

```python
agents = {
    "researcher": {"skills": ["search", "retrieve"]},
    "writer": {"skills": ["write", "summarize"]},
    "reviewer": {"skills": ["review", "critique"]}
}

tasks = [
    {"name": "資料を調べる", "skill": "search"},
    {"name": "要約を書く", "skill": "write"},
    {"name": "レビューする", "skill": "review"}
]

def assign_task(task, agents):
    for agent_name, profile in agents.items():
        if task["skill"] in profile["skills"]:
            return agent_name
    return None

for task in tasks:
    print(task["name"], "->", assign_task(task, agents))
```

想定出力：

```text
資料を調べる -> researcher
要約を書く -> writer
レビューする -> reviewer
```

### このコードは何を教えてくれるのか？

ここで学ぶべき、とても大事な考え方があります。

> タスク分配はランダムに仕事を振ることではなく、「タスクの要求」と「Agent の能力」を対応づけることです。 

---

## タスク協調は、割り当てるだけでなく順序も管理する

### 並列にできないタスクもある

たとえば：

1. まず資料を調べる
2. 次に要約を書く
3. 最後にレビューする

順番を逆にすると、システムはうまく動きません。

### 最小の順序スケジューリング例

```python
dependencies = {
    "retrieve": [],
    "write": ["retrieve"],
    "review": ["write"]
}

done = set()
execution_order = []

while len(done) < len(dependencies):
    for task, need in dependencies.items():
        if task not in done and all(n in done for n in need):
            done.add(task)
            execution_order.append(task)

print(execution_order)
```

出力は次のようになります。

```text
['retrieve', 'write', 'review']
```

これが、多 Agent の協調でとても重要なもう一つの層です。  
**誰がやるかだけでなく、どの順番でやるかも知る必要があります。**

---

## タスク協調でよく起きる競合

### 重複作業

2つの Agent が同じことをしてしまう。

### 結論の衝突

ある Agent は「返金できる」と言い、別の Agent は「返金できない」と言う。

### 状態の不一致

writer はまだ資料が見つかっていないと思っているのに、retriever はすでに返している。

### なぜこうした問題がよく起きるのか？

多 Agent は本質的に、「分散システムの小型版」だからです。  
分業が始まると、次の問題が必ず出てきます。

- 同期
- 競合
- 収束

---

## 競合解決の考え方を入れた最小例

```python
results = {
    "agent_a": {"decision": "approve", "confidence": 0.7},
    "agent_b": {"decision": "reject", "confidence": 0.9}
}

def resolve_conflict(results):
    best_agent = max(results.items(), key=lambda x: x[1]["confidence"])
    return {
        "final_decision": best_agent[1]["decision"],
        "source": best_agent[0]
    }

print(resolve_conflict(results))
```

想定出力：

```text
{'final_decision': 'reject', 'source': 'agent_b'}
```

### なぜこれは最小版にすぎないのか？

実際のシステムでは、競合解決に次のような方法を使うことがあります。

- 置信度
- 投票
- レビュー担当の裁定
- 監督者の最終判断

でも、まずは少なくとも次を理解する必要があります。

> 多 Agent では必ず競合が起こります。競合は異常ではなく、むしろ通常のことです。 

![多 Agent の協調、競合、収束の図](/img/course/ch09-multi-agent-coordination-cost-map-ja.webp)

:::tip[図の読み方]
この図は協調コストを表しています。タスク分配、依存順序、共有状態、競合の裁定はすべて複雑さを増やします。多 Agent の利点が、こうした通信コストや収束コストを上回る場合にだけ、導入する価値があります。
:::
---

## タスク協調と通信の関係は？

通信が解決するのは：

- 情報をどう伝えるか

協調が解決するのは：

- タスクをどう並べるか
- 誰が何を担当するか
- 競合が起きたときにどう収束させるか

つまり、次のように覚えるとよいです。

- 通信は「回線」に近い
- 協調は「スケジューリング」に近い

どちらも欠かせません。

---

## 実際のシステムでよく使われる協調戦略

### 中央調度型

supervisor が一括でタスクの流れを決めます。

利点：

- 管理しやすい

### 分散協議型

Agent 同士が提案し合い、協議します。

利点：

- 柔軟

欠点：

- 調整が難しい

### 半中央型

大きな方針は supervisor が管理し、細かい部分は worker が自律的に進めます。

実務では、これが比較的バランスのよい選択になることが多いです。

---

## 初学者がよくやってしまう落とし穴

### 分業だけして、最後の回収を設計しない

タスクが途中まで進んでも、最後を誰も担当しないのは、とてもよくある問題です。

### 成功パスだけを設計する

一度でも Agent がタイムアウト、失敗、競合すると、システムが壊れます。

### 「Agent が増えれば効率も上がる」と思い込む

協調がうまくできなければ、Agent が増えるほど管理コストも増えます。

---

## 小結

この節で最も重要なのは、タスクを「分ける」ことそのものではなく、次を理解することです。

> **タスク分配と協調の核心は、タスク、役割、順序、競合処理を、収束できる1つのシステムとしてまとめることです。**

これこそが、多 Agent が「にぎやかに見える」段階から、「本当に効率よく協働する」段階へ進むための鍵です。

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

1. タスク分配の例に `planner` Agent を追加し、実行順序を決める役割を持たせてみましょう。
2. 「検索 -> 執筆 -> レビュー -> 修正」という協調フローを設計してみましょう。
3. 2つの Agent の結論が衝突したら、投票、信頼度による裁定、レビュー担当の判断のどれを選びますか？ なぜですか？
4. 自分の言葉で説明してみましょう。なぜ多 Agent の協調は、本質的に小さなタスクスケジューリングシステムに似ているのでしょうか？

<details>
<summary>参考実装と解説</summary>

1. planner Agent は依存関係つきの ordered task list を作ります。たとえば retrieve を先に行い、evidence がある後に write、draft がある後に review、review が変更を求めた場合だけ revise します。
2. coordination flow は、retrieve evidence -> write draft with citations -> review correctness and gaps -> revise rejected parts -> final check のようにできます。
3. 結論が衝突した場合は risk に応じて arbitration を選びます。低リスクの意見なら voting、測定可能な evidence があるなら confidence、基準と責任が重要なら reviewer judgment が向きます。
4. Multi-Agent coordination が task scheduling に似ているのは、dependencies、resource use、order、retry、stopping conditions、final acceptance を管理するからです。

</details>
