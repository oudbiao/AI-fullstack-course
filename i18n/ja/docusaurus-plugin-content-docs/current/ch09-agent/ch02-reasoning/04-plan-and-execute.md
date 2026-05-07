---
title: "9.2.5 計画してから実行する方式（Plan-and-Execute）"
sidebar_position: 8
description: "「まず計画し、それから実行する」を分けてわかりやすく説明し、長いタスクや複雑なタスクがなぜ全工程を即興の ReAct で進めるのに向いていないのか、そしてなぜより安定した計画層が必要なのかを理解する。"
keywords: [plan and execute, planner, executor, workflow, long horizon tasks, agent planning]
---

# 9.2.5 計画してから実行する方式（Plan-and-Execute）

:::tip この節の位置づけ
ReAct は、行きながら様子を見るのにとても向いています。  
でも、タスクが長くなると、典型的な問題が出てきます。

- 毎回その場で決めると、全体の方向がぶれやすい

そんなとき、多くのシステムは次のような組み立て方に切り替えます。

> **まず planner で計画を分解し、そのあと executor が計画どおりに順番に実行する。**

これが Plan-and-Execute の核心です。
:::

## 学習目標

- Plan-and-Execute が長いタスクに向いている理由を理解する
- planner と executor の役割分担を理解する
- 実行できるサンプルを通して、最小の「まず計画してから実行する」システムを理解する
- それが ReAct とどう違い、どう使い分けるかを理解する

---

## まず地図を一枚つくる

Plan-and-Execute は、「高いレベルで先にルートを決め、低いレベルで手順を進める」と考えるとわかりやすいです。

```mermaid
flowchart LR
    A["ユーザーの目標"] --> B["Planner が先に手順へ分解"]
    B --> C["Executor が順番に実行"]
    C --> D["context に書き戻す"]
    D --> E["結果をまとめる"]
```

この節で本当に解決したいのは、次の点です。

- なぜ長いタスクは即興だけでは向かないのか
- なぜ計画層と実行層を分けると、システムがより安定するのか

---

## なぜタスクが長くなると「先に計画する」必要があるのか？

### 行きながら考えると全体像を見失いやすい

タスクが1〜2手で終わるなら、  
ReAct の即興的な判断でたいてい足ります。

でも、タスクが例えば次のようになると、

- 1週間分のサポートデータを整理する
- よくある質問を集計する
- レポートを作る
- さらに改善案を出す

こうした長いタスクには、より強い全体構造があります。

毎回その場で決めていると、よくある問題は次のとおりです。

- 手順を飛ばす
- 順番を間違える
- 同じ作業を重ねてしまう

### Planner の役割：大きな仕事を小さな仕事に分ける

Planner の一番大事な価値は、「もっと賢いこと」ではなく、  
次のことです。

- 先に道筋を描く

Planner は次のような問いに答えます。

- 何のステップが必要か
- どの順番で進めるか
- どの結果を次のステップに渡すか

### Executor の役割：今のステップをしっかりやり切る

計画を分けておくと、  
executor は「戦略を考える」負担を少し減らして、  
次に集中できます。

- 今のステップをどう終わらせるか
- 今のツールをどう呼ぶか
- 今の結果をどう保存するか

これでシステムは安定しやすくなり、デバッグもしやすくなります。

### 初心者向けのいちばん身近なたとえ

Plan-and-Execute は、次のように考えるとよいです。

- まず工事のチェックリストを書いて、そのあと作業員がチェックリストどおりに進める

チェックリストがなくても、作業員はその場で考えながら進められます。  
でも、仕事が長くなると、どうしても次の問題が起こりやすくなります。

- 手順を飛ばす
- 順番を間違える
- やり直しが増える

このたとえは初心者にとても向いています。  
なぜなら、「planner / executor」を、日常的な仕事の組み立てとして捉え直せるからです。

---

## Plan-and-Execute と ReAct の違いはどこにあるのか？

### ReAct は、調べながら進むイメージに近い

向いているのは次のような場面です。

- 未知の情報が多い
- 次の一手が、前回の observation によって決まる

### Plan-and-Execute は、先に作業表を作るイメージに近い

向いているのは次のような場面です。

- タスクの構造が比較的はっきりしている
- ステップを前もって分解できる
- 即興のぶれを減らしたい

### この2つは対立するものではない

実際のシステムでは、よく組み合わせて使います。

- 上位ではまず Plan-and-Execute
- 各実行ステップの中では ReAct を使う

つまり、

- 計画は全体を担当する
- ReAct は局所的な探索を担当する

### 初学者が最初に覚えるとよい選び方

| タスクの特徴 | まず安定しやすい選び方 |
|---|---|
| ルートがはっきりしていて、手順が多い | Plan-and-Execute |
| 未知情報が多く、その場で調べながら進む | ReAct |
| 全体計画も局所探索も必要 | 両方を組み合わせる |

この表は初心者にとても役立ちます。  
「どの推論の組み立て方を使うべきか」を、判断できる問題にしてくれるからです。

![Plan-and-Execute モニタリングと再計画の図](/img/course/ch09-plan-execute-monitor-replan-map-ja.png)

:::tip 図の読み方
図を見るときは、2つの役割に注目してください。Planner は全体のルートを担当し、Executor は今のステップを担当します。Monitor は、情報不足・ツール失敗・目標変更を見つけたら、無理に元の計画を進めるのではなく replan を起こします。
:::

---

## まずは最小の Plan-and-Execute サンプルを動かしてみる

以下の例は、「サポート週報 Agent」をまねたものです。  
ユーザーのタスクは次のとおりです。

- サポートの問題を集計する
- よく出る intent を見つける
- 簡単な要約を作る

ここでは、明確に次の2つを分けます。

- planner
- executor

```python
tickets = [
    {"intent": "refund", "text": "注文がまだ発送されていません。返金できますか？"},
    {"intent": "refund", "text": "返金はいつ入りますか？"},
    {"intent": "password", "text": "パスワードを忘れました。どうすればいいですか？"},
    {"intent": "address", "text": "住所を間違えました。まだ変更できますか？"},
    {"intent": "refund", "text": "返金がまだ反映されないのはなぜですか？"},
]


def planner(goal):
    return [
        {"step": "load_tickets", "description": "今週のサポートチケットを読み込む"},
        {"step": "count_intents", "description": "各種類の問題数を集計する"},
        {"step": "find_top_intent", "description": "最も多い問題を見つける"},
        {"step": "draft_report", "description": "短い週報を作る"},
    ]


def executor(task, context):
    name = task["step"]

    if name == "load_tickets":
        context["tickets"] = tickets
        return "5件のチケットを読み込みました"

    if name == "count_intents":
        counts = {}
        for item in context["tickets"]:
            counts[item["intent"]] = counts.get(item["intent"], 0) + 1
        context["intent_counts"] = counts
        return counts

    if name == "find_top_intent":
        counts = context["intent_counts"]
        top_intent = max(counts, key=counts.get)
        context["top_intent"] = top_intent
        return top_intent

    if name == "draft_report":
        counts = context["intent_counts"]
        top_intent = context["top_intent"]
        report = (
            f"今週は合計 {len(context['tickets'])} 件のサポートチケットを対応しました。"
            f"最も多かった問題は {top_intent} で、{counts[top_intent]} 回ありました。"
            f"まずは {top_intent} の手順と FAQ 文を改善することをおすすめします。"
        )
        context["report"] = report
        return report

    raise ValueError(f"Unknown step: {name}")


goal = "今週のサポート問題の週報を作る"
plan = planner(goal)
context = {}
trace = []

for task in plan:
    output = executor(task, context)
    trace.append({"task": task["step"], "output": output})

print("plan:")
for item in plan:
    print("-", item)

print("\ntrace:")
for item in trace:
    print(item)

print("\nfinal report:")
print(context["report"])
```

期待される出力：

```text
plan:
- {'step': 'load_tickets', 'description': '今週のサポートチケットを読み込む'}
- {'step': 'count_intents', 'description': '各種類の問題数を集計する'}
- {'step': 'find_top_intent', 'description': '最も多い問題を見つける'}
- {'step': 'draft_report', 'description': '短い週報を作る'}

trace:
{'task': 'load_tickets', 'output': '5件のチケットを読み込みました'}
{'task': 'count_intents', 'output': {'refund': 3, 'password': 1, 'address': 1}}
{'task': 'find_top_intent', 'output': 'refund'}
{'task': 'draft_report', 'output': '今週は合計 5 件のサポートチケットを対応しました。最も多かった問題は refund で、3 回ありました。まずは refund の手順と FAQ 文を改善することをおすすめします。'}

final report:
今週は合計 5 件のサポートチケットを対応しました。最も多かった問題は refund で、3 回ありました。まずは refund の手順と FAQ 文を改善することをおすすめします。
```

### このコードでいちばん大事な点は何か？

次の2つをはっきり分けていることです。

1. 計画  
   どんなステップを実行するかを決める
2. 実行  
   実際にステップを進め、その結果を context に入れる

これが Plan-and-Execute のもっとも本質的な構造です。

### ここでの `context` は何をしているのか？

`context` は、実行中の共有状態です。

前のステップで出た

- `tickets`
- `intent_counts`
- `top_intent`

は、次のステップでも使われます。

つまり Plan-and-Execute で大切なのは、  
「plan があるか」だけではなく、

- 中間結果をどう安全に受け渡すか

でもあります。

### これは単なる `for step in plan` より、なぜ学ぶ価値があるのか？

これは単なるループの説明ではなく、  
次のことを示しているからです。

- 長いタスクをどう分けるか
- 依存関係をどう渡すか
- 最終結果をどう少しずつまとめるか

### さらに小さな「計画チェック表」の例

```python
plan_quality = {
    "steps_clear": True,
    "order_defined": True,
    "handoff_defined": False,
}


def next_fix(plan_quality):
    if not plan_quality["steps_clear"]:
        return "まずステップの説明をわかりやすくしましょう。"
    if not plan_quality["order_defined"]:
        return "まず実行順をはっきりさせましょう。"
    if not plan_quality["handoff_defined"]:
        return "まず各ステップの出力が次のステップにどう渡るかを書きましょう。"
    return "この計画は、基本的な実行可能性があります。"


print(next_fix(plan_quality))
```

期待される出力：

```text
まず各ステップの出力が次のステップにどう渡るかを書きましょう。
```

この例は初心者にとても向いています。  
なぜなら、次のことを思い出させてくれるからです。

- よい計画は、ただステップを書くだけではない
- ステップ同士の受け渡しも考える必要がある

---

## Plan-and-Execute はどんな場面で特に価値があるのか？

### 長いタスク

たとえば次のようなものです。

- レポートを書く
- 研究内容をまとめる
- ナレッジベースを整理する
- 複数ステップの業務フローを組む

### 安定して再現したい流程

同じ種類のタスクを、毎回なるべく近い形で実行したいなら、  
明示的な計画のほうが安定しやすいです。

### 人が計画を確認したい場面

タスクによっては、  
実行前に plan を人間に見せて、実行してよいか確認することもあります。

たとえば、

- リスクの高い操作
- 複雑なデータ処理
- 自動化フローの変更

---

## いちばん起きやすい問題は何か？

### 最初の計画分解が間違っている

planner がタスクを誤解すると、  
その後 executor がどれだけ丁寧でも意味がありません。

### 計画が固定されすぎていて、新しい観察に対応できない

これが Plan-and-Execute の典型的な弱点です。

外の状況がすぐ変わる場合、  
固定しすぎた計画は硬く見えてしまいます。

### executor と計画の説明がかみ合っていない

よくあるのは、

- planner があいまいなステップを書く
- executor が、どう実行すればいいかわからない

というケースです。

なので、計画のステップはできるだけ次のようにします。

- 粒度がはっきりしている
- 実行できる
- 入出力が明確

---

## 実務ではどうすれば Plan-and-Execute をより安定させられるか？

### plan を構造化する

自然文をただ長く並べるだけにしないほうがよいです。  
よりよい形は、たとえば次のようなものです。

- step id
- description
- input
- output

### 各ステップの実行後に必ず context に書き戻す

これにより、次のことがやりやすくなります。

- デバッグ
- 再実行
- リプレイ

### 必要なら replan を許可する

Plan-and-Execute で本当に安定しやすい形は、次のようなものです。

- 一度計画したら絶対に変えない

ではなく、

- 大きな方向は先に決める
- 大きくずれたら再計画してよい

## これをプロジェクトやシステム設計として見せるなら、何を見せるのが一番よいか

見せる価値が高いのは、たいてい次のような内容です。

- 「システムがまず計画を作りました」という事実だけ
ではありません。

1. ユーザーの目標
2. Planner が分解したステップ
3. 各ステップ実行後に context がどう変わったか
4. どこで replan が必要だったか

これを見ると、他の人にも次のことが伝わりやすくなります。

- 長いタスクをどう組み立てているかを理解している
- ただ prompt を1段増やしただけではない

---

## よくある勘違い

### 勘違い1：plan があれば、必ずもっと賢くなる

計画は安定性を上げます。  
でも前提として、計画そのものの質が十分である必要があります。

### 勘違い2：すべてのタスクで、必ず planner → executor にするべき

そんなことはありません。  
短いタスクや、対話性が強いタスクでは、ReAct のほうが自然なことが多いです。

### 勘違い3：計画はステップ名だけ書けば十分

本当に実行できる計画には、次の情報も必要です。

- ステップの粒度
- 状態の依存関係
- 何を出力するか

---

## まとめ

この節でいちばん大事なのは、`Plan-and-Execute` を単なる流行語として覚えることではなく、  
その本質的なエンジニアリング価値を理解することです。

> **タスクが長く、複雑で、より安定した再現性が必要なとき、まず計画してから実行すると、即興のぶれを大きく減らせて、システムをデバッグしやすく、レビューしやすく、保守しやすくなる。**

この理解ができていれば、  
次に DAG での計画、多 Agent の分担、タスクグラフのスケジューリングを見たときにも、かなりスムーズにつながります。

---

## 練習

1. サンプルの「サポート週報」を「ナレッジベースの回答整理」または「競合調査」に置き換えて、もう一度 plan を書いてみましょう。
2. なぜ長いタスクは短いタスクより planner が必要になりやすいのでしょうか？
3. 実行の途中で目標が変わったら、replan の仕組みをどう設計しますか？
4. 考えてみましょう。どんなタスクが ReAct に向いていて、どんなタスクが Plan-and-Execute に向いているでしょうか？
