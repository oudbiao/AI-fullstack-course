---
title: "9.2.1 Reasoning ロードマップ：Plan、Act、Check"
sidebar_position: 0
description: "Agent reasoning と planning の短い実践ロードマップ：中間手順を作り、行動を選び、進捗を監視し、失敗を評価する。"
keywords: [Agent reasoning guide, ReAct, Plan-and-Execute, planning]
---

# 9.2.1 Reasoning ロードマップ：Plan、Act、Check

Agent reasoning は長い回答ではありません。使える中間手順を作り、次に何をするか決め、計画がまだ有効か確認する力です。

## まず planning loop を見る

![Agent reasoning and planning 章の学習順序図](/img/course/ch09-reasoning-chapter-flow-ja.webp)

![Plan execute monitor replan map](/img/course/ch09-plan-execute-monitor-replan-map-ja.webp)

![Reasoning state checkpoint map](/img/course/ch09-reasoning-state-checkpoint-map-ja.webp)

基本習慣は、1 step を計画し、実行し、結果を観察し、state checkpoint を残し、状況が変われば replan することです。

## Plan checklist を動かす

tools を追加する前に、明示的な steps を作ります。print できない plan は inspect しにくいです。

```python
task = "prepare a cited RAG demo answer"
plan = ["inspect question", "retrieve sources", "draft answer", "check citations"]

print("task:", task)
for index, step in enumerate(plan, start=1):
    print(f"{index}. {step}")
print("checkpoint:", plan[-1])
```

期待される出力：

```text
task: prepare a cited RAG demo answer
1. inspect question
2. retrieve sources
3. draft answer
4. check citations
checkpoint: check citations
```

良い planning は見えるものです。失敗を見つけやすくし、最後の文章の裏に隠さないようにします。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | LLM reasoning | 答えを知ることと、道筋を導くことを区別する |
| 2 | Chain reasoning | 中間 state と self-check point を作る |
| 3 | ReAct | thought、action、observation、next step を交互に行う |
| 4 | Plan-and-Execute | タスクが大きいとき planning と execution を分ける |
| 5 | 高度な計画 | dependency、priority、rollback、replan を扱う |
| 6 | Reasoning evaluation | final result、path quality、failure type を採点する |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
task_goal: what the agent is trying to solve
plan_or_trace: reasoning steps, plan, ReAct trace, or execution graph
observation: what changed after each action
failure_check: hallucinated step, stale observation, loop, or unverified conclusion
eval_action: compare against expected result and revise the plan
```

## 合格ライン

計画の失敗理由を、分解不足、tool 選択ミス、古い observation、checkpoint 不足、弱い最終検証として説明できれば、この章は合格です。

出口ミニプロジェクトは、1 つの task に対する見える reasoning trace です：plan steps、observations、replans、final answer を残します。
