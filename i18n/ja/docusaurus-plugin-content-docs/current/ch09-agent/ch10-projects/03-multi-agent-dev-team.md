---
title: "9.10.4 プロジェクト：マルチ Agent 開発チーム【選択】"
sidebar_position: 56
description: "planner、coder、reviewer、tester の4種類の役割を中心に、マルチ Agent 開発チームプロジェクトの作品レベルの最小ループを作る。"
keywords: [multi-agent dev team, planner, coder, reviewer, tester, project]
---

# 9.10.4 プロジェクト：マルチ Agent 開発チーム【選択】

:::tip この節の位置づけ
マルチ Agent 開発チームのプロジェクトは、すぐに“見せるだけ”になりがちです。

- 役割が多い
- 会話が多い
- でも結果が安定しない

だから本当に価値があるのは、役割の数ではなく、次の点です。

> **タスクが安定して分解されているか、引き継ぎが明確か、失敗後に巻き戻せるか。**

この章では、作品レベルの「最小ループ」を作ります。
:::

## 学習目標

- マルチ Agent 開発チームの最小役割セットを定義できるようになる
- 役割間で最も重要な引き継ぎ用アーティファクトが何かを理解する
- そのまま見せられる、検証可能なマルチ Agent プロジェクトの骨組みを作る
- なぜ「何回も会話すること」より、プロトコルと状態のほうが大事なのかを理解する

---

## なぜ最小役割セットだけで十分なことが多いのか？

とても安定した最小ループには、普通は次の4つだけあれば十分です。

- プランナー
- 実装担当（コーダー）
- レビュー担当
- テスト担当

この4種類の役割があれば、次の流れを十分に示せます。

- タスク分解
- 実装
- レビュー
- 検証

最初から役割を増やしすぎると、  
システムは忙しそうに見えても、実際には空回りしやすくなります。

---

## まずは役割間のアーティファクト引き継ぎ例を動かしてみよう

この例では実際にはコードを変更しません。  
ただし、最も重要な「引き継ぎアーティファクト」の形を表します。

```python
from dataclasses import dataclass


@dataclass
class TaskPlan:
    goal: str
    files_to_change: list
    acceptance_test: str


@dataclass
class Patch:
    summary: str
    changed_files: list


@dataclass
class ReviewNote:
    approved: bool
    issues: list


@dataclass
class TestReport:
    passed: bool
    cases: list


plan = TaskPlan(
    goal="返金ページの金額表示エラーを修正する",
    files_to_change=["refund.py", "test_refund.py"],
    acceptance_test="100円と8割引を入力したら、結果は80円になること",
)

patch = Patch(
    summary="割引計算ロジックを修正し、テストを追加する",
    changed_files=["refund.py", "test_refund.py"],
)

review = ReviewNote(
    approved=False,
    issues=["変数名がわかりにくい", "境界条件のテストが不十分"],
)

test_report = TestReport(
    passed=False,
    cases=["test_discount_basic", "test_discount_zero"],
)

print(plan)
print(patch)
print(review)
print(test_report)
```

実行結果の例：

```text
TaskPlan(goal='返金ページの金額表示エラーを修正する', files_to_change=['refund.py', 'test_refund.py'], acceptance_test='100円と8割引を入力したら、結果は80円になること')
Patch(summary='割引計算ロジックを修正し、テストを追加する', changed_files=['refund.py', 'test_refund.py'])
ReviewNote(approved=False, issues=['変数名がわかりにくい', '境界条件のテストが不十分'])
TestReport(passed=False, cases=['test_discount_basic', 'test_discount_zero'])
```

![マルチ Agent artifact 引き継ぎ結果図](/img/course/ch09-multi-agent-artifact-handoff-anatomy-result-map-ja.webp)

### この例でいちばん大事な点は何か？

この例が示しているのは、マルチ Agent プロジェクトで本当に見せるべきものは、  
単なる会話ログではなく、次のようなものだということです。

- 引き継ぎアーティファクト
- タスク状態
- 結果の検証

### なぜアーティファクトのほうが会話より重要なのか？

アーティファクトこそが、その後の役割が実際に依存する入力だからです。  
会話だけを見ても、システムが安定して協調できるかどうかは判断しにくいです。

---

## 最小ワークフローのループ

同じファイルまたは同じ Python セッションで続けて実行してください。このブロックは前の例の dataclass を再利用します。

次に、4つの役割を1本の最小フローにつなげます。

```python
def planner(goal):
    return TaskPlan(
        goal=goal,
        files_to_change=["refund.py", "test_refund.py"],
        acceptance_test="100円と8割引を入力したら、結果は80円になること",
    )


def coder(plan):
    return Patch(
        summary=f"タスク目標に基づいて実装: {plan.goal}",
        changed_files=plan.files_to_change,
    )


def reviewer(patch):
    if "test_refund.py" not in patch.changed_files:
        return ReviewNote(approved=False, issues=["テストファイルの変更がありません"])
    return ReviewNote(approved=True, issues=[])


def tester(review_note):
    if not review_note.approved:
        return TestReport(passed=False, cases=["review_failed"])
    return TestReport(passed=True, cases=["test_discount_basic", "test_discount_zero"])


goal = "返金ページの金額表示エラーを修正する"
plan = planner(goal)
patch = coder(plan)
review = reviewer(patch)
test_report = tester(review)

print(plan)
print(patch)
print(review)
print(test_report)
```

実行結果の例：

```text
TaskPlan(goal='返金ページの金額表示エラーを修正する', files_to_change=['refund.py', 'test_refund.py'], acceptance_test='100円と8割引を入力したら、結果は80円になること')
Patch(summary='タスク目標に基づいて実装: 返金ページの金額表示エラーを修正する', changed_files=['refund.py', 'test_refund.py'])
ReviewNote(approved=True, issues=[])
TestReport(passed=True, cases=['test_discount_basic', 'test_discount_zero'])
```

![マルチ Agent 開発チームの artifact トレース 結果図](/img/course/ch09-multi-agent-dev-team-artifact-trace-result-map-ja.webp)

:::tip 結果の読み方
出力を artifact chain として読みます。`TaskPlan` が目標と受け入れ条件を定義し、`Patch` が実装とテストファイルを変え、`ReviewNote` がゲートになり、`TestReport` が最後の証拠になります。
:::

### なぜこのループだけでもかなり実際のプロジェクトに近いのか？

マルチ Agent プロジェクトで本当に大事な3つの点が入っているからです。

1. 役割分担
2. 明確なアーティファクトの引き継ぎ
3. レビューとテストにもとづくフィードバックループ

### レビュー担当が承認しないのにテスト担当が続けないほうがいいのはなぜ？

これは、マルチ Agent システムが「みんなが並行で好きにやる」ものではなく、  
次のことをきちんと守る必要があるからです。

- 段階依存
- 引き継ぎ品質

![マルチ Agent 開発チームの成果物ループ図](/img/course/ch09-multi-agent-dev-team-delivery-map-ja.webp)

:::tip 図の見方
この図が強調しているのは、「役割の数が重要なのではなく、アーティファクトの引き継ぎが重要」ということです。planner は plan を出し、coder は patch を出し、reviewer は issue を出し、tester は test report を出します。失敗したら、それぞれの役割に戻って修正します。
:::

---

## 作品レベルのプロジェクトで本当に見せるべきものは？

### 1本の完全なタスク トレース

たとえば次のような流れです。

- タスク目標
- 計画（plan）
- パッチ（patch）
- レビュー指摘（review issues）
- テストレポート（test report）

### 1回の失敗からの巻き戻し

これは非常に説得力があります。  
たとえば：

- レビュー担当が差し戻す
- コーダー が再修正する
- テスト担当が再度検証する

### はっきりした役割の境界

作品集では、次の点に答えられる必要があります。

- なぜこの4つの役割が必要なのか
- 各役割の入力と出力は何か

---

## いちばんハマりやすい落とし穴

### 役割は多いのに境界が不明確

これだと、システムは複雑に見えますが、  
実際には同じ作業を繰り返しているだけです。

### 共有状態や統一されたアーティファクト形式がない

この場合、役割間の引き継ぎが安定しません。

### 成功パスしか見せない

良いマルチ Agent プロジェクトでは、むしろ次のことを見せるべきです。

- 失敗後にどう巻き戻すか
- どの段階で問題が起きやすいか

---

期待される結果：各 role の入力と出力、共有 artifact、失敗時の巻き戻しを trace に残し、マルチ Agent が単なる会話ではなく工程として動くことを示せる状態です。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
project_goal: what the agent should accomplish and what it must not do
baseline: single-agent loop before adding advanced features
trace_pack: goal, plan, tool calls, observations, memory, evaluation
failure_log: one failed or unsafe run with root cause
成果物：README、実行コマンド、trace スクリーンショット/ログ、次の一手
```

## まとめ

この節でいちばん大事なのは、作品レベルの判断基準を持つことです。

> **マルチ Agent 開発チームプロジェクトの本当の価値は、役割が多くて派手なことではなく、タスク分解、アーティファクトの引き継ぎ、失敗からの巻き戻しを安定したループとして組み立てられるかどうかにあります。**

このループさえしっかり作れれば、このプロジェクトはマルチ Agent システムへの理解をしっかり示すのにとても向いています。

---



## バージョン別の進め方

| バージョン | 目的 | 重点的に作るもの |
|---|---|---|
| 基礎版 | 最小ループを通す | 入力できる、処理できる、出力できる、そして1組の例を残す |
| 標準版 | 見せられるプロジェクトにする | 設定、ログ、エラー処理、README、スクリーンショットを追加する |
| チャレンジ版 | 作品集レベルに近づける | 評価、比較実験、失敗サンプル分析、次のステップの道筋を追加する |

まずは基礎版を完成させましょう。最初から何でも盛り込もうとしないことが大切です。バージョンを1つ上げるたびに、「何が新しくできるようになったか、どう検証したか、まだ何が課題か」を README に書きましょう。

## 練習

1. ワークフローに `ops_agent` を追加して、どの段階に接続すべきか考えてみましょう。
2. 考えてみましょう：なぜマルチ Agent プロジェクトでは「統一されたアーティファクト形式」が「役割が会話できること」より大事なのか？
3. レビュー担当が patch をよく差し戻す場合、どの層を優先して改善すべきでしょうか？
4. このプロジェクトをデモページにするなら、いちばん見せたい完全な トレース はどれですか？
