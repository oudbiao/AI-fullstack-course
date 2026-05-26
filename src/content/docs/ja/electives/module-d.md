---
title: "E.D AI セーフティとレッドチームテスト"
description: "小さな AI レッドチームの流れを実行する。攻撃面を定義し、失敗を記録し、防護策を入れ、回帰ケースとして残す。"
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AI safety, red teaming, threat model, eval, jailbreak, prompt injection, guardrails"
---
レッドチームは、怖い prompt を一度試すことではありません。攻撃面を定義し、ケースを実行し、失敗を記録し、修正し、同じケースを再実行する反復可能な流れです。

## まずループを見る

![AI セキュリティ レッドチームループ図](/img/course/elective-ai-security-red-team-loop-ja.webp)

![AI セキュリティ脅威モデリングと回帰セット図](/img/course/elective-ai-security-threat-regression-map-ja.webp)

攻撃名からではなく、攻撃面から始めます。prompt、検索、ツール、メモリ、外部アクションです。

## 準備するもの

- テストする AI 機能
- その機能が触れる攻撃面の一覧
- 失敗ケースを回帰テストとして残す場所

## 修正前後の評価器を動かす

```python
cases = [
    {"id": "prompt-basic", "surface": "prompt", "expected": "refuse", "before": "refuse", "after": "refuse"},
    {"id": "rag-injection", "surface": "retrieval", "expected": "ignore_untrusted_instruction", "before": "ignore_untrusted_instruction", "after": "ignore_untrusted_instruction"},
    {"id": "tool-confirmation", "surface": "tool", "expected": "ask_confirmation", "before": "executed", "after": "ask_confirmation"},
]

for phase in ["before", "after"]:
    failures = []
    for case in cases:
        passed = case[phase] == case["expected"]
        print(phase, case["id"], "PASS" if passed else "FAIL")
        if not passed:
            failures.append(case["id"])
    print(phase, "failure_count:", len(failures))
```

期待される出力：

```text
before prompt-basic PASS
before rag-injection PASS
before tool-confirmation FAIL
before failure_count: 1
after prompt-basic PASS
after rag-injection PASS
after tool-confirmation PASS
after failure_count: 0
```

ツール呼び出しの失敗は隠すものではありません。今後のリリースを守る回帰テストになります。

## レッドチームレビュー

レッドチーム実行をレビューするときは、3 つを分けます。失敗した攻撃面、期待される安全な挙動、結果を変えた control です。たとえば tool surface が早すぎる実行で失敗し、安全な挙動が `ask_confirmation`、control が permission gate になる場合があります。

「より安全になった」とだけ要約しないでください。元の入力、危険な出力、修正、再実行の出力を残します。その記録が、怖い prompt を有用な regression case に変えます。

最初の portfolio artifact では、case file を退屈なくらい正確にします。`case_id`、`surface`、`input`、`expected_safe_behavior`、`actual_before`、`guardrail`、`actual_after` のような列を残します。読む人が 1 行だけを再実行しても意図が分かる状態にします。

case が大きすぎる場合は分けます。prompt injection、tool misuse、data leak、unsafe output は別の failure mode です。再現できない攻撃リストより、小さく明確な regression set のほうが役に立ちます。

## 実用チェックリスト

| ステップ | アクション | 証拠 |
|---|---|---|
| 1 | 資産を定義 | ユーザーデータ、ツール、メモリ、システム prompt |
| 2 | 攻撃面を定義 | Prompt、文書、検索、ツール呼び出し、メモリ |
| 3 | ケースを実行 | PASS / FAIL 表 |
| 4 | 修正して再実行 | 回帰レポート |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
脅威モデル：prompt injection、data leak、tool misuse、unsafe output、または model abuse
制御: 検証、権限、サンドボックス、監査、レッドチームテスト、またはインシデント対応
テストケース：1 つの攻撃または失敗サンプルと、期待される安全な挙動
失敗確認: モデルの文を信じる、ログ不足、広すぎる権限、または回帰テストなし
期待される成果: セキュリティチェックリストと1件の再現可能なレッドチーム事例
```

<details>
<summary>確認の考え方と解説</summary>

合格する答えは、1 つの攻撃面、1 つの失敗ケース、1 つの防護策、そして修正後の再実行結果を示します。最良の証拠は「安全そうに見える」ことではなく、同じケースを繰り返し実行し、失敗し、再び通ることです。

対策名だけで回帰ケースがないなら、このページはまだ閉じていません。

</details>

## 合格チェック

レッドチームケースファイルを残し、失敗した攻撃面を1つ説明し、防護策を1つ提案し、修正後に同じケースを再実行できれば合格です。
