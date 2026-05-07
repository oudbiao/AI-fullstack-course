---
title: "E.D AI セーフティとレッドチームテスト"
sidebar_position: 4
description: "AI レッドチームの短い実践ガイド。モデル資産、攻撃面、失敗カテゴリ、修正、回帰チェックを扱います。"
keywords: [AI safety, red teaming, threat model, eval, jailbreak, prompt injection, guardrails]
---

# E.D AI セーフティとレッドチームテスト

レッドチームは「危なそうなプロンプトを 1 回試すこと」ではありません。脅威を整理し、ケースを実行し、失敗を記録し、システムを直し、その失敗を回帰テストとして残すループです。

## まずループを見る

![AI セキュリティのレッドチームループ図](/img/course/elective-ai-security-red-team-loop-ja.png)

![AI セキュリティの脅威モデリングと回帰セット図](/img/course/elective-ai-security-threat-regression-map-ja.png)

攻撃名から始めるより、面から始めます。プロンプト、検索、ツール、メモリ、外部アクションです。

## 最小のレッドチーム評価器を動かす

```python
cases = [
    {"surface": "prompt", "expected": "refuse", "observed": "refuse"},
    {"surface": "retrieval", "expected": "ignore_untrusted_instruction", "observed": "ignore_untrusted_instruction"},
    {"surface": "tool", "expected": "ask_confirmation", "observed": "executed"},
]

failures = []
for case in cases:
    passed = case["expected"] == case["observed"]
    print(case["surface"], "PASS" if passed else "FAIL")
    if not passed:
        failures.append(case["surface"])

print("failure_count:", len(failures))
print("regression_cases:", failures)
```

期待される出力:

```text
prompt PASS
retrieval PASS
tool FAIL
failure_count: 1
regression_cases: ['tool']
```

大事なのは失敗を隠すことではありません。失敗を残し、直し、もう一度実行することです。

## 実践チェックリスト

| Step | 作業 | 証拠 |
|---|---|---|
| 1 | 守る資産を定義する | ユーザーデータ、ツール、メモリ、システム指示 |
| 2 | 攻撃面を定義する | プロンプト、文書、検索、ツール呼び出し、メモリ |
| 3 | ケースを実行する | PASS / FAIL 表 |
| 4 | 修正して再実行する | 回帰レポート |

## 合格チェック

レッドチームのケースファイルを保ち、失敗した面を 1 つ説明し、ガードレールを提案し、修正後に同じケースを再実行できれば合格です。
