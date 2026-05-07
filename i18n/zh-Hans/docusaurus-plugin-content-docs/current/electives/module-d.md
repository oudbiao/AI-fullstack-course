---
title: "E.D AI 安全与红队测试"
sidebar_position: 4
description: "AI 红队测试的简明实操指南：建模资产、攻击面、失败类别、修复和回归检查。"
keywords: [AI 安全, 红队测试, 威胁建模, eval, jailbreak, prompt injection, guardrails]
---

# E.D AI 安全与红队测试

红队测试不是“随便试一个吓人的 Prompt”。它是一个循环：建模威胁，运行样例，记录失败，修复系统，再把失败样例保留为回归测试。

## 先看循环

![AI 安全红队循环图](/img/course/elective-ai-security-red-team-loop.png)

![AI 安全威胁建模与回归集图](/img/course/elective-ai-security-threat-regression-map.png)

先看攻击面，不要先背攻击名：Prompt、检索、工具、记忆和外部动作。

## 跑一个最小红队评估器

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

预期输出：

```text
prompt PASS
retrieval PASS
tool FAIL
failure_count: 1
regression_cases: ['tool']
```

重点不是隐藏失败，而是保留它、修复它、再运行它。

## 实操检查表

| 步骤 | 动作 | 证据 |
|---|---|---|
| 1 | 定义资产 | 用户数据、工具、记忆、系统指令 |
| 2 | 定义攻击面 | Prompt、文档、检索、工具调用、记忆 |
| 3 | 运行样例 | PASS / FAIL 表 |
| 4 | 修复并重跑 | 回归报告 |

## 通过标准

你能维护一份红队样例文件，解释一个失败攻击面，提出一个 guardrail，并在修复后重跑样例，就算通过本选修。
