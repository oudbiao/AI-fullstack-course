---
title: "E.D AI 安全与红队测试"
sidebar_position: 4
description: "跑一个最小 AI 红队闭环：定义攻击面、记录失败、加防护、保留回归用例。"
keywords: [AI safety, red teaming, threat model, eval, jailbreak, prompt injection, guardrails]
---

# E.D AI 安全与红队测试

红队测试是可重复流程，不是随便试一个吓人的 prompt。你要定义攻击面、跑用例、记录失败、修复系统，再用同一批用例回归。

## 先看闭环

![AI 安全红队闭环图](/img/course/elective-ai-security-red-team-loop.webp)

![AI 安全威胁建模与回归集图](/img/course/elective-ai-security-threat-regression-map.webp)

先从攻击面开始，而不是从攻击名字开始：prompt、检索、工具、记忆和外部动作。

## 准备内容

- 一个要测试的 AI 功能
- 功能涉及的攻击面列表
- 一个保存失败用例作为回归测试的地方

## 运行修复前后评估器

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

预期输出：

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

工具调用失败不是丢脸的结果；它现在变成了保护未来版本的回归测试。

## 实用清单

| 步骤 | 动作 | 证据 |
|---|---|---|
| 1 | 定义资产 | 用户数据、工具、记忆、系统提示 |
| 2 | 定义攻击面 | Prompt、文档、检索、工具调用、记忆 |
| 3 | 运行用例 | PASS / FAIL 表 |
| 4 | 修复并重跑 | 回归报告 |

## 通过标准

能保存一份红队用例文件，解释一个失败攻击面，提出一个防护方案，并在修复后重跑用例，就算通过本选修。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
product_question: user problem, workflow, value metric, and risk boundary
experiment: hypothesis, smallest test, metric, and decision rule
artifact: feature spec, prototype note, user story, or evaluation result
failure_check: building demos without measuring value or ignoring user workflow
Expected_output: AI product decision note that can guide implementation
```
