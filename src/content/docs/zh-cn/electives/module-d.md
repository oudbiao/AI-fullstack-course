---
title: "E.D AI 安全与红队测试"
description: "跑一个最小 AI 红队闭环：定义攻击面、记录失败、加防护、保留回归用例。"
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AI safety, red teaming, threat model, eval, jailbreak, prompt injection, guardrails"
---
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

## 红队复盘

复盘一次红队测试时，把三件事分开：失败的攻击面、预期安全行为、改变结果的控制措施。比如工具攻击面失败于过早执行，预期行为是 `ask_confirmation`，控制措施是权限确认。

不要只总结“现在更安全了”。保留原始输入、不安全输出、修复方式和重跑输出。这个记录才会把吓人的 prompt 变成有用的回归用例。

交付检查时，用表格保存 `case_id`、`surface`、`input`、`expected_safe_behavior`、`actual_before`、`guardrail` 和 `actual_after`。别人应该能只拿一行用例重跑，不需要猜你的测试意图。

如果某个用例太大，就拆开。Prompt 注入、工具滥用、数据泄露和不安全输出是不同失败模式。清楚的小回归集，比无法复现的攻击清单更有价值。

用例通过后不要删除它。把它移入回归集，并在每次发布前重跑。安全工作最重要的不是一次性发现问题，而是防止同类问题悄悄回来。

交付时至少保留一个失败前后的对照案例。它应包含原始攻击、错误响应、防护规则和修复后响应。这样读者能看到系统到底变安全在哪里。

如果修复只是在当前 prompt 上凑巧有效，也要标出来。真正的红队价值来自可复现、可回归、可解释，而不是一次性的“骗过或没骗过”。

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
威胁模型：prompt 注入、数据泄露、工具滥用、不安全输出或模型滥用
控制：验证、权限、沙箱、审计、红队测试或事件响应
测试用例：一个攻击或失败样本及预期的安全行为
失败检查：轻信模型文本、缺少日志、权限过大或没有回归测试
期望产出：安全清单加一个可复现的红队案例
```

<details>
<summary>检查思路与讲解</summary>

一个合格答案会说明一个攻击面、一个失败用例、一个防护措施，以及修复后同样用例的回归结果。最好的证据不是“看起来安全”，而是能重复运行、重复失败、再重复通过。

如果只写策略名而没有回归案例，这一页还没有闭环。

</details>
