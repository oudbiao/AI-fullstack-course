---
title: "E.0 选修模块实操工作坊"
sidebar_position: 0
description: "把选修模块 A-F 串成一个可运行证据包的跟做式实操课。"
keywords: [选修模块, 实操工作坊, 部署, Python 进阶, 经典机器学习, AI 安全, 前端, 产品设计]
---

# E.0 选修模块实操工作坊

![选修模块实操路线图](/img/course/elective-workshop-route-map.png)

:::tip 使用方式
先看图，再跑代码。这个工作坊不是替代 A-F 各选修模块，而是把它们串起来：让你知道怎样把选修知识做成一个小而完整、可以复跑、可以检查的证据包。
:::

## 学习目标

- 根据要解决的问题选择选修方向
- 用一个脚本串起六类选修主题：部署、Python 工程、经典 ML、安全、前端证据、产品判断
- 检查生成的 CSV、JSON、HTML 和 Markdown 文件，而不是只停留在概念理解
- 练习“运行 -> 检查 -> 修复 -> 记录”的工程节奏
- 为自己的选修项目留下一份可以放进作品集的检查清单

---

## 选修模块到底用来做什么？

选修模块不是“学完主线之后随便看看”的额外内容。它们更像专题工具箱：当项目里出现具体痛点时，你回来补对应能力。

| 模块 | 什么时候该用它 | 实操产出 |
|---|---|---|
| Module A: C++ 与模型部署 | 模型能跑了，但推理延迟、内存或服务成本开始变成问题 | 部署评分表和发布说明 |
| Module B: Python 进阶专题 | 原型代码开始重复、混乱、难维护 | 带装饰器、流式、异步或注册机制的工程管道 |
| Module C: 经典 ML 补充算法 | 中小数据任务需要一个可靠 baseline | baseline 预测结果和准确率报告 |
| Module D: AI 安全与红队测试 | 系统可能被提示词、工具、检索或记忆攻击 | 红队回归测试报告 |
| Module E: Web 前端基础速成 | 用户需要操作并理解 AI 功能 | 静态仪表盘或最小交互页面 |
| Module F: AI 产品设计思维 | 需要判断下一步到底值不值得做 | 产品优先级画布 |

新手可以先记住这条规则：

1. 先选一个主方向。
2. 跑一个小例子。
3. 产出一个别人能检查的文件。
4. 记录一个失败案例和一个下一步动作。

---

## 证据包流程

![选修实操证据包流水线图](/img/course/elective-workshop-evidence-pipeline.png)

在这个工作坊里，每个概念最后都要落到一个文件里。原因很简单：真正的工程成果通常要靠证据判断。

- 有一条可以复跑的命令
- 有一个输出表格或报告
- 有一个以后能对比的指标
- 有一个可以沉淀成回归测试的失败案例
- 有一份简短 README 说明发生了什么

下面的代码会生成这个目录：

```text
elective_workshop_run/
  data/module_tasks.json
  outputs/module_a_deployment_score.csv
  outputs/module_b_python_trace.json
  outputs/module_c_knn_predictions.csv
  outputs/module_d_red_team_report.md
  outputs/module_e_dashboard.html
  outputs/module_f_product_canvas.md
  reports/readiness_score.json
  reports/failure_cases.md
  README.md
```

### 运行前先认识几个术语

- **Artifact（证据文件）**：能证明代码做了什么的文件，例如 CSV 报告或 HTML 仪表盘。
- **Baseline（基线）**：第一个简单可比的结果，用来判断更复杂方案是否真的变好。
- **Regression case（回归案例）**：保留下来的失败样本，防止同一个问题以后悄悄回来。
- **RICE**：产品优先级公式：Reach x Impact x Confidence / Effort。
- **Readiness score（就绪分）**：这里用来粗略汇总证据包是否已经适合拿出来讨论。

---

## 跟着跑完整工作坊

![选修实操代码执行顺序图](/img/course/elective-workshop-code-execution-sequence.png)

### 创建一个干净目录

```bash
mkdir elective-workshop
cd elective-workshop
```

### 创建 `elective_workshop.py`

把下面代码保存为 `elective_workshop.py`。

这个示例只使用 Python 标准库，不需要安装额外 SDK。适合 Python 3.10+，本地已用 Python 3.13 测试通过。

```python title="elective_workshop.py"
from __future__ import annotations

import csv
import html
import json
import math
import shutil
import statistics
import time
from functools import wraps
from pathlib import Path

RUN_DIR = Path("elective_workshop_run")
DATA_DIR = RUN_DIR / "data"
OUTPUT_DIR = RUN_DIR / "outputs"
REPORT_DIR = RUN_DIR / "reports"

MODULE_CARDS = [
    {
        "id": "module-a",
        "name": "C++ and Model Deployment",
        "practice_goal": "choose a deployment candidate from latency, memory, and accuracy evidence",
        "evidence": "outputs/module_a_deployment_score.csv",
    },
    {
        "id": "module-b",
        "name": "Advanced Python Topics",
        "practice_goal": "turn a repeated data-cleaning flow into a traceable pipeline",
        "evidence": "outputs/module_b_python_trace.json",
    },
    {
        "id": "module-c",
        "name": "Supplementary Classic ML Algorithms",
        "practice_goal": "build a tiny KNN baseline and record accuracy",
        "evidence": "outputs/module_c_knn_predictions.csv",
    },
    {
        "id": "module-d",
        "name": "AI Safety and Red Team Testing",
        "practice_goal": "run red-team cases and keep failures as regression tasks",
        "evidence": "outputs/module_d_red_team_report.md",
    },
    {
        "id": "module-e",
        "name": "Web Front-End Basics in Fast Track",
        "practice_goal": "generate a static dashboard that explains the current evidence",
        "evidence": "outputs/module_e_dashboard.html",
    },
    {
        "id": "module-f",
        "name": "AI Product Design Thinking",
        "practice_goal": "prioritize the next product improvement with explicit scores",
        "evidence": "outputs/module_f_product_canvas.md",
    },
]


def reset_workspace() -> None:
    if RUN_DIR.exists():
        shutil.rmtree(RUN_DIR)
    for folder in (DATA_DIR, OUTPUT_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_markdown_table(path: Path, headers, rows) -> None:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_module_a():
    candidates = [
        {"variant": "baseline-fp32", "latency_ms": 118, "memory_mb": 640, "accuracy": 0.912, "ops_risk": 2},
        {"variant": "quantized-int8", "latency_ms": 54, "memory_mb": 310, "accuracy": 0.901, "ops_risk": 3},
        {"variant": "distilled-small", "latency_ms": 43, "memory_mb": 220, "accuracy": 0.872, "ops_risk": 2},
    ]
    latency_target = 80
    memory_target = 450
    accuracy_floor = 0.89

    scored_rows = []
    for item in candidates:
        latency_score = min(latency_target / item["latency_ms"], 1.2) * 35
        memory_score = min(memory_target / item["memory_mb"], 1.2) * 20
        accuracy_score = max(0, (item["accuracy"] - accuracy_floor) / 0.04) * 35
        risk_score = (5 - item["ops_risk"]) * 2
        score = round(latency_score + memory_score + accuracy_score + risk_score, 2)
        scored_rows.append({**item, "deployment_score": score})

    path = OUTPUT_DIR / "module_a_deployment_score.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(scored_rows[0]))
        writer.writeheader()
        writer.writerows(scored_rows)

    best = max(scored_rows, key=lambda row: row["deployment_score"])
    return {
        "best_variant": best["variant"],
        "latency_target_ms": latency_target,
        "p95_latency_ms": statistics.quantiles([row["latency_ms"] for row in scored_rows], n=20)[18],
        "artifact": str(path),
    }


def timed_step(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        return {
            "step": fn.__name__,
            "duration_ms": round((time.perf_counter() - started) * 1000, 3),
            "result": result,
        }

    return wrapper


def clean_rows(rows):
    for row in rows:
        yield {
            "ticket_id": row["ticket_id"].strip().upper(),
            "text": " ".join(row["text"].lower().split()),
            "priority": int(row["priority"]),
        }


@timed_step
def build_python_pipeline_trace():
    raw_rows = [
        {"ticket_id": " a-001 ", "text": " Slow response in edge demo ", "priority": "3"},
        {"ticket_id": " a-002 ", "text": "Need retry when API times out", "priority": "2"},
        {"ticket_id": " a-003 ", "text": " dashboard missing evidence links ", "priority": "1"},
    ]
    cleaned = list(clean_rows(raw_rows))
    batches = [cleaned[index : index + 2] for index in range(0, len(cleaned), 2)]
    return {"rows": len(cleaned), "batches": len(batches), "first_ticket": cleaned[0]}


def run_module_b():
    trace = build_python_pipeline_trace()
    path = OUTPUT_DIR / "module_b_python_trace.json"
    write_json(path, trace)
    return {"pipeline_rows": trace["result"]["rows"], "artifact": str(path)}


def euclidean_distance(left, right):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def predict_knn(train_rows, point, k=3):
    neighbors = sorted(train_rows, key=lambda row: euclidean_distance(row["features"], point))[:k]
    votes = {}
    for row in neighbors:
        votes[row["label"]] = votes.get(row["label"], 0) + 1
    return max(votes, key=votes.get)


def run_module_c():
    train_rows = [
        {"features": [0.1, 0.2], "label": "low_risk"},
        {"features": [0.2, 0.1], "label": "low_risk"},
        {"features": [0.8, 0.7], "label": "high_risk"},
        {"features": [0.9, 0.8], "label": "high_risk"},
        {"features": [0.45, 0.5], "label": "review"},
        {"features": [0.5, 0.45], "label": "review"},
    ]
    test_rows = [
        {"features": [0.12, 0.18], "expected": "low_risk"},
        {"features": [0.82, 0.74], "expected": "high_risk"},
        {"features": [0.48, 0.52], "expected": "review"},
        {"features": [0.88, 0.79], "expected": "high_risk"},
    ]
    predictions = []
    for row in test_rows:
        predicted = predict_knn(train_rows, row["features"])
        predictions.append({**row, "predicted": predicted, "correct": predicted == row["expected"]})

    path = OUTPUT_DIR / "module_c_knn_predictions.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["features", "expected", "predicted", "correct"])
        writer.writeheader()
        writer.writerows(predictions)

    correct = sum(1 for row in predictions if row["correct"])
    accuracy = correct / len(predictions)
    return {"accuracy": accuracy, "correct": correct, "total": len(predictions), "artifact": str(path)}


def run_module_d():
    cases = [
        {"id": "rt-001", "surface": "prompt", "expected": "refuse_internal_policy", "observed": "refuse_internal_policy"},
        {"id": "rt-002", "surface": "retrieval", "expected": "ignore_untrusted_instruction", "observed": "ignore_untrusted_instruction"},
        {"id": "rt-003", "surface": "tool", "expected": "ask_for_confirmation", "observed": "allowed_tool_call"},
        {"id": "rt-004", "surface": "memory", "expected": "do_not_store_secret", "observed": "do_not_store_secret"},
    ]
    rows = []
    failures = []
    for case in cases:
        passed = case["expected"] == case["observed"]
        rows.append([case["id"], case["surface"], case["expected"], case["observed"], "PASS" if passed else "FAIL"])
        if not passed:
            failures.append(case)

    path = OUTPUT_DIR / "module_d_red_team_report.md"
    write_markdown_table(path, ["id", "surface", "expected", "observed", "status"], rows)
    return {"passed": len(cases) - len(failures), "total": len(cases), "failures": failures, "artifact": str(path)}


def run_module_f():
    ideas = [
        {"name": "Deployment evidence checker", "reach": 7, "impact": 8, "confidence": 0.80, "effort": 3},
        {"name": "Async batch runner", "reach": 5, "impact": 7, "confidence": 0.65, "effort": 4},
        {"name": "Red-team regression gate", "reach": 6, "impact": 9, "confidence": 0.75, "effort": 5},
    ]
    for idea in ideas:
        idea["rice"] = round(idea["reach"] * idea["impact"] * idea["confidence"] / idea["effort"], 2)
    ranked = sorted(ideas, key=lambda item: item["rice"], reverse=True)
    rows = [[item["name"], item["reach"], item["impact"], item["confidence"], item["effort"], item["rice"]] for item in ranked]

    path = OUTPUT_DIR / "module_f_product_canvas.md"
    write_markdown_table(path, ["idea", "reach", "impact", "confidence", "effort", "rice"], rows)
    return {"top_feature": ranked[0]["name"], "artifact": str(path)}


def run_module_e(results):
    cards = []
    for module in MODULE_CARDS:
        module_result = results.get(module["id"], {"status": "generated in this step"})
        detail = html.escape(json.dumps(module_result, ensure_ascii=False))
        cards.append(
            f"<section><h2>{html.escape(module['name'])}</h2>"
            f"<p>{html.escape(module['practice_goal'])}</p>"
            f"<code>{html.escape(module['evidence'])}</code>"
            f"<pre>{detail}</pre></section>"
        )
    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Elective Workshop Dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; background: #f7f7fb; }}
    main {{ max-width: 960px; margin: auto; }}
    section {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 16px; margin: 12px 0; }}
    code, pre {{ background: #eef2f7; padding: 4px 6px; border-radius: 4px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <main>
    <h1>Elective Workshop Dashboard</h1>
    {''.join(cards)}
  </main>
</body>
</html>
"""
    path = OUTPUT_DIR / "module_e_dashboard.html"
    path.write_text(page, encoding="utf-8")
    return {"dashboard_cards": len(cards), "artifact": str(path)}


def build_readiness_report(results):
    failure_count = len(results["module-d"]["failures"])
    score_cards = [
        {"area": "deployment", "score": 88 if results["module-a"]["best_variant"] == "quantized-int8" else 72},
        {"area": "python_engineering", "score": 90 if results["module-b"]["pipeline_rows"] >= 3 else 70},
        {"area": "classic_ml_baseline", "score": round(results["module-c"]["accuracy"] * 100, 1)},
        {"area": "safety_regression", "score": 100 - failure_count * 20},
        {"area": "frontend_evidence", "score": 85 if results["module-e"]["dashboard_cards"] == 6 else 70},
        {"area": "product_priority", "score": 86 if results["module-f"]["top_feature"] else 60},
    ]
    readiness = round(sum(item["score"] for item in score_cards) / len(score_cards), 1)
    report = {"readiness_score": readiness, "score_cards": score_cards, "failure_cases": failure_count}
    write_json(REPORT_DIR / "readiness_score.json", report)

    failure_lines = ["# Failure Cases", ""]
    for failure in results["module-d"]["failures"]:
        failure_lines.append(f"- {failure['id']} on {failure['surface']}: expected {failure['expected']}, observed {failure['observed']}")
    if not results["module-d"]["failures"]:
        failure_lines.append("- No failure cases in this run.")
    (REPORT_DIR / "failure_cases.md").write_text("\n".join(failure_lines) + "\n", encoding="utf-8")
    return report


def write_readme(results, readiness_report):
    lines = [
        "# Elective Workshop Evidence Pack",
        "",
        "Run command:",
        "",
        "~~~bash",
        "python elective_workshop.py",
        "~~~",
        "",
        "Generated evidence:",
    ]
    for module in MODULE_CARDS:
        lines.append(f"- {module['id']}: {module['evidence']}")
    lines.extend(
        [
            "- reports/readiness_score.json",
            "- reports/failure_cases.md",
            "",
            f"Readiness score: {readiness_report['readiness_score']}",
            f"Recommended next feature: {results['module-f']['top_feature']}",
        ]
    )
    (RUN_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    reset_workspace()
    write_json(DATA_DIR / "module_tasks.json", MODULE_CARDS)

    results = {}
    results["module-a"] = run_module_a()
    results["module-b"] = run_module_b()
    results["module-c"] = run_module_c()
    results["module-d"] = run_module_d()
    results["module-f"] = run_module_f()
    results["module-e"] = run_module_e(results)

    readiness_report = build_readiness_report(results)
    write_readme(results, readiness_report)

    print("STEP 1: elective paths")
    print(f"modules: {len(MODULE_CARDS)}")
    print("selected_capstone: module-a + module-f")
    print("STEP 2: generated evidence")
    print(f"deployment_best_variant: {results['module-a']['best_variant']}")
    print(f"knn_accuracy: {results['module-c']['accuracy']:.3f} ({results['module-c']['correct']}/{results['module-c']['total']})")
    print(f"red_team_passed: {results['module-d']['passed']}/{results['module-d']['total']}")
    print(f"product_top_feature: {results['module-f']['top_feature']}")
    print(f"failure_cases: {readiness_report['failure_cases']}")
    print("STEP 3: files to inspect")
    print(RUN_DIR / "README.md")
    print(REPORT_DIR / "readiness_score.json")
    print(OUTPUT_DIR / "module_e_dashboard.html")


if __name__ == "__main__":
    main()
```

### 运行

```bash
python3 elective_workshop.py
```

预期输出：

```text
STEP 1: elective paths
modules: 6
selected_capstone: module-a + module-f
STEP 2: generated evidence
deployment_best_variant: quantized-int8
knn_accuracy: 1.000 (4/4)
red_team_passed: 3/4
product_top_feature: Deployment evidence checker
failure_cases: 1
STEP 3: files to inspect
elective_workshop_run/README.md
elective_workshop_run/reports/readiness_score.json
elective_workshop_run/outputs/module_e_dashboard.html
```

---

## 像工程师一样读结果

### Module A：部署是取舍题

打开 `elective_workshop_run/outputs/module_a_deployment_score.csv`。

你会看到三个部署候选方案。脚本选择 `quantized-int8`，因为它在延迟和内存目标内，同时保留了可以接受的准确率。

操作提示：如果你的真实项目不能接受准确率下降，就提高 `accuracy_floor` 再跑一次，最佳方案可能会改变。

### Module B：Python 进阶要让流程更可追踪

打开 `elective_workshop_run/outputs/module_b_python_trace.json`。

装饰器记录耗时，生成器一条条清洗数据。Python 进阶的价值不在炫技，而在让管道更容易观察、复用和维护。

### Module C：经典 ML 给你 baseline

打开 `elective_workshop_run/outputs/module_c_knn_predictions.csv`。

KNN 示例使用距离投票。在真实项目里，这个 baseline 能帮你回答：

- 简单特征是否已经能把数据分开？
- 更重的模型是否真的带来提升？
- 哪些样本最先被分错？

### Module D：安全工作必须保留失败案例

打开 `elective_workshop_run/reports/failure_cases.md`。

其中有一个工具调用相关案例被故意设置为失败。这不是坏事，训练目的就是：记录失败、决定护栏、以后把它作为回归案例反复检查。

### Module E：前端证据让结果更容易被检查

用浏览器打开 `elective_workshop_run/outputs/module_e_dashboard.html`。

这个页面是静态的，但它证明了一个关键点：产品用户需要能读懂的界面，而不只是后端日志。

### Module F：产品思维决定下一步做什么

打开 `elective_workshop_run/outputs/module_f_product_canvas.md`。

RICE 分数把优先级判断显式化。你可以不同意这些数字，但接下来就要改数字、重跑排序。这已经比凭感觉拍板更可靠。

---

## 常见错误与排查

![选修实操常见错误排查闭环图](/img/course/elective-workshop-debug-loop.png)

| 现象 | 常见原因 | 修复方法 |
|---|---|---|
| `python3: command not found` | 你的终端可能使用 `python` 命令 | 先运行 `python --version`，如果指向 Python 3，就用 `python elective_workshop.py` |
| 输出目录不存在 | 脚本没有跑完，或你在别的目录运行了命令 | 看终端输出，并用 `pwd` 确认当前目录 |
| CSV 或 JSON 为空 | 某个函数在写入前提前返回了 | 在写文件前加 `print(scored_rows)` 或 `print(trace)` 定位 |
| 红队结果出现失败 | 模拟护栏允许了有风险的工具调用 | 把 `observed` 改成 `ask_for_confirmation`，重跑后确认 `failure_cases` 变成 0 |
| 仪表盘打不开 | 打开的文件路径不对 | 直接用浏览器打开 `elective_workshop_run/outputs/module_e_dashboard.html` |

---

## 把它改成你的选修项目

![选修实操作品集证据包图](/img/course/elective-workshop-portfolio-pack.png)

选择一条路线继续改：

- **部署路线**：把 Module A 里的虚拟候选方案换成你真实模型的延迟、内存和准确率。
- **Python 工程路线**：把 Module B 的示例工单换成真实 API、RAG 或批处理记录。
- **经典 ML 路线**：把 Module C 的小 KNN 数据集换成你项目里的 CSV。
- **安全路线**：把 Module D 的案例换成真实的 prompt injection、retrieval injection、tool misuse 和 memory leakage 测试。
- **前端路线**：把 Module E 的仪表盘改成读取你生成的 JSON 和 CSV。
- **产品路线**：把 Module F 的想法换成真实产品选项，并和同伴讨论 RICE 数字是否合理。

### 作品集检查清单

在宣布选修项目完成前，确认你有：

- 一个包含复跑命令的 `README.md`
- 至少一个指标文件，例如 CSV 或 JSON
- 至少一个失败案例，以及下一步准备怎么修
- 一个面向用户的轻量产物，例如 HTML 仪表盘或截图
- 一段说明：为什么你选择这个选修方向

---

## 总结

选修模块真正有价值的地方，是把专题能力连接到真实产出。完成这个工作坊后，你应该能选择一个方向、跑通最小例子、检查证据文件、修复一个失败点，并把结果整理成可以展示的小型作品集证据包。
