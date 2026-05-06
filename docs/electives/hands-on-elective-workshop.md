---
title: "Elective Hands-on Workshop"
sidebar_position: 0
description: "A guided practice workshop that connects elective modules A-F into one runnable evidence pack."
keywords: [elective module, hands-on workshop, deployment, advanced Python, classic ML, AI safety, frontend, product design]
---

# Elective Hands-on Workshop

![Elective module hands-on route map](/img/course/elective-workshop-route-map-en.png)

:::tip How to use this page
Read the diagram first, then run the code. This workshop is not a replacement for modules A-F. It is the glue lesson: it shows how to turn elective knowledge into a small, repeatable evidence pack.
:::

## Learning Objectives

- Choose an elective direction based on the problem you want to solve
- Run one script that touches the six elective themes: deployment, Python engineering, classic ML, safety, frontend evidence, and product judgment
- Inspect generated CSV, JSON, HTML, and Markdown artifacts instead of only reading concepts
- Practice the habit of “run -> inspect -> fix -> document”
- Leave with a portfolio-ready checklist for your own elective project

---

## 1. What Are the Electives For?

The elective modules are not extra pages to read only after you finish everything else. They are focused toolboxes. You come back to them when a project exposes a real need.

| Module | When you should use it | Practical output |
|---|---|---|
| Module A: C++ and Model Deployment | Your model works, but inference latency, memory, or serving cost becomes a problem | Deployment score table and release notes |
| Module B: Advanced Python Topics | Your prototype grows into repeated, messy engineering code | Decorated, streaming, async, or registry-based pipeline |
| Module C: Classic ML Algorithms | You need a strong baseline for small or medium data | Baseline predictions and accuracy report |
| Module D: AI Safety and Red Team Testing | Your system can be attacked, misused, or tricked through prompts, tools, retrieval, or memory | Red-team regression report |
| Module E: Web Front-End Basics | Users need to operate and understand the AI feature | Static dashboard or minimal interactive page |
| Module F: AI Product Design Thinking | You need to decide what is worth building next | Prioritized product canvas |

The beginner-friendly rule is:

1. Pick one main module.
2. Run a small example.
3. Produce an artifact someone else can inspect.
4. Record one failure case and one next action.

---

## 2. The Evidence Pack Flow

![Elective workshop evidence pipeline](/img/course/elective-workshop-evidence-pipeline-en.png)

In this workshop, every concept must land in a file. This matters because real engineering work is judged by evidence:

- a command that can be rerun
- an output table or report
- a metric that can be compared later
- a failure case that can become a regression test
- a short README that explains what happened

The code below creates this folder:

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

### Terms You Should Know Before Running

- **Artifact**: a file that proves what your code did, such as a CSV report or HTML dashboard.
- **Baseline**: a simple result used as the first comparison point.
- **Regression case**: a failure sample you keep so the same bug does not silently return.
- **RICE**: a product prioritization formula: Reach x Impact x Confidence / Effort.
- **Readiness score**: a simple combined score used here to summarize whether the evidence pack is ready to discuss.

---

## 3. Run the Full Workshop

![Elective workshop code execution sequence](/img/course/elective-workshop-code-execution-sequence-en.png)

### 3.1 Create a Clean Folder

```bash
mkdir elective-workshop
cd elective-workshop
```

### 3.2 Create `elective_workshop.py`

Copy the code below into a file named `elective_workshop.py`.

This example uses only the Python standard library, so there is no SDK to install. It is designed for Python 3.10+ and was tested locally with Python 3.13.

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

### 3.3 Run It

```bash
python3 elective_workshop.py
```

Expected output:

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

## 4. Read the Results Like an Engineer

### 4.1 Module A: Deployment Is a Trade-off

Open `elective_workshop_run/outputs/module_a_deployment_score.csv`.

You should see three deployment candidates. The script chooses `quantized-int8` because it stays under the latency and memory targets while keeping acceptable accuracy.

Operation tip: if your real project cannot accept the accuracy drop, raise `accuracy_floor` and run again. The best variant may change.

### 4.2 Module B: Advanced Python Should Make Work Traceable

Open `elective_workshop_run/outputs/module_b_python_trace.json`.

The decorator records timing. The generator cleans rows one by one. This is the practical value of advanced Python: not clever syntax, but a pipeline that is easier to observe and reuse.

### 4.3 Module C: Classic ML Gives You a Baseline

Open `elective_workshop_run/outputs/module_c_knn_predictions.csv`.

The KNN example uses distance voting. In a real project, this baseline helps answer:

- Is the dataset already separable with simple features?
- Does a heavier model actually improve the result?
- Which examples are misclassified first?

### 4.4 Module D: Safety Work Must Keep Failure Cases

Open `elective_workshop_run/reports/failure_cases.md`.

One tool-related case intentionally fails. This is not bad for learning. The point is to record the failure, decide the guardrail, and rerun it later as a regression case.

### 4.5 Module E: Frontend Evidence Makes the Result Inspectable

Open `elective_workshop_run/outputs/module_e_dashboard.html` in your browser.

The page is static, but it proves an important idea: product users need a readable surface, not only backend logs.

### 4.6 Module F: Product Thinking Decides the Next Step

Open `elective_workshop_run/outputs/module_f_product_canvas.md`.

The RICE score makes your prioritization explicit. You can disagree with the numbers, but then you must change them and rerun the ranking. That is already better than deciding by instinct.

---

## 5. Common Errors and How to Debug Them

![Elective workshop debugging loop](/img/course/elective-workshop-debug-loop-en.png)

| Symptom | Likely cause | Fix |
|---|---|---|
| `python3: command not found` | Your terminal uses `python` instead of `python3` | Run `python --version`, then use `python elective_workshop.py` if it points to Python 3 |
| Output folder is missing | The script did not finish or you ran it from a different folder | Check the terminal output and run `pwd` to confirm the current directory |
| CSV or JSON is empty | A function returned before writing rows | Add `print(scored_rows)` or `print(trace)` before the write step |
| Red-team result has a failure | The simulated guardrail allowed a risky tool action | Change `observed` to `ask_for_confirmation`, rerun, and confirm `failure_cases` becomes 0 |
| Dashboard will not open | You opened the wrong file path | Open `elective_workshop_run/outputs/module_e_dashboard.html` directly in a browser |

---

## 6. Turn This Into Your Own Elective Project

![Elective workshop portfolio evidence pack](/img/course/elective-workshop-portfolio-pack-en.png)

Choose one route:

- **Deployment route**: replace the fake candidates in Module A with real latency, memory, and accuracy numbers from your model.
- **Python engineering route**: replace the sample tickets in Module B with real API, RAG, or batch-processing records.
- **Classic ML route**: replace the tiny KNN dataset in Module C with a CSV from your own project.
- **Safety route**: replace the Module D cases with your actual prompt injection, retrieval injection, tool misuse, and memory leakage tests.
- **Frontend route**: turn the Module E dashboard into a page that reads your generated JSON and CSV.
- **Product route**: replace Module F ideas with real product options and discuss the RICE numbers with a teammate.

### Portfolio Checklist

Before calling your elective project done, make sure you have:

- A `README.md` with one command that reruns the project
- At least one metric file, such as CSV or JSON
- At least one failure case and the fix you would try next
- A small user-facing artifact, such as an HTML dashboard or screenshot
- A short explanation of why you chose this elective route

---

## Summary

The electives become valuable when you attach them to a real output. After this workshop, you should be able to pick a direction, run a minimal example, inspect the artifacts, fix one failure, and turn the result into a small portfolio-ready evidence pack.
