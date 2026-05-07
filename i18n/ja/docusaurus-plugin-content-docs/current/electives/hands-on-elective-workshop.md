---
title: "E.0 選択モジュール実践ワークショップ"
sidebar_position: 0
description: "選択モジュール A-F を、実行できる証拠パックとしてつなぐハンズオン教材。"
keywords: [選択モジュール, ハンズオン, デプロイ, Python 応用, 古典的 ML, AI 安全性, フロントエンド, プロダクト設計]
---

# E.0 選択モジュール実践ワークショップ

![選択モジュール実践ルートマップ](/img/course/elective-workshop-route-map-ja.png)

:::tip このページの使い方
まず図を見て、それからコードを実行してください。このワークショップは A-F の各選択モジュールを置き換えるものではありません。選択内容を、小さくても再実行でき、確認できる証拠パックに変えるための「つなぎ」のレッスンです。
:::

## 学習目標

- 解きたい問題に合わせて選択モジュールを選ぶ
- 1 つのスクリプトで、デプロイ、Python エンジニアリング、古典的 ML、安全性、フロントエンド証拠、プロダクト判断を体験する
- 概念を読むだけでなく、生成された CSV、JSON、HTML、Markdown を確認する
- 「実行 -> 確認 -> 修正 -> 記録」の流れに慣れる
- 自分の選択プロジェクトを作品集に入れるためのチェックリストを作る

---

## 選択モジュールは何のためにあるのか

選択モジュールは、主コースのあとにただ読む補足ではありません。プロジェクトで具体的な困りごとが出たときに戻ってくる、専門別の道具箱です。

| モジュール | 使うタイミング | 実践で残すもの |
|---|---|---|
| Module A: C++ とモデルデプロイ | モデルは動くが、推論遅延、メモリ、サービスコストが問題になったとき | デプロイスコア表とリリースメモ |
| Module B: Python 上級トピック | 試作コードが増えて、重複や見通しの悪さが目立ってきたとき | デコレータ、ストリーム、非同期、registry を使ったパイプライン |
| Module C: 古典的 ML 補足アルゴリズム | 小-中規模データで信頼できる baseline が必要なとき | baseline 予測と精度レポート |
| Module D: AI セキュリティとレッドチーム | プロンプト、ツール、検索、メモリ経由の攻撃や誤用が心配なとき | レッドチーム回帰レポート |
| Module E: Web フロントエンド基礎速習 | ユーザーが AI 機能を操作し、結果を理解する画面が必要なとき | 静的ダッシュボード、または最小限の対話ページ |
| Module F: AI プロダクト設計の考え方 | 次に何を作るべきか判断したいとき | 優先度付きプロダクトキャンバス |

初心者は、まずこの順番で十分です。

1. 主に取り組むモジュールを 1 つ選ぶ。
2. 小さな例を実行する。
3. 他の人が確認できるファイルを 1 つ残す。
4. 失敗ケースを 1 つ、次のアクションを 1 つ記録する。

---

## 証拠パックの流れ

![選択実践の証拠パックパイプライン](/img/course/elective-workshop-evidence-pipeline-ja.png)

このワークショップでは、概念を必ずファイルに落とします。実際のエンジニアリングでは、成果は説明だけでなく証拠で判断されるからです。

- 再実行できるコマンド
- 出力された表やレポート
- 後から比較できる指標
- 回帰テストにできる失敗ケース
- 何が起きたかを説明する短い README

下のコードは、このフォルダを生成します。

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

### 実行前に知っておきたい用語

- **Artifact（成果物・証拠ファイル）**: コードが何をしたかを示すファイルです。CSV レポートや HTML ダッシュボードが該当します。
- **Baseline（基準結果）**: 最初に比較対象として置くシンプルな結果です。
- **Regression case（回帰ケース）**: 同じ不具合が戻ってこないように残す失敗サンプルです。
- **RICE**: プロダクト優先度の式です。Reach x Impact x Confidence / Effort で計算します。
- **Readiness score（準備度スコア）**: ここでは、証拠パックを議論できる状態かどうかをざっくり見るための総合スコアです。

---

## ワークショップを実行する

![選択実践コードの実行順序図](/img/course/elective-workshop-code-execution-sequence-ja.png)

### 空の作業フォルダを作る

```bash
mkdir elective-workshop
cd elective-workshop
```

### `elective_workshop.py` を作る

下のコードを `elective_workshop.py` という名前で保存します。

この例は Python 標準ライブラリだけで動きます。追加 SDK のインストールは不要です。Python 3.10+ を想定し、ローカルでは Python 3.13 で確認しています。

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

### 実行する

```bash
python3 elective_workshop.py
```

期待される出力：

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

## エンジニアとして結果を読む

### Module A: デプロイはトレードオフで決まる

`elective_workshop_run/outputs/module_a_deployment_score.csv` を開きます。

3 つのデプロイ候補が見えるはずです。スクリプトは `quantized-int8` を選びます。遅延とメモリの目標を満たしつつ、許容できる精度を保っているからです。

操作のヒント：実プロジェクトで精度低下を許容できない場合は、`accuracy_floor` を上げて再実行してください。最適候補が変わることがあります。

### Module B: Python 応用は流れを追跡しやすくするために使う

`elective_workshop_run/outputs/module_b_python_trace.json` を開きます。

デコレータは処理時間を記録し、ジェネレータは行を 1 つずつ整えます。Python 応用の価値は派手な文法ではなく、観察しやすく再利用しやすいパイプラインを作ることです。

### Module C: 古典的 ML は baseline を作る

`elective_workshop_run/outputs/module_c_knn_predictions.csv` を開きます。

KNN の例では距離にもとづいて投票します。実プロジェクトでは、この baseline が次の判断に役立ちます。

- シンプルな特徴量だけでデータは分けられるか
- 重いモデルを使う価値が本当にあるか
- どのサンプルが最初に誤分類されるか

### Module D: 安全性では失敗ケースを残す

`elective_workshop_run/reports/failure_cases.md` を開きます。

ツール呼び出しに関するケースが 1 つ、意図的に失敗するようにしてあります。これは学習上の失敗ではありません。失敗を記録し、ガードレールを決め、後で回帰ケースとして再実行することが目的です。

### Module E: フロントエンド証拠は結果を確認しやすくする

ブラウザで `elective_workshop_run/outputs/module_e_dashboard.html` を開きます。

このページは静的ですが、大切な考え方を示しています。ユーザーが必要としているのは、バックエンドログだけではなく、読み取れる画面です。

### Module F: プロダクト思考は次の一手を決める

`elective_workshop_run/outputs/module_f_product_canvas.md` を開きます。

RICE スコアは優先度判断を明示します。数字に納得できないなら、数字を変えてもう一度ランキングを出します。直感だけで決めるより、すでに一歩前進です。

---

## よくあるエラーと直し方

![選択実践のよくあるエラー対処ループ](/img/course/elective-workshop-debug-loop-ja.png)

| 症状 | よくある原因 | 対処 |
|---|---|---|
| `python3: command not found` | 端末では `python` が Python 3 を指している場合がある | `python --version` を確認し、Python 3 なら `python elective_workshop.py` を使う |
| 出力フォルダがない | スクリプトが最後まで終わっていない、または別フォルダで実行した | 端末出力を確認し、`pwd` で現在地を見る |
| CSV や JSON が空 | 関数が書き込み前に戻っている | 書き込み直前に `print(scored_rows)` や `print(trace)` を入れて確認する |
| レッドチーム結果に失敗がある | 模擬ガードレールが危険なツール実行を許可している | `observed` を `ask_for_confirmation` に変えて再実行し、`failure_cases` が 0 になるか確認する |
| ダッシュボードが開けない | 別のファイルパスを開いている | `elective_workshop_run/outputs/module_e_dashboard.html` をブラウザで直接開く |

---

## 自分の選択プロジェクトに発展させる

![選択実践の作品集証拠パック](/img/course/elective-workshop-portfolio-pack-ja.png)

次のどれか 1 つを選んで改造してみましょう。

- **デプロイルート**: Module A の仮の候補を、自分のモデルの遅延、メモリ、精度に置き換える。
- **Python エンジニアリングルート**: Module B のサンプルチケットを、実際の API、RAG、バッチ処理の記録に置き換える。
- **古典的 ML ルート**: Module C の小さな KNN データセットを、自分の CSV に置き換える。
- **安全性ルート**: Module D のケースを、実際の prompt injection、retrieval injection、tool misuse、memory leakage テストに置き換える。
- **フロントエンドルート**: Module E のダッシュボードを、生成した JSON や CSV を読むページにする。
- **プロダクトルート**: Module F の案を実際のプロダクト選択肢に置き換え、RICE の数字を誰かと議論する。

### 作品集チェックリスト

選択プロジェクトを完了と呼ぶ前に、次を確認してください。

- 再実行コマンドを書いた `README.md`
- CSV や JSON など、少なくとも 1 つの指標ファイル
- 少なくとも 1 つの失敗ケースと、次に試す修正
- HTML ダッシュボードやスクリーンショットなど、ユーザー向けの小さな成果物
- なぜその選択ルートを選んだのかの短い説明

---

## まとめ

選択モジュールは、実際の出力につながったときに価値が出ます。このワークショップを終えたら、方向を選び、最小例を実行し、証拠ファイルを確認し、失敗を 1 つ修正し、小さな作品集用の証拠パックにまとめられるはずです。
