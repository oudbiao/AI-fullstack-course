---
title: "13.1 実践：オープンソース LLM を動かしてサービス化する"
description: "environment check、Transformers の初回推論、5-case evaluation、local OpenAI-style API まで、再現できる open-source LLM loop を動かす。"
sidebar:
  order: 2
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM lab, Transformers local inference, vLLM serving, OpenAI compatible API, LLM evaluation"
---
![オープンソース LLM ランタイム運用ループ](/img/course/ch13-open-source-llm-runtime-loop-ja.webp)

このページは、実際に手を動かして走らせるためのラボです。まず極小モデルから始めて、環境確認、推論、評価、API 化、停止までの一連の流れを証拠つきで確認します。デフォルトモデルは回答品質のために選んでいるのではありません。大きなモデルに時間や GPU 代を使う前に、普通のマシンでもコード経路を検証できるようにするためです。

このループが通ったら、`MODEL_ID` を Qwen、Llama、InternLM、ChatGLM などのモデルに差し替えます。Self-LLM はモデルごとの具体的な分岐が強い教材です。このページでは、その前に共通のエンジニアリング骨格を作ります。

## 最終的に作るもの

作業フォルダは、最後に次の形になります。

```text
openllm_lab/
  environment_report.py
  environment_report.txt
  requirements-freeze.txt
  model_decision.md
  run_local_llm.py
  first_run.md
  eval_cases.csv
  eval_openllm.py
  eval_results.csv
  eval_summary.json
  serve_openai_like.py
  gpu_plan.md
  lora_decision.md
  README.md
```

合格ラインは「賢そうな答えが出た」ではありません。次を満たせば、このラボの目的は達成です。

- 環境レポートを再現できる。
- ローカルモデルを読み込み、テキストを生成できる。
- 固定した 5 ケースを繰り返し評価できる。
- `curl` から API を呼び出せる。
- サービスの停止方法と証拠の残し方が分かる。

## 0. モデルのルートを選ぶ

**スモークテスト：`sshleifer/tiny-gpt2`**

ふつうの PC に向いています。回答品質ではなく、コード経路を確認するためのモデルです。

**小さな実用モデル：`Qwen/Qwen2.5-0.5B-Instruct`**

安定したネットワークとディスクがある環境に向いています。実際のチャットモデルに近い一方、ダウンロードには時間がかかります。

**GPU サービング：7B 級の Instruct モデル**

レンタル GPU、または十分なローカル VRAM がある場合に使います。先に小さなループを通し、その後 vLLM に進みます。

最初は必ずスモークテストから始めます。いきなり大きなモデルをダウンロードしないでください。

まず `model_decision.md` を書きます。

```md
# Model Decision

## Task

コース知識アシスタント。まずローカルモデル実行経路を検証する。

## Selected model

- Smoke test: sshleifer/tiny-gpt2
- Next model: Qwen/Qwen2.5-0.5B-Instruct

## License and source

- Source: Hugging Face model page
- License check: real deployment の前に model card を読む

## Runtime

- First run: Transformers
- GPU server candidate: vLLM

## Rejected for now

- 7B model: tiny と 0.5B の loop に evidence が出るまで待つ
- Fine-tuning: fixed eval cases が繰り返し失敗するまで待つ
```

Self-LLM は後で具体モデルの道筋を見るときに使えますが、この decision card は自分の project に残します。

## 1. プロジェクト環境を作る

```bash
mkdir openllm_lab
cd openllm_lab

python -m venv .venv
source .venv/bin/activate

python -m pip install -U pip
python -m pip install "torch" "transformers>=4.41" "accelerate" "safetensors" "sentencepiece" "fastapi" "uvicorn"
python -m pip freeze > requirements-freeze.txt
```

`torch` のインストールに失敗した場合は、PyTorch 公式サイトで OS とアクセラレータに合うコマンドを選びます。この手順は飛ばさないでください。以降のモデル読み込みはすべてここに依存します。

`requirements-freeze.txt` は暗記するためのものではありません。この実行がどの package environment で起きたかを説明する証拠です。

## 2. 環境チェックを書く

`environment_report.py` を作ります。

```python
import platform
import shutil
import subprocess
from pathlib import Path


def run_optional(command):
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"not available: {exc!r}"


lines = [
    f"python_platform: {platform.platform()}",
    f"python_version: {platform.python_version()}",
]

try:
    import torch

    lines.extend(
        [
            f"torch_version: {torch.__version__}",
            f"cuda_available: {torch.cuda.is_available()}",
            f"cuda_device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}",
            f"mps_available: {getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()}",
        ]
    )
except Exception as exc:
    lines.append(f"torch_check_failed: {exc!r}")

disk = shutil.disk_usage(".")
lines.append(f"disk_free_gb: {round(disk.free / 1024**3, 2)}")
lines.append("nvidia_smi:")
lines.append(run_optional(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]))

report = "\n".join(lines) + "\n"
Path("environment_report.txt").write_text(report, encoding="utf-8")
print(report)
```

実行します。

```bash
python environment_report.py
```

読み方は次の通りです。

- `cuda_available: True` なら NVIDIA GPU が使えます。
- `mps_available: True` なら Apple Silicon の MPS が使える可能性があります。
- どちらも `False` でも、デフォルトの極小モデルなら問題ありません。
- `environment_report.txt` は必ず残す証拠です。

## 3. ローカル推論を書く

`run_local_llm.py` を作ります。

```python
import argparse
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device():
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = pick_device()
    kwargs = {"trust_remote_code": True}
    if device == "cuda":
        kwargs["device_map"] = "auto"
        kwargs["torch_dtype"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if device != "cuda":
        model.to(device)
    model.eval()
    return tokenizer, model, device


def build_inputs(tokenizer, prompt, device):
    messages = [{"role": "user", "content": prompt}]
    if getattr(tokenizer, "chat_template", None):
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(device)
        return {"input_ids": input_ids}, input_ids.shape[-1]

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    return inputs, inputs["input_ids"].shape[-1]


def generate_once(tokenizer, model, device, prompt, max_new_tokens=80):
    inputs, input_length = build_inputs(tokenizer, prompt, device)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    started = time.time()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=pad_id,
        )
    elapsed = time.time() - started
    new_tokens = output_ids[0][input_length:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text or "(empty output)", elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2"))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", "ローカル LLM ランタイムが何をするものか、1 文で説明してください。"))
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.model)
    answer, elapsed = generate_once(tokenizer, model, device, args.prompt, args.max_new_tokens)

    report = f"""# First local LLM run

model: {args.model}
device: {device}
prompt: {args.prompt}
latency_seconds: {elapsed:.2f}

## Output

{answer}
"""
    Path("first_run.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
```

デフォルトモデルを実行します。

```bash
python run_local_llm.py
```

出力が不自然でも大丈夫です。`tiny-gpt2` はスモークテスト専用です。ここでの合格は、スクリプトがモデルをダウンロードし、重みを読み込み、テキストを生成し、`first_run.md` を書けたことです。

次に、小さな実用モデルを試します。

```bash
MODEL_ID="Qwen/Qwen2.5-0.5B-Instruct" \
PROMPT="ローカル LLM デプロイで環境レポートを残すべき理由を 3 文で説明してください。" \
python run_local_llm.py
```

ダウンロードが遅い場合は、まだモデルを切り替えなくて構いません。先に極小モデルで評価と API サービスまで通してください。

## 4. 5 つの評価ケースを固定する

`eval_cases.csv` を作ります。

```csv
id,prompt,expected_behavior,must_include_any
case_001,デプロイ前にモデルライセンスを確認する理由を説明してください。,ライセンスまたは利用制限に触れる,license|ライセンス|利用|制限
case_002,LoRA の前に固定評価セットを走らせる理由を 1 つ挙げてください。,前後比較に触れる,before|after|比較|評価
case_003,ローカルモデルの初回実行後に何を保存すべきですか。,コマンド、プロンプト、出力、環境のどれかに触れる,command|prompt|output|environment|コマンド|出力|環境
case_004,実験後にレンタル GPU を停止すべき理由は何ですか。,コストまたは課金に触れる,cost|billing|課金|コスト|料金
case_005,ファインチューニング前に RAG を試すべきなのはどんな時ですか。,非公開知識または検索に触れる,private|retrieval|document|knowledge|非公開|検索|知識
```

`eval_openllm.py` を作ります。

```python
import csv
import json
import os
from pathlib import Path

from run_local_llm import generate_once, load_model


model_id = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")
tokenizer, model, device = load_model(model_id)

rows = []
with open("eval_cases.csv", newline="", encoding="utf-8") as file:
    for case in csv.DictReader(file):
        output, elapsed = generate_once(tokenizer, model, device, case["prompt"], max_new_tokens=80)
        output_lower = output.lower()
        keywords = [item.strip().lower() for item in case["must_include_any"].split("|") if item.strip()]
        matched_keywords = [keyword for keyword in keywords if keyword in output_lower]
        passed = bool(matched_keywords)
        rows.append(
            {
                "id": case["id"],
                "prompt": case["prompt"],
                "expected_behavior": case["expected_behavior"],
                "must_include_any": case["must_include_any"],
                "passed": passed,
                "matched_keywords": "|".join(matched_keywords),
                "latency_seconds": round(elapsed, 2),
                "output": output.replace("\n", " "),
            }
        )

with open("eval_results.csv", "w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "model": model_id,
    "device": device,
    "total": len(rows),
    "passed_keyword_check": sum(row["passed"] for row in rows),
}
Path("eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
print(json.dumps(summary, indent=2))
```

実行します。

```bash
python eval_openllm.py
```

これは粗い keyword check です。`tiny-gpt2` は失敗するかもしれませんが、それも有用です。「local で動く」と「task を解ける」は別の主張だからです。実プロジェクトでは `eval_results.csv` を開き、回答を人間が読み、`passed` を本当の合否メモに置き換えます。

評価表は次の3点で読みます。

1. **再実行できるか**
   同じ prompt を、model、runtime、settings を変えた後にも再実行できるか。

2. **失敗を分類できるか**
   missing knowledge、wrong format、wrong language、refusal、high latency のどれか。

3. **次の変更が1要素だけか**
   先に eval set を固定し、その後 model、prompt、RAG、quantization、LoRA のどれか1つだけ変えます。一度に全部変えないでください。

1 回の良いチャットより固定ケースが重要です。モデル、ランタイム、量子化、LoRA の変更を比較できるからです。

## 5. ローカル OpenAI 風 API として提供する

`serve_openai_like.py` を作ります。

```python
import os
import time

from fastapi import FastAPI
from pydantic import BaseModel

from run_local_llm import generate_once, load_model


MODEL_ID = os.environ.get("MODEL_ID", "sshleifer/tiny-gpt2")
tokenizer, model, device = load_model(MODEL_ID)
app = FastAPI(title="Open LLM local lab")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 120


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "device": device}


@app.post("/v1/chat/completions")
def chat_completions(request: ChatRequest):
    prompt = "\n".join(
        f"{message.role}: {message.content}"
        for message in request.messages
        if message.role != "system"
    )
    answer, elapsed = generate_once(tokenizer, model, device, prompt, request.max_tokens)
    return {
        "id": f"local-{int(time.time())}",
        "object": "chat.completion",
        "model": MODEL_ID,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": answer},
                "finish_reason": "stop",
            }
        ],
        "usage": {"latency_seconds": round(elapsed, 2)},
    }
```

起動します。

```bash
uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000
```

別のターミナルを開きます。

```bash
curl http://127.0.0.1:8000/health

curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "ローカル LLM のデプロイチェックリストを 1 つ挙げてください。"}
    ],
    "max_tokens": 80
  }'
```

サーバーを止めます。

```bash
Ctrl+C
```

health のレスポンス、リクエスト JSON、レスポンス JSON を保存します。きれいに止められないサービスは、まだ本番運用できる状態ではありません。

## 6. GPU があるなら vLLM に進む

小さな FastAPI サービスは学習用の骨格であり、高スループットサーバーではありません。NVIDIA GPU がある場合は vLLM を試します。

まず `gpu_plan.md` を書きます。

```md
# GPU 計画

- 目的：小さな instruct model を OpenAI-compatible endpoint として提供する
- 最大予算：今回の実験で使ってよい上限を書く
- 停止予定時刻：インスタンスを止める予定時刻を明記する
- インスタンス：GPU 種類、VRAM、ディスク、リージョン
- アクセス：SSH key を使い、デフォルトでは model API を公開しない
- ローカルへ持ち帰る証拠：environment_report.txt、first_run.md、eval_results.csv、README.md
- 停止証拠：スクリーンショット、またはクラウド事業者の停止記録
```

リモートマシンではまずローカルアドレスに bind し、SSH tunnel でテストします。

```bash
python -m pip install "vllm"
vllm serve Qwen/Qwen2.5-0.5B-Instruct --host 127.0.0.1 --port 8000
```

ローカルマシンで別の terminal を開きます。

```bash
ssh -L 8000:127.0.0.1:8000 user@your-gpu-host
```

その後テストします。

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "vLLM を 1 文で説明してください。"}]
  }'
```

レンタル GPU では、セキュリティ規則を単純に保ちます。

- デフォルトではポートを公開インターネットに出さない。
- まず SSH トンネル、またはプラットフォーム内のプライベートネットワークを使う。
- 起動コマンドと停止コマンドを記録する。
- 証拠をコピーし終えたら、すぐインスタンスを停止する。

## 7. LoRA が必要か判断する

1 つの回答が悪かっただけでファインチューニングしないでください。まず `eval_results.csv` の失敗を分類します。

**非公開知識が足りない**

先に RAG、より良い文書、より良い検索を試します。検索が正しいのに振る舞いがまだ間違う場合だけ、LoRA を考えます。

**形式が不安定**

先にスキーマ、few-shot、パーサを試します。固定ケースで同じ形式失敗が繰り返される場合だけ、LoRA を考えます。

**文体が不安定**

先に system prompt と例示を調整します。多数の例で文体問題が繰り返される場合だけ、LoRA を考えます。

**推論が遅い**

先に小さいモデル、量子化、vLLM を試します。振る舞いは良いのに実行制約に失敗する場合だけ、学習ルートへ進みます。

LoRA が妥当だと判断したら、学習前に次を用意します。

```text
train.jsonl        # high-quality training samples
eval_cases.csv     # fixed eval cases, separate from training data
base_model_note.md # base model, license, version, and reason
```

同時に `lora_decision.md` を書きます。

```md
# LoRA Decision

## Repeated failure

Fixed eval case IDs that keep failing:

## Tried before LoRA

- Prompt/schema:
- RAG/retrieval:
- Smaller or larger model:
- Decoding settings:

## Training data

- Sample count:
- Data owner:
- Privacy check:
- Train/eval split:

## Decision

Current choice: no_lora / prepare_lora / full_finetune_not_allowed

Reason:
```

Self-LLM の LoRA 章は、この地点の後に読むとつながります。すでに環境レポート、ベースモデルの選定、固定評価セット、初回実行の証拠がそろっているからです。

## 8. 症状別に確認する

`environment_report.py` が失敗する場合は、まず Python バージョン、仮想環境が有効か、`torch` が現在の環境に入っているかを確認します。まだモデルを変えないでください。

`run_local_llm.py` のダウンロードが遅い場合は、`sshleifer/tiny-gpt2` のまま最後まで通します。大きなモデルのダウンロードはモデル選択の問題であり、環境、評価、API の練習を止める理由にはなりません。

出力が空の場合は、まず prompt を短くし、`pad_token_id` のエラーがないか確認し、`max_new_tokens` を少なめにして再実行します。

API が起動しない場合は、8000 番ポートが使われていないかを確認し、`uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000` を `openllm_lab` の中で実行しているかを見ます。

vLLM がメモリ不足を報告する場合は、先に小さいモデルや低い並列度を試します。すぐ LoRA に進まないでください。メモリ圧迫は通常、学習では解決しません。

最終検収は証拠パックで判断します。`environment_report.txt`、`first_run.md`、`eval_results.csv`、`eval_summary.json`、API の health/request/response 記録がそろっていることを確認します。

## 9. README を書く

`README.md` を作ります。

````md
# Open LLM Lab

## モデル

- スモークテスト：sshleifer/tiny-gpt2
- 次に試すモデル：Qwen/Qwen2.5-0.5B-Instruct

## 実行

```bash
source .venv/bin/activate
python environment_report.py
python run_local_llm.py
python eval_openllm.py
uvicorn serve_openai_like:app --host 127.0.0.1 --port 8000
```

## 証拠

- environment_report.txt
- requirements-freeze.txt
- model_decision.md
- first_run.md
- eval_cases.csv
- eval_results.csv
- eval_summary.json
- gpu_plan.md
- lora_decision.md

## 停止

Ctrl+C でローカル API を停止します。レンタル GPU は、証拠をコピーし終えたらすぐ停止します。
````

このページを終えると、「オープンソースモデルはデプロイできる」と読んだだけではなくなります。環境 -> モデル -> 出力 -> 評価 -> API -> 停止、という再現可能な 1 本の経路を実際に走らせた状態になります。
