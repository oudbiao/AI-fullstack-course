---
title: "13 オープンソース LLM のデプロイと微調整"
description: "オープンソース LLM を選び、動かし、サービス化し、評価し、軽量に微調整する流れを、再現可能な環境メモと実行証拠で学ぶ。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "オープンソース LLM, ローカル LLM デプロイ, vLLM, SGLang, Transformers, LoRA, モデル serving"
---
![オープンソース LLM ランタイムデプロイループ](/img/course/ch13-open-source-llm-runtime-loop-ja.webp)

第 13 章では、オープンソース LLM の利用をエンジニアリングワークフローとして扱います。目的はモデル名を集めることではありません。モデルを選び、明確な環境で動かし、安定したインターフェースで公開し、挙動を評価し、他のエンジニアが再現できる証拠を残すことです。

[Datawhale Self-LLM](https://github.com/datawhalechina/self-llm) は広いモデルと事例のリファレンスとして使えます。この章では、それを学習コースとして進めるために、選択肢を絞り、手順と合格基準を明確にします。

## この章の位置づけ

ここまでで、LLM、RAG、Agent のワークフローを作れるようになっています。この章では別の問いを扱います。

> モデルが cloud API ではなく、自分で download、host、quantize、serve、fine-tune するものになったら何が変わるか？

オープンソース LLM の作業は、多くの場合システム作業です。ハードウェア、driver、model files、runtime、API contract、logs、evaluation cases、rollback を管理する必要があります。

## デプロイループ

| 手順 | 決めること | 残す証拠 |
|---|---|---|
| 選定 | model family、license、size、context、language、modality | model card、license note、選定理由 |
| 準備 | GPU/CPU、CUDA、PyTorch、disk、network、secrets | environment report、cost estimate |
| 実行 | Transformers、Ollama、llama.cpp、vLLM、SGLang、vendor runtime | exact command、model path、first response |
| サービス化 | OpenAI-compatible API、internal SDK、batch script | request/response sample、error path |
| 評価 | fixed prompts、RAG cases、safety cases、latency/cost | eval table、failure notes |
| 適応 | Prompt、RAG、quantization、LoRA、full fine-tune | decision memo、adapter artifact、before/after |
| 公開 | README、container、runbook、monitoring、shutdown plan | deployment checklist、rollback notes |

## Learning Order And Task List

| 順番 | やること | 止める証拠 |
|---|---|---|
| 1 | 1つのモデルと1つの runtime を選ぶ | model/runtime decision |
| 2 | 環境を確認する | Python、PyTorch、CUDA または CPU 状態 |
| 3 | ローカル推論を1回動かす | prompt、output、command、model version |
| 4 | API または script として包む | 再実行できる request/response |
| 5 | 小さな評価セットを動かす | 5つ以上の prompt と pass/fail notes |
| 6 | fine-tune が必要か判断する | no tuning、LoRA、full training の理由 |
| 7 | runbook をまとめる | README、commands、cost、limits、shutdown |

この stage deliverables は、実行できる runbook、environment report、5-case evaluation table、model/runtime decision memo、shutdown または rollback を含む README です。

## 最初に動かすループ：モデル runbook を作る

この offline script はモデルを download しません。GPU を借りる前、大きなモデルを落とす前に必要な計画習慣を練習します。

`ch13_open_llm_runbook.py` を作り、Python 3.10 以降で実行します。

```python
import json
from pathlib import Path


project = {
    "task": "course assistant",
    "privacy": "local documents may be private",
    "expected_users": "small internal group",
    "latency_target_seconds": 4,
    "available_vram_gb": 24,
    "needs_fine_tuning": False,
}


def choose_runtime(info):
    if info["available_vram_gb"] >= 24:
        return {
            "runtime": "vLLM or SGLang",
            "model_size": "7B to 14B instruct model",
            "why": "enough VRAM for a practical server and OpenAI-compatible API",
        }
    if info["available_vram_gb"] >= 8:
        return {
            "runtime": "Transformers or Ollama",
            "model_size": "1B to 7B instruct model, possibly quantized",
            "why": "simpler setup and acceptable for a small lab",
        }
    return {
        "runtime": "CPU quantized runtime or cloud API fallback",
        "model_size": "small quantized model",
        "why": "local GPU memory is too limited for serving a larger model",
    }


def choose_adaptation(info):
    if info["needs_fine_tuning"]:
        return "prepare a LoRA experiment with a fixed eval set first"
    if info["privacy"] == "local documents may be private":
        return "try RAG before fine-tuning"
    return "start with prompt and decoding settings"


plan = {
    "project": project["task"],
    "runtime_choice": choose_runtime(project),
    "adaptation_choice": choose_adaptation(project),
    "minimum_evidence": [
        "environment report",
        "model card and license note",
        "first prompt/output",
        "five-case evaluation table",
        "latency and memory note",
        "shutdown or rollback step",
    ],
}

Path("open_llm_runbook.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
print(json.dumps(plan, indent=2))
```

期待される出力：

```text
{
  "project": "course assistant",
  "runtime_choice": {
    "runtime": "vLLM or SGLang",
    "model_size": "7B to 14B instruct model",
    "why": "enough VRAM for a practical server and OpenAI-compatible API"
  },
  "adaptation_choice": "try RAG before fine-tuning",
  "minimum_evidence": [
    "environment report",
    "model card and license note",
    "first prompt/output",
    "five-case evaluation table",
    "latency and memory note",
    "shutdown or rollback step"
  ]
}
```

この出力が deploy runbook の原型です。大きなモデルを動かす前に、VRAM、privacy、task、fine-tuning flag を変更してください。

## Runbook を一段ずつ読む

| コード部分 | 何を意味するか | まず変えるところ |
|---|---|---|
| `project = {...}` | project constraint card です。「model を動かしたい」を hardware、privacy、users、latency、tuning requirements に分けます。 | `task`、`privacy`、`available_vram_gb`、`needs_fine_tuning` を変えます。 |
| `choose_runtime(info)` | runtime decision rule です。memory を確認する前に GPU を借りたり model を download したりするのを防ぎます。 | 実際の instance または local machine が分かってから VRAM threshold を調整します。 |
| `choose_adaptation(info)` | fine-tuning gate です。private knowledge は普通、training の前に RAG を試します。 | fixed eval cases が繰り返し失敗するときだけ `needs_fine_tuning` を `True` にします。 |
| `plan = {...}` | model choice、runtime choice、required evidence をつなぐ deployment checklist です。 | auth、logging、rollback など project 固有の evidence を追加します。 |
| `write_text(...)` と `print(...)` | 同じ plan を disk と terminal output に残し、後で review できるようにします。 | `open_llm_runbook.json` を experiment notes と一緒に保存します。 |

各行を説明できれば、この script は理解できています。曖昧な行があれば、GPU に触る前に project card を編集してもう一度実行してください。

## 最小環境チェック

モデルを download する前に実行します。

```bash
python -V
python - <<'PY'
import platform
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda:", torch.cuda.is_available())
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
except Exception as exc:
    print("torch check failed:", repr(exc))
print("platform:", platform.platform())
PY
```

出力を保存します。環境が見えない状態では、モデル結果も再現できません。

## GPU を借りるときの閉ループ

借りた GPU は短い experiment として扱い、恒久的な computer として扱いません。まずは無料または追加費用のない道を確認します。local quantized model、学校や会社の GPU、notebook platform の無料枠が使えるなら先に使います。足りない場合だけ、1つの model/runtime loop を証明する時間だけ借ります。

| 手順 | やること | 保存する証拠 |
|---|---|---|
| 1. run を定義する | target model size、runtime、最初の prompt、最大 rental time を書く。 | `gpu_plan.md`、stop time、budget guardrail |
| 2. instance を選ぶ | Linux、十分な VRAM、十分な disk、SSH access があるものを選ぶ。 | instance type、VRAM、disk、hourly price note |
| 3. access を閉じる | SSH key を使い、model API は default private、必要な port だけ開く。 | security note と exposed ports |
| 4. environment を準備する | `python -V`、torch/CUDA check、可能なら `nvidia-smi` と disk check を実行する。 | `environment_report.txt` |
| 5. 1つの model path を通す | model を download または mount し、1つの prompt を実行し、command、output、failure notes を保存する。 | `first_run.md` |
| 6. stop して archive する | runbook、logs、eval cases、README を project に戻し、instance を stop または destroy する。 | shutdown screenshot または stop note |

一番大事な command は最後の stop かもしれません。実験が成功しても、課金が静かに続くなら engineering failure です。

## Runtime の選び方

| Runtime | 使う場面 | 避ける場面 |
|---|---|---|
| Transformers | 学習、debug、custom Python pipeline | すぐに高スループット serving が必要 |
| Ollama / LM Studio | local demo、laptop test、非エンジニアへの受け渡し | production の細かい制御が必要 |
| llama.cpp | CPU または quantized edge experiment | 標準的な GPU server 機能が必要 |
| vLLM | OpenAI-compatible な高スループット API | GPU や依存関係がまだ整っていない |
| SGLang | structured generation、serving、agentic workloads | 最初の単純な実行だけが目的 |
| Cloud model API | 運用負荷の低い product prototype | privacy、cost、latency が local control を要求 |

まずは product behavior を証明できる最も単純な runtime を使います。latency、cost、privacy、throughput が必要になったときだけ upgrade します。

## Fine-Tuning 判断

1回の悪い回答だけで fine-tune しないでください。

| 現象 | 先に試すこと | Fine-tune を考える条件 |
|---|---|---|
| private knowledge が足りない | RAG | retrieval は正しいが挙動がまだ間違う |
| 出力形式が不安定 | schema、parser、examples | 固定ケースで大量に失敗する |
| tone や role が違う | system prompt と examples | 同じ style 問題が多くの例で繰り返す |
| domain terms が弱い | glossary、RAG、few-shot | 高品質な domain examples が十分ある |
| 遅い、または高い | small model、quantization、batching | 挙動は良いが runtime 制約を満たせない |

多くのコースプロジェクトでは、LoRA が最初の本格的な適応方法です。Full fine-tuning は後の engineering choice であり、default ではありません。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
モデル選択：model、size、license、選定理由
runtime 選択：Transformers/Ollama/llama.cpp/vLLM/SGLang/API と理由
環境情報：Python、torch、CUDA/device、disk、cost estimate
初回実行：exact command、prompt、output、latency、memory note
適応判断：Prompt/RAG/quantization/LoRA/full fine-tune の選択
期待成果：runbook、evaluation table、README、rollback または shutdown note
```

## よくある失敗

- disk、network、VRAM を確認する前に大きなモデルを download する。
- 1回の chat response 成功を deployment evidence とみなす。
- model license や data-use restriction を無視する。
- fixed before/after eval set なしで fine-tune する。
- auth、logging、shutdown rules なしで local model server を公開する。
- 実験後に rented GPU を止め忘れる。

## 合格ライン

1つのプロジェクトに対して model/runtime pair を選び、environment check を動かし、`open_llm_runbook.json` を作り、次に Prompt、RAG、quantization、LoRA、full fine-tuning のどれを選ぶべきか説明し、他のエンジニアが従える README command を書ければ合格です。
