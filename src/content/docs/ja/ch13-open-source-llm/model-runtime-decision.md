---
title: "13.3 モデルと Runtime の決定"
description: "モデル選定を、ライセンス、サイズ、context length、hardware、quantization、runtime、fallback path を含む engineering decision に変える。"
sidebar:
  order: 4
head:
  - tag: meta
    attrs:
      name: keywords
      content: "open-source LLM model selection, runtime decision, Ollama, llama.cpp, vLLM, SGLang, quantization"
---
![オープンソース LLM runtime loop](/img/course/ch13-open-source-llm-runtime-loop-ja.webp)

よいオープンソース LLM プロジェクトは、最初の download より前に始まります。このレッスンでは、model selection を書面の decision に変え、hardware、license、latency target、product boundary に合わない model へ時間を使いすぎないようにします。

## このページで解くこと

初学者はよく「どのモデルが一番よいか？」と聞きます。engineering question はもっと具体的です。「この project、この hardware、この license constraint、この rollback path に対して、どの model/runtime pair で十分か？」です。

まず project behavior を証明できる最小の model と最も単純な runtime を使います。quality、context length、throughput、privacy、cost の証拠が必要性を示したときだけ上に進みます。

## 判断の階段

1. **タスク適合**
   project が chat、extraction、code、多言語、long context、tool calling、multimodal behavior のどれを必要とするか決めます。

2. **ライセンス適合**
   model を中心に system を作る前に model card と license を読みます。commercial use、redistribution、data-use restrictions のメモを残します。

3. **ハードウェア適合**
   download 前に VRAM/RAM と disk を見積もります。local で動かないなら、より小さな model、quantization、rented GPU、cloud API fallback を選びます。

4. **Runtime 適合**
   学習には Transformers、local handoff には Ollama/LM Studio、quantized CPU/edge test には llama.cpp、server-style inference には vLLM/SGLang を使います。

5. **証拠適合**
   model version、command、first output、evaluation table、stop procedure がそろうまで、decision は完了ではありません。

## モデル判断表

`model_runtime_decision.md` を作ります。

```md
# Model Runtime Decision

project_goal: support-operations SOP assistant
must_have: private document handling, Chinese/English answers, stable JSON output
nice_to_have: low latency, long context, OpenAI-compatible endpoint

candidate_1: Qwen2.5-0.5B-Instruct
license_note: check model card before deployment
runtime: vLLM when GPU is available, Transformers for first local test
hardware_note: small enough for first experiment; still validate memory
risk: quality may be too weak for complex SOP reasoning

candidate_2: 7B instruct model
license_note: check commercial and redistribution terms
runtime: vLLM or SGLang on rented GPU
hardware_note: requires planned GPU budget and shutdown proof
risk: higher cost and slower iteration

fallback: cloud model API or RAG with current API model
why_now: prove the deployment loop before chasing larger models
rejected_for_now: full fine-tuning, because eval failures are not proven yet
```

具体的な model name は変わります。しかし decision shape は変えません。

## Runtime の選択ルール

**token、prompt、Python behavior を確認したいときは Transformers から始めます。** debug しやすく model API に近い一方、通常は最終的な high-throughput server ではありません。

**laptop demo や非エンジニアへの handoff が目的なら Ollama または LM Studio を使います。** setup friction は下がりますが、production control は一部失います。

**CPU、quantized、edge constraints が重要なら llama.cpp を使います。** 小さな local experiment には強いですが、明確な API と evaluation story は必要です。

**OpenAI-compatible serving と throughput が必要なら vLLM を使います。** GPU、driver、memory、security posture が明確になるまではここから始めません。

**structured generation や agentic serving pattern が必要なら SGLang を使います。** 強力ですが、新しさではなく project requirements で正当化します。

## ミニ演習

第 8 章または第 9 章の project を 1 つ選び、次の decision paragraph を書きます。

```text
この project では、まず _____ を _____ で使います。理由は _____ です。
まだ大きな model を使わない理由は _____ です。
fixed eval set が _____ を示したときだけ切り替えます。
```

<details>
<summary>判断の考え方と解説</summary>

よい答えは、model size と runtime を evidence に結びつけます。たとえば、小さな instruct model を Transformers または Ollama で動かし、prompt、RAG context、output schema を先に証明します。同じ eval cases で品質が十分で、project が service endpoint を本当に必要としたときだけ vLLM に進みます。prompt、RAG、schema、quantization の後に残る failure が分かる前に、LoRA や大きな GPU を選ばないでください。

</details>

## 残す証拠

```text
model_decision: 選んだ model、license note、size、context length、rejected alternatives
runtime_decision: 選んだ runtime、hardware reason、fallback runtime
hardware_note: local CPU/GPU または rented GPU estimate、disk、expected stop time
eval_gate: model または runtime を変える根拠になる fixed cases
expected_output: model_runtime_decision.md と first-run command
```

## 合格ライン

現在の model/runtime pair がこの project に十分な理由、upgrade する条件、random model demo を uncontrolled deployment にしないための evidence を説明できれば、このレッスンは合格です。
