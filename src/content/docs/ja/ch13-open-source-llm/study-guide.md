---
title: "13.0 学習チェックリスト：オープンソース LLM デプロイ"
description: "第 13 章のチェックリスト。model selection、runtime choice、environment checks、serving evidence、evaluation、fine-tuning decisions を確認する。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "オープンソース LLM チェックリスト, ローカルモデルデプロイ, LoRA チェックリスト, vLLM チェックリスト"
---
このページは印刷用チェックリストとして使います。詳しい説明が必要なときは、[第 13 章入口ページ](/ja/ch13-open-source-llm/) に戻ってください。

![オープンソース LLM デプロイ証拠パック](/img/course/ch13-open-source-llm-evidence-pack-ja.webp)

まだ手を動かしていない場合は、この checklist の前に [13.1 実践：オープンソース LLM を動かしてサービス化する](/ja/ch13-open-source-llm/hands-on-open-llm-lab/) を完了してください。

## 2時間の初回通読

1. **20 分：deployment loop を読む**
   「runtime、API、logs、eval、rollback が分かって初めて model deployment と言える」と言えたら止めます。

2. **20 分：environment check を動かす**
   「このマシンが CUDA を使えるか、CPU だけか分かる」と言えたら止めます。

3. **25 分：runbook script を動かす**
   「hardware と project constraints から runtime を選べる」と言えたら止めます。

4. **25 分：5 prompt の eval table を作る**
   「runtime や tuning を変える前に model behavior を比較できる」と言えたら止めます。

5. **30 分：adaptation decision を書く**
   「Prompt、RAG、quantization、LoRA、no tuning の理由を説明できる」と言えたら止めます。

## 必ず残す証拠

- `environment_report.txt`：Python、torch、CUDA/device、platform、disk または instance note。
- `model_decision.md`：model、size、license、source、reason、rejected alternatives。
- `open_llm_runbook.json`：runtime choice、adaptation choice、required evidence。
- `first_run.md`：exact command、prompt、output、latency または memory note。
- `eval_cases.csv`：5つ以上の prompts、expected behavior、pass/fail、notes。
- `README.md`：setup、run、evaluate、stop server、rollback または shutdown。

## 品質ゲート

- **Reproducibility**：他のエンジニアが model version、runtime、command、environment を特定できる。
- **Safety**：共有前に license、privacy、auth、logging、shutdown を確認している。
- **Evaluation**：runtime や tuning の変更を同じ eval cases で比較している。
- **Cost control**：GPU rental time、memory、latency、stop procedure を記録している。
- **Adaptation**：fine-tuning が1回の不満ではなく繰り返す証拠に基づいている。

## 章を出る前の質問

- なぜこの model size と license を選んだか説明できますか？
- なぜこの runtime が今の project に十分か説明できますか？
- environment check を実行または再現できますか？
- 変更後、同じ5つの prompt で output を比較できますか？
- Prompt、RAG、quantization、LoRA、full fine-tune の選択を説明できますか？

答えがすべて「はい」なら、オープンソース LLM をランダムな model demo ではなく engineering option として扱えます。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
環境レポート：Python、torch、CUDA/device、platform、hardware/cost note
モデル決定：selected model、license、size、source、rejected alternatives
runtime contract：command または endpoint、request format、response format、error path
評価：fixed prompts、outputs、pass/fail notes、latency または memory note
適応選択：Prompt/RAG/quantization/LoRA/full fine-tune decision with reason
```
