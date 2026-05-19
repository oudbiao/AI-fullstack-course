---
title: "7.5.1 Prompt Engineering ロードマップ：ブリーフ、出力、評価"
sidebar_position: 0
description: "Prompt Engineering の短い実践ロードマップ：曖昧な依頼を再利用できるタスクブリーフ、構造化出力、反復評価に変える。"
keywords: [Prompt ガイド, Prompt Engineering, 構造化出力, Prompt 評価]
---

# 7.5.1 Prompt Engineering ロードマップ：ブリーフ、出力、評価

Prompt Engineering は、アプリケーションとモデルのあいだのインターフェースです。目的は気の利いた一文を書くことではなく、1 回のモデル呼び出しを予測可能、解析可能、テスト可能、改善可能にすることです。

## まず Prompt のループを見る

![Prompt エンジニアリング章の関係図](/img/course/ch07-prompt-chapter-flow-ja.webp)

![Prompt 三層タスク仕様図](/img/course/ch07-prompt-spec-three-layer-map-ja.webp)

![Prompt 反復テストの閉ループ図](/img/course/ch07-prompt-iteration-loop-ja.webp)

モデルに基本能力はあるのに、結果が曖昧、不安定、形式違い、評価しにくい場合に、この章の方法を使います。

## Prompt 契約チェックを動かす

LLM を呼ぶ前に、Prompt を契約として書きます：タスク、コンテキスト、出力形式、制約です。この小さなスクリプトで、その契約がテストできる程度にそろっているか確認します。

```python
prompt_contract = {
    "task": "Extract chapter metadata",
    "context": "One course markdown file",
    "output_format": ["chapter", "goals", "prerequisites", "risks"],
    "constraints": ["return JSON only", "mark missing facts as null"],
}

required = ["task", "context", "output_format", "constraints"]
missing = [field for field in required if not prompt_contract.get(field)]

print("ready:", not missing)
print("fields:", ", ".join(required))
print("test_case_count:", 3)
```

期待される出力：

```text
ready: True
fields: task, context, output_format, constraints
test_case_count: 3
```

![Prompt 契約チェックの実行結果図](/img/course/ch07-prompt-contract-check-result-map-ja.webp)

`ready` が `False` なら、追加例を試す前にタスクブリーフを直します。曖昧な Prompt は、曖昧なデバッグを生みます。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Prompt 基礎 | 曖昧な依頼をタスク、コンテキスト、形式、制約に書き換える |
| 2 | 高度な Prompt | 必要なときだけ例、手順、役割、境界条件を加える |
| 3 | 構造化出力 | プログラムで解析できる JSON、表、Markdown を作る |
| 4 | Prompt 実践 | 同じ固定入力で Prompt バージョンを比較する |
| 5 | 評価ラボ | 合格率、失敗タイプ、次の変更を記録する |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
prompt_contract: task, context, constraints, output format
fixed_cases: same inputs used across prompt versions
schema_check: structured output validated by parser
failure_note: prompt failure grouped by cause
bridge: Chapter 8 adds retrieved context to this loop
```

## 合格ライン

固定した入力セットを使い、毎回 1 つの Prompt 層だけを変更し、感覚ではなく証拠で改善を説明できれば、この章は合格です。

出口ミニプロジェクトは、コース内容抽出 Prompt です。1 つのコース文書を入力し、章のテーマ、学習目標、前提知識、重要語、練習案、リスクメモを JSON または Markdown 表で出力します。
