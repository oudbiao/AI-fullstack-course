---
title: "7.5.1 プロンプトエンジニアリングロードマップ：ブリーフ、出力、評価"
description: "プロンプトエンジニアリングの短い実践ロードマップ：曖昧な依頼を再利用できるタスクブリーフ、構造化出力、反復評価に変える。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Prompt ガイド, Prompt Engineering, 構造化出力, Prompt 評価"
---
プロンプトエンジニアリングは、アプリケーションとモデルのあいだのインターフェースです。目的は気の利いた一文を書くことではなく、1 回のモデル呼び出しを予測可能、解析可能、テスト可能、改善可能にすることです。

## まず Prompt のループを見る

![Prompt エンジニアリング章の関係図](/img/course/ch07-prompt-chapter-flow-ja.webp)

![Prompt 三層タスク仕様図](/img/course/ch07-prompt-spec-three-layer-map-ja.webp)

![Prompt 反復テストの閉ループ図](/img/course/ch07-prompt-iteration-loop-ja.webp)

モデルに基本能力はあるのに、結果が曖昧、不安定、形式違い、評価しにくい場合に、この章の方法を使います。

## プロンプト契約チェックを動かす

LLM を呼ぶ前に、プロンプトを契約として書きます：タスク、コンテキスト、出力形式、制約です。この小さなスクリプトで、その契約がテストできる程度にそろっているか確認します。

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

このページを終えたら、この証拠カードを残します。

```text
プロンプト契約：タスク、文脈、制約、出力形式
固定ケース：プロンプト版全体で同じ入力を使用
スキーマ確認: 構造化出力がパーサーで検証される
失敗ノート: 原因ごとにまとめたプロンプト失敗
橋渡し：第8章ではこのループに検索で取得した文脈を追加する
```

## 合格ライン

固定した入力セットを使い、毎回 1 つの Prompt 層だけを変更し、感覚ではなく証拠で改善を説明できれば、この章は合格です。

出口ミニプロジェクトは、コース内容抽出 Prompt です。1 つのコース文書を入力し、章のテーマ、学習目標、前提知識、重要語、練習案、リスクメモを JSON または Markdown 表で出力します。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、token、context、attention、prompt、生成挙動が1回の request-response path でどうつながるかを説明します。
2. 証拠には、再現できる prompt または structured-output test を1つ残し、出力が通った理由または失敗した理由を書きます。
3. prompt 設計、RAG、fine-tuning、alignment を切り分け、観察した問題を直す最も軽い方法を選べれば十分です。

</details>
