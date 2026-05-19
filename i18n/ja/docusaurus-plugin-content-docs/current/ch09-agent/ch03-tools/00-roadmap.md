---
title: "9.3.1 ツールロードマップ：スキーマ、権限、観察"
sidebar_position: 0
description: "Agent ツールの短い実践ロードマップ：スキーマを設計し、引数を検証し、ツール呼び出しをルーティングし、観察を記録し、境界を守る。"
keywords: [Tools overview, Function Calling, Tool Use, Code Agent, Agent tools]
---

# 9.3.1 ツールロードマップ：スキーマ、権限、観察

ツールは Agent を言語から行動へ進めます。ツールが多いほど強いわけではありません。曖昧なツールは誤呼び出し、不安全な行動、ループ、コスト漏れを生みます。

## まず行動境界を見る

![Agent ツール行動レイヤー図](/img/course/ch09-tools-action-layer-map-ja.webp)

![Agent tools 章の学習順序図](/img/course/ch09-tools-chapter-flow-ja.webp)

![Agent の制御付きツール呼び出し閉ループ図](/img/course/ch09-tool-control-loop-ja.webp)

ツール呼び出しは常に制御します：ツールを選ぶ、引数を検証する、権限を確認する、実行する、観察する、次のステップを決める。

## ツールスキーマチェックを動かす

どのツール呼び出しも、実行前にスキーマを使います。

```python
tool_call = {
    "name": "search_course_docs",
    "args": {"query": "RAG evaluation", "top_k": 3},
}

schema = {
    "name": "search_course_docs",
    "required": ["query", "top_k"],
    "max_top_k": 5,
}

name_ok = tool_call["name"] == schema["name"]
args_ok = all(field in tool_call["args"] for field in schema["required"])
limit_ok = tool_call["args"]["top_k"] <= schema["max_top_k"]

print("can_execute:", name_ok and args_ok and limit_ok)
print("observation_needed:", True)
```

期待される出力：

```text
can_execute: True
observation_needed: True
```

ツール実行後、Agent は結果を観察して要約しなければなりません。失敗したツールを成功したふりで進めないでください。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Function Calling | モデルの意図を構造化された行動に変える |
| 2 | ツール説明 | 目的、入力、制約、例、失敗モードを書く |
| 3 | ツール戦略 | ツール順序、fallback、timeout、stop rule を選ぶ |
| 4 | ツール安全 | 権限、sandbox、audit、human confirmation を追加する |
| 5 | 複数ツール実践 | 成功呼び出しと失敗呼び出しの trace を記録する |

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
tool_contract: name, description, input schema, output schema
permission: what the tool is allowed to read or change
call_trace: arguments, result, error, retry or fallback
failure_check: wrong tool, bad arguments, unsafe action, or missing observation
safety_action: validate, confirm, sandbox, rate-limit, or rollback
```

## 合格ライン

ツール trace を読み、失敗が計画、パラメータ化、実行、観察、権限制御のどこで起きたか判断できれば、この章は合格です。

出口ミニプロジェクトは学習アシスタントです：3 つのツールスキーマ、5 つのテスト呼び出し、1 つの失敗呼び出し記録、出力可能な trace を含めます。
