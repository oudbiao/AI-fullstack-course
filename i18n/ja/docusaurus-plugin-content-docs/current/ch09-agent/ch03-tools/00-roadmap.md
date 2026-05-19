---
title: "9.3.1 Tools ロードマップ：Schema、Permission、Observation"
sidebar_position: 0
description: "Agent tools の短い実践ロードマップ：schema を設計し、arguments を検証し、tool calls を routing し、observations を記録し、境界を守る。"
keywords: [Tools overview, Function Calling, Tool Use, Code Agent, Agent tools]
---

# 9.3.1 Tools ロードマップ：Schema、Permission、Observation

Tools は Agent を言語から action に進めます。tools が多いほど強いわけではありません。曖昧な tools は誤呼び出し、不安全な行動、loop、cost leak を生みます。

## まず action boundary を見る

![Agent tool action layer map](/img/course/ch09-tools-action-layer-map-ja.webp)

![Agent tools 章の学習順序図](/img/course/ch09-tools-chapter-flow-ja.webp)

![Agent controlled tool-calling closed loop diagram](/img/course/ch09-tool-control-loop-ja.webp)

Tool calling は常に制御します：tool を選ぶ、arguments を検証する、permission を確認する、実行する、観察する、次の step を決める。

## Tool schema check を動かす

どの tool call も、実行前に schema を使います。

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

tool 実行後、Agent は結果を observe して summarize しなければなりません。失敗した tool を成功したふりで進めないでください。

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Function Calling | モデルの intent を structured action に変える |
| 2 | Tool descriptions | 目的、入力、制約、例、失敗モードを書く |
| 3 | Tool strategy | tool order、fallback、timeout、stop rule を選ぶ |
| 4 | Tool safety | permission、sandbox、audit、human confirmation を追加する |
| 5 | Multi-tool practice | 成功 call と失敗 call の trace を記録する |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
tool_contract: name, description, input schema, output schema
permission: what the tool is allowed to read or change
call_trace: arguments, result, error, retry or fallback
failure_check: wrong tool, bad arguments, unsafe action, or missing observation
safety_action: validate, confirm, sandbox, rate-limit, or rollback
```

## 合格ライン

tool trace を読み、失敗が planning、parameterization、execution、observation、permission control のどこで起きたか判断できれば、この章は合格です。

出口ミニプロジェクトは learning assistant です：3 tool schemas、5 test calls、1 failed-call record、printable trace を含めます。
