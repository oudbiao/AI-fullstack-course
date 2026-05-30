---
title: "9.5.7 MCP、A2A、Agent Protocol Layer"
description: "Agent protocol がなぜ必要か、MCP と agent-to-agent 契約の違い、接続前の capability card の作り方を学びます。"
sidebar:
  order: 30
head:
  - tag: meta
    attrs:
      name: keywords
      content: "MCP, A2A, agent protocol, capability card, agent interoperability"
---
![MCP と A2A の Agent プロトコル層の白板図](/img/course/ch09-agent-protocol-layer-mcp-a2a-whiteboard-ja.webp)

各ツール、モデル、アプリ、peer agent がそれぞれ独自の統合形状を持つと、Agent system はすぐに複雑になります。Protocol は capability discovery、calling、permission、error を予測しやすくするために存在します。

[Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) は、tools、resources、prompts を model application に公開する標準的な方法です。Agent-to-agent パターンは隣の問題を扱います。つまり、ある agent が自分をどう説明し、仕事を受け、status を返し、artifact を別の agent や host に渡すかです。

まず製品名ではなく境界を見ます。

```text
MCP-style boundary: app / host が capability server に接続する
A2A-style boundary: agent / host が別 agent と作業を交渉する
```

## なぜ登場したのか

Function calling は tool use を構造化しましたが、エコシステム問題は残りました。各アプリは tool registry、auth rule、transport、error shape、discovery を個別に作る必要がありました。

Protocol layer は次の問いに答えるために登場しました。

1. どの tools / resources が利用可能か。
2. capability はどんな schema を受け取るか。
3. 誰が呼び出せるか。
4. 失敗時はどう表現するか。
5. 他の agent が同じ contract を理解できるか。

Protocol layer がないと、Agent system は一回限りの glue code だらけになります。

## 概念図

| レイヤー | 主な問い | 典型オブジェクト | 注意すべき失敗 |
|---|---|---|---|
| Tool schema | 何を呼べるか | JSON schema、input/output type | 曖昧な引数 |
| MCP server | 何を公開するか | Tools、resources、prompts | 広すぎる server 権限 |
| Agent card | この agent は何ができるか | Skills、limits、handoff format | 曖昧または過大な capability |
| Policy layer | 誰が何を呼べるか | Allow/deny/confirm rules | 隠れた権限昇格 |
| Trace layer | 何が起きたか | Call log、artifact id、error | デバッグ不能な handoff |

## 判断表

| 必要なこと | 使うパターン | 理由 |
|---|---|---|
| local files、database search、browser actions を model app に公開 | MCP server | Capability discovery と安定した host-client-server 形状 |
| 専門 agent から別 agent へ仕事を渡す | Agent-to-agent contract | 受信側に identity、task shape、status、artifact が必要 |
| 1つのアプリ内で単一関数を呼ぶ | Function calling | シンプルで十分 |
| 多数の tools を security requirements つきで接続 | Protocol + policy layer | discovery だけでは危険 |
| multi-agent run の失敗を debug | Trace + artifact contract | 最終回答だけでは足りない |

## 実行できる演習: Capability Contract を作る

`agent_protocol_contract.py` を作り、Python 3.10 以上で実行します。

```python
import json
from pathlib import Path


capability_server = {
    "name": "course-search-server",
    "protocol": "MCP-style",
    "tools": {
        "search_docs": {
            "input": ["query", "language"],
            "output": ["title", "url", "snippet"],
            "risk": "read",
        }
    },
    "resources": ["course://chapter/{id}"],
}

peer_agent = {
    "name": "qa-review-agent",
    "protocol": "A2A-style",
    "accepts_tasks": ["review_lesson", "check_links"],
    "artifact_contract": ["findings", "commands_run", "risk_notes"],
}

request = {"caller": "course-builder", "action": "search_docs", "risk": "read"}


def authorize(requested_action, server):
    tool = server["tools"].get(requested_action)
    if not tool:
        return {"allowed": False, "reason": "unknown action"}
    if tool["risk"] != "read":
        return {"allowed": False, "reason": "requires human confirmation"}
    return {"allowed": True, "reason": "read-only capability"}


contract = {
    "server": capability_server["name"],
    "peer_agent": peer_agent["name"],
    "authorization": authorize(request["action"], capability_server),
    "handoff_required_fields": peer_agent["artifact_contract"],
}

Path("agent_protocol_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
print(json.dumps(contract, indent=2))
```

期待される出力:

```text
{
  "server": "course-search-server",
  "peer_agent": "qa-review-agent",
  "authorization": {
    "allowed": true,
    "reason": "read-only capability"
  },
  "handoff_required_fields": [
    "findings",
    "commands_run",
    "risk_notes"
  ]
}
```

## コードを一行ずつ読む

`capability_server` は host が discover して call できるものを表します。重要なのは input shape、output shape、risk です。

`peer_agent` は worker-like agent を表します。単なる tool ではなく、task を受け取り artifact を返します。

`authorize()` は safety gate です。Protocol は capability を記述できますが、policy はアプリ側に必要です。

`contract` は discovery、authorization、handoff evidence をつなぎます。

## 小さな練習

新しい tool を追加します。

```python
capability_server["tools"]["delete_doc"] = {
    "input": ["doc_id"],
    "output": ["deleted"],
    "risk": "destructive",
}
request["action"] = "delete_doc"
```

次に `request["action"]` を `"delete_doc"` に変えます。authorization は block するか confirmation を要求すべきです。そうならないなら policy が弱すぎます。

## 残す証拠

外部 tool や agent を接続する前に、このカードを書きます。

```text
capability_name: tool, resource, prompt, peer agent
input_schema: required fields
output_schema: returned fields
risk_level: read, write, external, destructive
auth_rule: allow, deny, confirm
trace_fields: request id, caller, target, result, error
artifact_contract: 受信側が返すべきもの
```

## まとめ

MCP や agent-to-agent contract は魔法ではありません。Capability boundary を inspectable にする方法です。隠れた glue code を減らすために使い、必ず policy と trace evidence を組み合わせます。

<details>
<summary>理解チェック</summary>

Tool schema、MCP server、agent handoff contract の違いを説明し、authorization がどこにあるべきかを示せれば合格です。

</details>
