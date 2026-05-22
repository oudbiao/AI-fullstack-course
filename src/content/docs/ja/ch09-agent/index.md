---
title: "9 AI Agent とエージェントシステム"
description: "goal、plan、tool、observation、memory、安全境界、評価、配置意識を備えた追跡可能な Agent ループを作る。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AI Agent, エージェント, Function Calling, ReAct, MCP, Multi-Agent, ツール呼び出し"
---

# 9 AI Agent とエージェントシステム

![AI Agent システムメインビジュアル](/img/course/ch09-agent-systems-ja.webp)

第 8 章では、モデルが文書に基づいて答えられるようにしました。第 9 章では、システムが**目標に向かって行動する**ようにします。次の手を計画し、ツールを呼び、観察を読み、方針を調整し、安全に止まり、人がレビューできる trace を残します。

多 Agent フレームワークから始める必要はありません。まず、すべてのステップを見せられる小さな Agent を作ります。

## メインルートでの位置

ここまでに LLM 回答ループと RAG 証拠ループを作りました。この章では制御された行動を加えます。システムが次の手を決め、許可されたツールを呼び、観察を読み、状態を更新し、再生可能な trace とともに安全に停止します。

これはメインルート最後の中核アプリケーション層です。この章の後、10-12 章は製品特化になります。Vision、NLP、Multimodal のワークフローは、同じ証拠、ツール、trace、安全習慣へ接続できます。

## まず Agent 実行ループを見る

![Agent 実行ループ](/img/course/ch09-agent-execution-loop-ja.webp)

Agent は「ツール付きチャットボット」ではなく、制御された実行ループです。

| 部分 | やさしい意味 | 必ず制御すること |
|---|---|---|
| 目標 | Agent が達成したいこと | 範囲、成功基準、停止条件 |
| 状態（State） | 今分かっていること | 現在の入力、過去の観察、残り手順 |
| 計画（Plan） | 次に試すこと | 最大ステップ、代替経路、人への引き継ぎ |
| ツール（Tool） | 検索、ファイル読み取り、API 呼び出し、コード実行など | スキーマ、検証、ホワイトリスト、リスクレベル |
| 観察 | ツールが返した結果 | エラー処理、再試行ルール、信頼境界 |
| 記憶（Memory） | ステップ間・実行間で残す情報 | 短期状態と長期好みの境界 |
| 追跡記録（トレース） | 実行を再生できる記録 | 目標、行動、引数、観察、コスト、最終結果 |

## 学習順序とタスク表

多 Agent の前に、単一 Agent を追跡可能にします。まず中核の単一 Agent ルート **9.1 -> 9.2 -> 9.3 -> 9.4 -> 9.8 -> 9.10** を進みます。MCP、フレームワーク、多 Agent、配置運用は、単一 Agent ループが安定してから学ぶ高度な章です。

| 手順 | 読む内容 | 手を動かすこと | 残す証拠 |
|---|---|---|---|
| 9.1 | Agent 基礎と構造 | 目標、状態、計画、ツール、観察、記憶を説明する | 構成スケッチ |
| 9.2 | 推論と計画 | 同じタスクで ReAct と Plan-and-Execute を比べる | ステップ追跡 |
| 9.3 | ツール呼び出し | パラメータとエラーを持つツールを1～2個定義する | `tools_schema.md` |
| 9.4 | 記憶 | 現在状態と長期記憶を分ける | 記憶境界メモ |
| 9.8 | 評価と安全 | 出力を採点し、危険な行動を止め、追跡記録を確認する | 追跡ログ、安全ブロック、評価ケース |
| 9.10 | ステージプロジェクト | [9.10.5 実践：追跡可能な単一 Agent アシスタントを作る](./ch10-projects/04-stage-hands-on-workshop.md) を動かす | `agent_traces.jsonl`、安全境界、評価ケース |
| 9.5 | MCP | MCP をツールとデータソース接続の標準方式として理解する | 接続メモ |
| 9.6-9.7 | フレームワークと多 Agent | 単一 Agent ループが安定してから学ぶ | フレームワーク選択メモ |
| 9.9 | 配置と運用 | 中核プロジェクトが動いてから runtime、recovery、cost、本番確認を足す | リリースチェックリストと rollback メモ |

## 必修ルート、拡張、深掘り

| 層 | いま学ぶこと | どう使うか |
|---|---|---|
| 必修コア | 単一 Agent ループ、ツールスキーマ、ホワイトリスト、最大ステップ、state 境界、memory 境界、トレース ログ、安全ブロック、評価ケース | 単なるデモではなく、レビューできる Agent の最小スキルです |
| 任意の拡張 | MCP、フレームワーク比較、多 Agent 協調、配置運用、コスト最適化 | 単一 Agent ループが安定し、統合や規模が必要になったときに戻ります |
| 深掘り課題 | 同じタスクを ワークフロー、RAG フロー、関数呼び出し、Agent トレース として比較し、最も単純で安全な設計を説明する | 流行ではなく意図を持って Agent を使うためです |

## 最初に動かすループ：トレース を表示する

このオフラインスクリプトは LLM に依存しません。学ぶのは工程習慣です。すべての action は再生可能であるべきです。あとで固定 `plan` をモデル生成の計画に置き換えても、trace 形式は残します。

`ch09_agent_trace.py` を作成し、Python 3.10 以降で実行してください。

```python
import json


def search_docs(tool_input: dict) -> str:
    return "Found notes about RAGOps, AgentOps, evaluation sets, and trace logs."


def make_todo(tool_input: dict) -> str:
    topic = tool_input["topic"]
    return f"1) Review {topic} notes; 2) add one eval case; 3) write failure notes."


TOOLS = {
    "search_docs": {"fn": search_docs, "risk": "read_only"},
    "make_todo": {"fn": make_todo, "risk": "draft_only"},
}

goal = "Prepare a short RAG review plan."
plan = [
    {
        "thought": "Find relevant course materials before making a plan.",
        "action": "search_docs",
        "input": {"query": "RAGOps AgentOps evaluation trace"},
    },
    {
        "thought": "Turn the materials into a small review checklist.",
        "action": "make_todo",
        "input": {"topic": "RAG evaluation"},
    },
]

trace = []
for step_number, step in enumerate(plan, start=1):
    tool = TOOLS.get(step["action"])
    if tool is None:
        observation = "Blocked: tool is not whitelisted."
        risk = "blocked"
    else:
        observation = tool["fn"](step["input"])
        risk = tool["risk"]

    trace.append(
        {
            "step": step_number,
            "goal": goal,
            "thought": step["thought"],
            "action": step["action"],
            "input": step["input"],
            "risk": risk,
            "observation": observation,
        }
    )

for item in trace:
    print(json.dumps(item, ensure_ascii=False))
```

期待される出力の先頭:

```text
{"step": 1, "goal": "Prepare a short RAG review plan.", "thought": "Find relevant course materials before making a plan.", "action": "search_docs", ...
{"step": 2, "goal": "Prepare a short RAG review plan.", "thought": "Turn the materials into a small review checklist.", "action": "make_todo", ...
```

操作メモ: `make_todo` をホワイトリスト外の `send_email` などに変えてください。スクリプトはそれをブロックするはずです。これが安全境界の最小版です。

## 深さの段階

| 段階 | 証明できること |
|---|---|
| 最低合格 | 1つの追跡記録を実行し、各目標、行動、入力、観察、結果を説明できる。 |
| 実務準備 | ツールスキーマ を定義し、ホワイトリスト外のツールをブロックし、最大ステップを設定し、失敗追跡を保存できる。 |
| 深い確認 | ワークフローの方が Agent より安全な場面と、危険な行動に人間承認を置く場所を判断できる。 |

## Agent、ワークフロー、RAG、関数呼び出し の選び方

![Agent 境界選択図](/img/course/ch09-agent-boundary-map-ja.webp)

Agent は強力ですが、常に最初の選択肢ではありません。

| 問題 | まず使うもの | Agent を使う目安 |
|---|---|---|
| 手順が固定で既知 | ワークフロー | 観察のたびに経路が変わる |
| 私有情報や新情報が必要 | RAG | 検索が大きな目標の一部にすぎない |
| 1回の構造化された行動で十分 | 関数呼び出し | 複数のツール呼び出しと状態更新が必要 |
| リスクが高い | 人間確認付きワークフロー | Agent は下書きし、高リスク行動は人が確認 |
| 探索に計画、ツール、記憶、回復が必要 | Agent | すべてのステップを記録し、安全に止められる |

## よくある失敗

- 単一 Agent が安定する前に多 Agent を作る。
- スキーマ、検証、有用なエラーメッセージなしでツールを呼ぶ。
- 停止条件がなく、ループとコスト増加を招く。
- 高リスクツールを人間確認なしで動かす。
- 成功デモだけ見せ、失敗追跡を残さない。
- 記憶を置き場にしてしまい、現在状態、長期好み、タスク履歴を分けない。

## クリア確認

この章を出る前に、次をできるようにしてください。

- 目標、状態、計画、ツール、観察、記憶、追跡記録、ガードレールを説明できる。
- 追跡スクリプトを動かし、ホワイトリスト外ツールをブロックできる。
- `agent_traces.jsonl`、`tools_schema.md`、`safety_boundary.md`、`failure_cases.md` を保存できる。
- タスクにワークフロー、RAG、関数呼び出し、Agent のどれが合うか判断できる。
- 第 9 章フルワークショップを動かし、評価タスク1つと安全ブロック例1つを追加できる。

印刷用チェックリストは [9.0 学習チェックリスト](./study-guide.md) を使ってください。プロジェクトから始めたい場合は [9.10.5 実践：追跡可能な単一 Agent アシスタントを作る](./ch10-projects/04-stage-hands-on-workshop.md) へ進みます。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
中核ルート: 9.1 -> 9.2 -> 9.3 -> 9.4 -> 9.8 -> 9.10 を先に
エージェントループ: 目標 -> 計画 -> ツール/アクション -> 観測 -> 記憶 -> 評価
追跡規則：すべての操作は 入力、出力、判断、エラー記録 を残すべき
安全ルール: 権限、ツール境界、ガードレール、ロールバックは設計の一部
段階分割：単一 Agent のループが安定してから MCP／フレームワーク／マルチ Agent／デプロイへ進む
```
