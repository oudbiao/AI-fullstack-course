---
title: "7.2.1 LLM 概要ロードマップ：能力、コスト、プロダクト適性"
sidebar_position: 0
description: "短い LLM 概要ロードマップです。発展史、コア概念、業界地図、最初の API 呼び出しワークベンチを扱います。"
keywords: [LLM 概要, 大規模言語モデル, モデル能力, LLM アプリケーション, API 呼び出し]
---

# 7.2.1 LLM 概要ロードマップ：能力、コスト、プロダクト適性

LLM 概要はモデル名リストではありません。大規模モデルに何ができ、何にコストがかかり、prompt、RAG、Agent、fine-tuning のどれが合うかを判断するための章です。

## 7.2.1.1 まず能力スタックを見る

![LLM 概要章関係図](/img/course/ch07-llm-overview-chapter-flow-ja.png)

![大規模モデル能力スタックとアプリケーション生態図](/img/course/ch07-llm-capability-stack-ja.png)

| ルート | 向いている場面 |
|---|---|
| prompt | モデルが十分知っていて、タスクが単純 |
| RAG | 私有知識や変化する知識を引用したい |
| Agent | ツール利用や複数ステップの行動が必要 |
| fine-tuning | 振る舞い、文体、形式を繰り返し適応したい |

## 7.2.1.2 ルート判断を一度動かす

```python
request = {
    "needs_private_docs": True,
    "needs_tool_action": False,
    "needs_repeated_style": False,
}

if request["needs_tool_action"]:
    route = "Agent"
elif request["needs_private_docs"]:
    route = "RAG"
elif request["needs_repeated_style"]:
    route = "fine-tuning"
else:
    route = "prompt"

print("recommended_route:", route)
```

出力：

```text
recommended_route: RAG
```

これは完全な設計判断ではありません。実際のプロダクト要件を満たす最小ルートを選ぶ練習です。

## 7.2.1.3 この順番で学ぶ

| 順番 | 読む | 残すもの |
|---|---|---|
| 1 | [7.2.2 発展史](./01-development-history.md) | scaling と instruction tuning がなぜ重要か |
| 2 | [7.2.3 コア概念](./02-core-concepts.md) | context、token、temperature、遅延、コスト |
| 3 | [7.2.4 業界地図](./03-industry-landscape.md) | モデル/プロバイダ選択メモ |
| 4 | [7.2.5 LLM 呼び出しワークベンチ](./04-llm-call-workbench.md) | 1つの request/response 記録 |

## 7.2.1.4 合格ライン

能力、context、コスト、遅延、データプライバシー、ルート適性からモデル選択を1つ説明できれば合格です。
