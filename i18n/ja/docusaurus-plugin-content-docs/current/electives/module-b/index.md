---
title: "E.B Python 上級ロードマップ"
sidebar_position: 0
description: "Python 上級選択モジュールの短い実践ロードマップ。デコレータ、ジェネレータ、asyncio、メタプログラミングを、追跡しやすいエンジニアリングパイプラインにつなげます。"
---

# E.B Python 上級ロードマップ

プロトタイプに重複が増えた、遅い呼び出しを待っている、データをストリーミングしたい、ツールを動的に登録したい。そんなときに使う選択モジュールです。

## まずエンジニアリング経路を見る

![Python 上級トピック モジュールマップ](/img/course/elective-python-advanced-module-map-ja.png)

![ジェネレータのストリームパイプライン](/img/course/elective-generator-stream-pipeline-ja.png)

Python 上級機能は、コードを観察しやすく、再利用しやすく、制御しやすくするために使うと効果が出ます。

## 最小の非同期トレースを動かす

```python
import asyncio

async def fetch(name, delay):
    await asyncio.sleep(delay)
    return f"{name}:done"

async def main():
    results = await asyncio.gather(
        fetch("retrieval", 0.1),
        fetch("rerank", 0.05),
    )
    print(results)

asyncio.run(main())
```

期待される出力：

```text
['retrieval:done', 'rerank:done']
```

これは非同期処理の最小習慣です。独立した処理を同時に開始し、すべての結果を待ち、あとから追える形で残します。

## この順番で学ぶ

| Step | レッスン | 実践で残す成果 |
|---|---|---|
| 1 | [E.B.1 デコレータ](./01-decorators-advanced.md) | 業務ロジックを変えずに時間計測やログを足す |
| 2 | [E.B.2 イテレータとジェネレータ](./02-iterators-advanced.md) | 全件を読み込まずに行をストリーミングする |
| 3 | [E.B.3 並行処理](./03-concurrency.md) | タイムアウトとキャンセルを考えながら async タスクを動かす |
| 4 | [E.B.4 メタプログラミング](./04-metaprogramming.md) | ツールやハンドラを明示的に登録する |

## 合格チェック

デコレータ、ジェネレータ、非同期呼び出し、レジストリのいずれかを使った追跡可能なパイプラインを 1 つ作り、なぜデバッグしやすくなったか説明できれば合格です。
