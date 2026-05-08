---
title: "E.B.3 並行プログラミング（asyncio を含む）"
sidebar_position: 10
description: "asyncio、セマフォ、タイムアウトを使って I/O タスクを並行実行しつつ、上流サービスを守る。"
keywords: [asyncio, concurrency, async, semaphore, gather, Python]
---

# E.B.3 並行プログラミング（asyncio を含む）

![asyncio 並行制御フローチャート](/img/course/elective-asyncio-concurrency-control-ja.png)

![非同期タスクのタイムアウト・キャンセル・レート制限図](/img/course/elective-asyncio-timeout-cancel-rate-limit-map-ja.png)

並行処理は、プログラムの多くの時間が「待ち」であるときに役立ちます。HTTP 呼び出し、DB 呼び出し、ファイル I/O、スクレイピング、RAG 検索、Agent のツール呼び出しなどです。CPU が重い処理を魔法のように速くするものではありません。

## 準備するもの

- Python 3.10+
- 外部パッケージ不要
- `python` を実行できるターミナル

## 重要用語

- **I/O-bound（I/O 待ち中心）**：大半の時間を外部システム待ちに使う処理。
- **CPU-bound（CPU 計算中心）**：大半の時間を計算に使う処理。
- **Coroutine（コルーチン）**：`await` で一時停止できる非同期関数。
- **`asyncio.gather`**：複数の awaitable を実行し、結果を集める。
- **Semaphore（セマフォ）**：同時に動くタスク数を制限する。
- **Timeout（タイムアウト）**：一定時間を超えたら待つのをやめる。

## 制御付き非同期 batch を動かす

`async_batch.py` を作成します。

```python
import asyncio


async def call_tool(name, delay):
    await asyncio.sleep(delay)
    return f"{name}:ok"


async def guarded_call(semaphore, name, delay, timeout):
    async with semaphore:
        try:
            return await asyncio.wait_for(call_tool(name, delay), timeout=timeout)
        except asyncio.TimeoutError:
            return f"{name}:timeout"


async def main():
    semaphore = asyncio.Semaphore(2)
    results = await asyncio.gather(
        guarded_call(semaphore, "search", 0.1, 0.5),
        guarded_call(semaphore, "database", 0.2, 0.5),
        guarded_call(semaphore, "slow_tool", 1.0, 0.3),
    )
    print(results)


asyncio.run(main())
```

実行します。

```bash
python async_batch.py
```

期待される出力：

```text
['search:ok', 'database:ok', 'slow_tool:timeout']
```

大切なのは `gather` だけではありません。`gather`、並行数の上限、タイムアウト処理を組み合わせることです。

## 上限を変える

この小さな確認コードで、2つの上限を見てみます。

```python
import asyncio

for limit in [2, 1]:
    semaphore = asyncio.Semaphore(limit)
    print("limit:", limit, "semaphore:", type(semaphore).__name__)
```

期待される出力：

```text
limit: 2 semaphore: Semaphore
limit: 1 semaphore: Semaphore
```

最終結果は同じですが、タスクはより保守的に実行されます。実サービスでは、これにより上流 API を急なリクエストから守れます。

## asyncio を使う場面

向いているもの：

1. 多数のネットワークリクエスト
2. 複数のツール呼び出し
3. 複数ソースからの RAG 検索
4. DB やキュー待ち

最初の選択肢にしにくいもの：

1. 重い数値計算
2. 大きな画像変換
3. 待ち時間のボトルネックがなく、単純さを優先したいコード

## よくある間違い

- I/O-bound か確認せず、どこにでも `async` を付ける。
- 並行数上限なしで `gather` を使う。
- タイムアウトを忘れ、遅い上流一つで全体が詰まる。
- 例外を握りつぶし、どのタスクが失敗したか記録しない。

## 練習

ツール呼び出しをさらに5つ追加し、`Semaphore(3)` にします。その後、タイムアウトを `0.15` に下げ、いくつが `:timeout` になるか数えてください。
