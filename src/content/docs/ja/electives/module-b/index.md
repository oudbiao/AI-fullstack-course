---
title: "E.B Python 上級ロードマップ"
description: "Python 上級選択モジュールの短い実践ロードマップ。デコレータ、ジェネレータ、asyncio、メタプログラミングを、追跡しやすいエンジニアリングパイプラインにつなげます。"
sidebar:
  order: 0
---
プロトタイプに重複が増えた、遅い呼び出しを待っている、データをストリーミングしたい、ツールを動的に登録したい。そんなときに使う選択モジュールです。

## まずエンジニアリング経路を見る

![Python 上級トピック モジュールマップ](/img/course/elective-python-advanced-module-map-ja.webp)

![ジェネレータのストリームパイプライン](/img/course/elective-generator-stream-pipeline-ja.webp)

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

## 実プロジェクトでの使い方

コードに観察性と制御が必要になったとき、この module を使います。decorator は多くの関数に同じ logging や timing を足すときに役立ちます。generator は stream data に向いています。async は複数の I/O 待ちを扱います。registry は tool を明示的に管理します。

高度に見せるために使う必要はありません。良い使い方には trace、最小実行例、debugging の利点があります。チームメイトが data の入口、待ち場所、出口を説明できないなら、pattern を単純に戻してください。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
Pythonパターン：デコレータ、イテレータ、ジェネレータ、並行処理プリミティブ、またはメタプログラミングフック
コード成果物：最小限の実行可能な例と表示された出力
使用場面：この pattern が AI app、pipeline、tool、または server を改善する場面
失敗確認：隠れた副作用、読みにくい抽象化、競合状態、または過度な設計
期待される成果: 実践的なAIシステム用途のメモを含む小さな高度Python例
```

## この順番で学ぶ

| ステップ | レッスン | 実践で残す成果 |
|---|---|---|
| 1 | [E.B.1 デコレータ](/ja/electives/module-b/01-decorators-advanced/) | 業務ロジックを変えずに時間計測やログを足す |
| 2 | [E.B.2 イテレータとジェネレータ](/ja/electives/module-b/02-iterators-advanced/) | 全件を読み込まずに行をストリーミングする |
| 3 | [E.B.3 並行処理](/ja/electives/module-b/03-concurrency/) | タイムアウトとキャンセルを考えながら async タスクを動かす |
| 4 | [E.B.4 メタプログラミング](/ja/electives/module-b/04-metaprogramming/) | ツールやハンドラを明示的に登録する |

## 合格チェック

デコレータ、ジェネレータ、非同期呼び出し、レジストリのいずれかを使った追跡可能なパイプラインを 1 つ作り、なぜデバッグしやすくなったか説明できれば合格です。

<details>
<summary>確認の考え方と解説</summary>

合格する答えは、追跡できるパイプラインを 1 つ示します。たとえば、デコレータでログを足す、ジェネレータでストリーミング処理する、async で複数の I/O を並行に回す、またはレジストリでツールを明示的に管理するやり方です。証拠には、最小実行例の出力と、なぜデバッグしやすくなったのかの説明を含めます。

「もっと巧妙に書けた」だけでは足りません。観察性、再利用性、制御性が本当に良くなった理由を示してください。

</details>
