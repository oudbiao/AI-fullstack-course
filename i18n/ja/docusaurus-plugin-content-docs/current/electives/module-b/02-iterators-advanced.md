---
title: "E.B.2 イテレータとジェネレータの応用"
sidebar_position: 9
description: "ジェネレータを使い、すべてを一度にメモリへ載せず、データストリームを一歩ずつ処理する。"
keywords: [iterator, generator, yield, yield from, lazy evaluation, streaming]
---

# E.B.2 イテレータとジェネレータの応用

![ジェネレータのストリーミングパイプライン図](/img/course/elective-generator-stream-pipeline-ja.png)

データがストリームとして届くとき、ジェネレータは役に立ちます。ログ、ファイル、API ページ、サンプル batch、検索結果、モデル出力などです。値を一つずつ出すため、不要な中間リストを作らずに済みます。

## 準備するもの

- Python 3.10+
- 外部パッケージ不要
- `for` ループの基本理解

## 重要用語

- **Iterator（イテレータ）**：次の値を順に出せるオブジェクト。
- **Generator（ジェネレータ）**：`yield` で値を遅延生成する関数。
- **Lazy evaluation（遅延評価）**：次の値が必要になったときだけ計算すること。
- **Pipeline（パイプライン）**：小さな処理ステップをつなげた流れ。
- **`yield from`**：別の iterable の値をそのまま外へ流す構文。

## ストリーミングパイプラインを動かす

`generator_pipeline.py` を作成します。

```python
def read_events():
    events = [
        "INFO request ok",
        "ERROR db timeout",
        "INFO cache hit",
        "ERROR auth failed",
        "ERROR model busy",
    ]
    for event in events:
        yield event


def filter_errors(events):
    for event in events:
        if event.startswith("ERROR"):
            yield event


def normalize(events):
    for event in events:
        yield event.lower()


def batch(items, size):
    group = []
    for item in items:
        group.append(item)
        if len(group) == size:
            yield group
            group = []
    if group:
        yield group


pipeline = batch(normalize(filter_errors(read_events())), size=2)

for group in pipeline:
    print(group)
```

実行します。

```bash
python generator_pipeline.py
```

期待される出力：

```text
['error db timeout', 'error auth failed']
['error model busy']
```

このパイプラインは、読み取り、フィルタ、正規化、batch 化を行いますが、各段階で完全なリストを作りません。

## `yield from` を使う

この単独で動く小さな例を実行します。

```python
def flatten(groups):
    for group in groups:
        yield from group

pipeline = [
    ["error db timeout", "error auth failed"],
    ["error model busy"],
]

for item in flatten(pipeline):
    print(item)
```

期待される出力：

```text
error db timeout
error auth failed
error model busy
```

ネストしたループよりも、「各グループ内の要素を外へ流す」という意図がはっきりします。

## ジェネレータが役立つ場面

向いているもの：

1. 入力が大きくなる可能性がある。
2. レコードを一件ずつ処理する。
3. 読み取り、フィルタ、変換、batch 化をつなげたい。
4. 全要素へのランダムアクセスが不要。

データが小さく、何度もアクセスするほうが分かりやすいなら、リストで十分です。

## よくある間違い

- 消費済みのジェネレータを再利用できると思い込む。
- ジェネレータは常に速いと思う。主な価値はメモリと構造化にあることが多い。
- 単純なリスト変換まで無理に `yield` にして読みにくくする。

## 練習

`batch` を変更し、`batch_id` も出力してください。その後、入力イベントを変えても、後続ステップを変えずに動くことを確認します。
