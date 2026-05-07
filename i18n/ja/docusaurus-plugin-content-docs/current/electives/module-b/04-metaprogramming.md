---
title: "E.B.4 メタプログラミング"
sidebar_position: 11
description: "レジストリやディスクリプタのような実用的なメタプログラミングを使い、普通のコードを魔法にしすぎない。"
keywords: [metaprogramming, type, registry, descriptor, dynamic class, Python]
---

# E.B.4 メタプログラミング

![Python メタプログラミングのレジストリ図](/img/course/elective-metaprogramming-registry-map-ja.png)

メタプログラミングとは、コードでコード構造を整理または生成することです。日常の Python 工程で役立つのは、派手な技ではなく、自動登録、フィールド検証、定型処理の削減であることが多いです。

## 準備するもの

- Python 3.10+
- 外部パッケージ不要
- クラスの基本理解

## 重要用語

- **Registry（レジストリ）**：利用可能な実装を覚えておくマッピング。
- **デコレータ登録**：デコレータでクラスや関数をレジストリに追加すること。
- **Descriptor（ディスクリプタ）**：属性の読み書きを制御するオブジェクト。
- **`__set_name__`**：ディスクリプタが自分に割り当てられた属性名を知るためのメソッド。
- **動的クラス**：実行時に作られるクラス。代表例は `type`。

## レジストリとディスクリプタを動かす

`metaprogramming_demo.py` を作成します。

```python
REGISTRY = {}


def register(name):
    def decorator(cls):
        REGISTRY[name] = cls
        return cls

    return decorator


@register("csv")
class CsvLoader:
    def load(self):
        return "csv rows"


@register("json")
class JsonLoader:
    def load(self):
        return "json rows"


class NonEmpty:
    def __set_name__(self, owner, name):
        self.private_name = "_" + name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return getattr(instance, self.private_name, None)

    def __set__(self, instance, value):
        if not value:
            raise ValueError("name cannot be empty")
        setattr(instance, self.private_name, value)


class JobConfig:
    name = NonEmpty()


loader = REGISTRY["json"]()
print(loader.load())
print(sorted(REGISTRY))

config = JobConfig()
config.name = "daily-import"
print(config.name)

try:
    config.name = ""
except ValueError as error:
    print("error:", error)
```

実行します。

```bash
python metaprogramming_demo.py
```

期待される出力：

```text
json rows
['csv', 'json']
daily-import
error: name cannot be empty
```

レジストリは手書きの対応表を減らします。ディスクリプタはフィールド検証をフィールド定義の近くに置けます。

## 使う価値がある場面

向いているもの：

1. 多くのクラスを自動登録したい。
2. フレームワークがプラグインを発見する必要がある。
3. 多くのフィールドが同じ検証動作を共有する。
4. 設定から繰り返し構造を作りたい。

普通のクラスや辞書のほうが分かりやすいなら、無理に使いません。

## よくある間違い

- 高度に見せるためだけに動的な技を使う。
- 振る舞いを深く隠しすぎて、デバッグしにくくする。
- 小さな重複を消すために、全体の読みやすさを犠牲にする。

## 練習

`yaml` loader を追加し、`sorted(REGISTRY)` に含まれることを確認してください。次に `retry_count` フィールド用の `IntegerRange(min_value, max_value)` ディスクリプタを作ります。
