---
title: "2.1.2 データ型と変数"
sidebar_position: 2
description: "Python の基本データ型と変数の使い方を身につける"
---

# 2.1.2 データ型と変数

![変数、オブジェクト、参照の関係図](/img/course/ch02-variable-object-reference-ja.webp)

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
概念: 変数、型、演算子、入力/出力、分岐、ループ、構造、関数、またはモジュール
コード：この概念のための最小限の実行可能な Python スニペット
出力：印字値、型、branch結果、loop trace、または返り値
失敗確認: 型不一致、インデント、オフバイワン、可変データ、または import パスの問題
期待される成果：概念が機能することを証明するコードと出力結果
```

## この節の位置づけ

この節では、Python がデータをどのように表し、保存するのかを理解します。変数、数値、文字列、ブール値、そして型変換は、後で条件分岐を書いたり、表データを扱ったり、モデル API を呼び出したりするときの土台になります。まずは、これらの「最小のデータ単位」に慣れていきましょう。

## 学習目標

- 変数とは何かを理解し、命名ルールを身につける
- Python の基本データ型である整数、浮動小数点数、文字列、ブール値を理解する
- データ型同士の変換を学ぶ
- 動的型付けの意味を理解する

---

## 変数とは？

変数は、**ラベルが貼られた箱**だと考えてみましょう。中にものを入れて、ラベルで見つけることができます。

```python
service_name = "ログイン API"  # 箱に「service_name」というラベルを貼る
latency_ms = 185             # 箱に「latency_ms」というラベルを貼る
timeout_seconds = 2.5        # 箱に「timeout_seconds」というラベルを貼る
```

Python の `=` は「等しい」ではなく、**代入**を表します。つまり、右側の値を左側の箱に入れるという意味です。

```python
# 代入の向き：右から左
x = 10      # 10 を x という箱に入れる

# 箱の中身は変更できる
x = 20      # 今は x の中身は 20 になった（10 はなくなる）

# 変数の値を使って計算できる
y = x + 5   # y = 20 + 5 = 25
print(y)    # 出力: 25
```

### 変数名のルール

Python には変数名に関するいくつかの決まりがあります。

| ルール | 正しい例 | 間違った例 |
|------|---------|---------|
| 英字、数字、アンダースコアのみ使える | `service_name`, `task2` | `service-name`, `task!` |
| 数字で始められない | `task1` | `1task` |
| Python のキーワードは使えない | `my_class` | `class`, `if`, `for` |
| 大文字小文字を区別する | `Service` と `service` は別の変数 | — |

### 命名の慣例（必須ではないですが、みんなこうします）

```python
# よい命名 ✅ —— 小文字 + アンダースコア（snake_case）
service_name = "ログイン API"
learning_rate = 0.001
max_epochs = 100

# あまりよくない命名 ❌ —— 使えないわけではないが、わかりにくい
a = "ログイン API"  # a では何かわからない
x1 = 0.001       # x1 が何を表すのか不明
SN = "ログイン API" # 略しすぎていて、他の人に伝わりにくい
```

:::tip 命名の黄金ルール
変数名は、**見ただけで何を表すか分かる**ようにしましょう。少し長くなっても（`task_count`）、意味の分からない略語（`tc`）よりずっとよいです。
:::

---

## 数値型

### 整数（int）

整数は、小数点のない数です。正の数、負の数、0 のいずれも含みます。

```python
retry_count = 3
queue_delta = -10
count = 0
big_number = 1_000_000  # アンダースコアで区切ると見やすい。1000000 と同じ

print(type(retry_count))  # <class 'int'>
```

:::info type() 関数
`type()` は、どんな値の型でも確認できる関数です。学習中によく使います。変数の型を確認するときに便利です。
:::

Python の整数には大きさの制限がありません（C/Java のような int の範囲制限はありません）。

```python
huge = 99999999999999999999999999999999
print(huge + 1)  # まったく問題ありません
```

### 浮動小数点数（float）

浮動小数点数は、小数点を含む数です。

```python
pi = 3.14159
timeout_seconds = 2.5
negative = -0.001

print(type(pi))  # <class 'float'>
```

**浮動小数点数の精度に注意**してください。これは、どのプログラミング言語にもある問題です。

```python
>>> 0.1 + 0.2
0.30000000000000004    # 正確な 0.3 ではない！
```

これは Python のバグではなく、コンピュータが小数を 2 進数で保存することによる、もともとの性質です。AI 開発では、このわずかな誤差が結果に影響しないことが多いです。ただし、金融計算のように正確な結果が必要な場合は、`decimal` モジュールを使うことができます。

### 整数と浮動小数点数の計算

```python
a = 10
b = 3

print(a + b)    # 13    足し算
print(a - b)    # 7     引き算
print(a * b)    # 30    掛け算
print(a / b)    # 3.333... 割り算（結果は常に float）
print(a // b)   # 3     切り捨て除算
print(a % b)    # 1     余り
print(a ** b)   # 1000  べき乗（10 の 3 乗）
```

よくある落とし穴：

```python
# 割り算 / の結果は、割り切れても必ず float
>>> 10 / 2
5.0         # 5 ではなく 5.0

# 整数がほしいなら // を使う
>>> 10 // 2
5
```

---

## 文字列（str）

文字列は**テキスト**です。文字の並びを引用符で囲みます。

### 文字列の作成

```python
# シングルクォートとダブルクォートのどちらも使える。効果は同じ
service = 'ログイン API'
status = "ready"

# 文字列の中に引用符がある場合は、別の種類の引用符で囲む
sentence = "レビュアーは言いました: 'リリースできます'"
command = 'CLI フラグは "--dry-run" です'

# トリプルクォート：複数行のテキストを書ける
release_notes = """
ログイン API
- タイムアウトを調整
- リトライログを有効化
"""
print(release_notes)

print(type(service))  # <class 'str'>
```

### 文字列の連結

```python
module_name = "チケット"
endpoint_name = " API"

# 方法 1: + で連結
full_endpoint = module_name + endpoint_name
print(full_endpoint)  # チケット API

# 方法 2: f-string を使う（おすすめ！Python 3.6+）
version = "v1"
intro = f"{full_endpoint} は {version} で動きます"
print(intro)  # チケット API は v1 で動きます

# 方法 3: format() を使う
intro2 = "{} は {} で動きます".format(full_endpoint, version)
print(intro2)  # チケット API は v1 で動きます
```

:::tip f-string はベストプラクティス
f-string（`f"...{変数}..."`）は、現代の Python で最もよく使われる文字列フォーマット方法です。簡潔で分かりやすく、この先の授業でもたくさん使います。
:::

### よく使う文字列操作

```python
text = "Hello, Python!"

# 長さを取得
print(len(text))         # 14

# 大文字・小文字の変換
print(text.upper())      # HELLO, PYTHON!
print(text.lower())      # hello, python!

# 部分文字列を探す
print(text.find("Python"))  # 7（7 番目の位置から）
print("Python" in text)     # True

# 置換
print(text.replace("Python", "AI"))  # Hello, AI!

# 前後の空白を削除
messy = "  hello  "
print(messy.strip())    # "hello"

# 分割
csv_line = "ログイン API,185,ready"
parts = csv_line.split(",")
print(parts)  # ['ログイン API', '185', 'ready']
```

### 文字列のインデックスとスライス

![文字列のインデックスとスライス図解](/img/course/ch02-string-index-slice-ja.webp)

文字列の各文字には**位置番号（インデックス）**があり、0 から始まります。

```python
text = "Python"
#       P y t h o n
# index: 0 1 2 3 4 5
# 負の index: -6 -5 -4 -3 -2 -1

print(text[0])    # P（最初の文字）
print(text[5])    # n（6 番目の文字）
print(text[-1])   # n（最後の文字）
print(text[-2])   # o（後ろから 2 番目の文字）
```

**スライス**を使うと、一部の文字列を取り出せます。

```python
text = "Python"

print(text[0:3])   # Pyt（index 0 から index 3 まで、3 は含まない）
print(text[2:5])   # tho
print(text[:3])    # Pyt（先頭から。0 は省略できる）
print(text[3:])    # hon（末尾まで。終了位置は省略できる）
print(text[:])     # Python（文字列全体のコピー）
print(text[::2])   # Pto（1 文字おきに取り出す）
print(text[::-1])  # nohtyP（文字列を逆順にする！）
```

:::info スライスの書き方
`text[start:stop:step]` —— `start` から始めて、`stop` で終わる（ただし stop は含まない）、`step` ごとに 1 つ取り出します。覚えておきましょう：**左閉右開**（開始位置は含み、終了位置は含まない）。
:::

### 文字列は変更できない

```python
text = "Hello"
# text[0] = "h"  # エラー！TypeError: 'str' object does not support item assignment

# 変更したい場合は、新しい文字列を作る
text = "h" + text[1:]  # "hello"
```

---

## ブール値（bool）

ブール値は `True`（真）と `False`（偽）の 2 つだけです。先頭は大文字にします。

```python
is_deployed = True
has_errors = False

print(type(is_deployed))  # <class 'bool'>
```

ブール値は、主に**比較演算**から得られます。

```python
print(5 > 3)       # True
print(5 < 3)       # False
print(5 == 5)      # True（`=` が 1 つだと代入、2 つで比較）
print(5 != 3)      # True（`!=` は等しくないという意味）
print("abc" == "abc")  # True
```

ブール値は、後で学ぶ条件分岐（if/else）でたくさん使います。

### 真値と偽値

Python では、いろいろなものをブール値として扱えます。次の値は「偽」と見なされます。

```python
# 以下はすべて "偽"（Falsy）
bool(0)        # False
bool(0.0)      # False
bool("")       # False（空文字列）
bool([])       # False（空リスト）
bool(None)     # False

# それ以外は "真"（Truthy）
bool(1)        # True
bool(-1)       # True（0 でなければ真）
bool("hello")  # True（空でない文字列は真）
bool([1, 2])   # True（空でないリストは真）
```

---

## None 型

`None` は Python の特別な値で、**「何もない」**ことを表します。

```python
result = None
print(result)        # None
print(type(result))  # <class 'NoneType'>
```

`None` は、「まだ値がない」や「結果がない」を表すときによく使います。

```python
# 関数が返り値を持たない場合、デフォルトで None を返す
def say_hello():
    print("Hello!")

result = say_hello()   # Hello! を表示
print(result)          # None
```

---

## 型変換

ときには、ある型を別の型に変換する必要があります。

```python
# 文字列 → 数値
latency_str = "185"
latency_ms = int(latency_str)      # 文字列を整数に変換
print(latency_ms + 10)             # 195

timeout_str = "2.5"
timeout_seconds = float(timeout_str)  # 文字列を浮動小数点数に変換
print(timeout_seconds)                # 2.5

# 数値 → 文字列
task_count = 12
task_count_str = str(task_count)   # 整数を文字列に変換
print("タスク数: " + task_count_str)  # タスク数: 12

# 整数 ↔ 浮動小数点数
x = int(3.7)    # 3（小数点以下をそのまま切り捨てる。四捨五入ではない）
y = float(5)    # 5.0
```

**よくあるエラー**：文字列と数値は、`+` でそのまま連結できない

```python
latency_ms = 185
# print("レイテンシ: " + latency_ms)  # エラー！TypeError

# 正しい方法 1: 文字列に変換する
print("レイテンシ: " + str(latency_ms))

# 正しい方法 2: f-string を使う（おすすめ）
print(f"レイテンシ: {latency_ms}")

# 正しい方法 3: カンマで区切る（print が自動で空白を入れる）
print("レイテンシ:", latency_ms)
```

### 型変換の早見表

| 変換 | 関数 | 例 | 結果 |
|------|------|------|------|
| → 整数 | `int()` | `int("25")` | `25` |
| → 浮動小数点数 | `float()` | `float("3.14")` | `3.14` |
| → 文字列 | `str()` | `str(100)` | `"100"` |
| → ブール値 | `bool()` | `bool(0)` | `False` |

---

## 動的型付け

Python は**動的型付け**の言語です。つまり、変数を先に宣言して型を決める必要はなく、同じ変数にいつでも別の型を代入できます。

```python
x = 10          # x は整数
print(type(x))  # <class 'int'>

x = "hello"     # 今度は x が文字列になった
print(type(x))  # <class 'str'>

x = True        # 今度は x がブール値になった
print(type(x))  # <class 'bool'>
```

とても柔軟ですが、注意も必要です。もともと数値を入れていた変数を、うっかり文字列にしてしまわないようにしましょう。

Java（静的型付け言語）と比べてみましょう。

```java
int x = 10;       // x は整数と宣言する
x = "hello";      // エラー！Java では型を変えられない
```

---

## 多重代入

Python には、便利な代入の書き方があります。

```python
# 複数の変数に同時に代入
a, b, c = 1, 2, 3
print(a, b, c)  # 1 2 3

# 2 つの変数の値を入れ替える（Python らしい簡潔な書き方）
a, b = b, a
print(a, b)  # 2 1

# 複数の変数に同じ値を代入
x = y = z = 0
print(x, y, z)  # 0 0 0
```

この変数の入れ替え方は、とても Pythonic（Python らしい）です。他の言語では、通常は一時変数が必要です。

```python
# 他の言語での書き方
temp = a
a = b
b = temp

# Python の書き方
a, b = b, a  # 1 行で完了！
```

---

## 手を動かしてみよう

### 練習 1: サービス状態カード

サービス状態を変数に保存し、f-string で出力してみましょう。

```python
service = "ログイン API"
latency_ms = 185
timeout_seconds = 2.5
is_ready = True

print(f"サービス: {service}")
print(f"レイテンシ: {latency_ms} ms")
print(f"タイムアウト: {timeout_seconds} 秒")
print(f"準備完了: {is_ready}")
print(f"15 ms 増えた場合: {latency_ms + 15} ms")
```

### 練習 2: レイテンシ単位変換器

ミリ秒を秒に変換する式：`seconds = milliseconds / 1000`

```python
latency_ms = 375.0
latency_seconds = latency_ms / 1000
print(f"{latency_ms} ms = {latency_seconds} 秒")
```

`latency_ms` の値を変えて、いくつかのリクエスト時間を計算してみましょう。

### 練習 3: 文字列操作

```python
email = "  Support.API@Example.COM  "

# 1. 前後の空白を削除する
# 2. 小文字に変換する
# 3. @ の位置を見つける
# 4. ユーザー名部分（@ の前）を取り出す
```

ヒント：`.strip()`、`.lower()`、`.find()`、スライスを組み合わせて使えます。

### 練習 4: 型を調べる

`type()` を使って、次の値の型を確認しましょう。まず予想してから確かめてみてください。

```python
print(type(42))
print(type(3.14))
print(type("3.14"))
print(type(True))
print(type(None))
print(type(1 + 2))
print(type(1 + 2.0))    # 整数 + 浮動小数点数 = ？
print(type("1" + "2"))  # 文字列 + 文字列 = ？
```

<details>
<summary>参考実装と解説</summary>

1. サービス状態カードでは `str`、`int`、`float`、`bool` を使います。f-string では変数の値に加え、`latency_ms + 15` のような式も表示できます。
2. `375.0` ms は `0.375` 秒です。`latency_ms` を変えると `latency_seconds` も変わるよう、答えを固定せず式をコードに残します。
3. 正規化後のメールアドレスは `support.api@example.com` です。空白を除き小文字にした後の `@` のインデックスは `11`、ユーザー名は `support.api` です。
4. 型の出力は順に `int`、`float`、`str`、`bool`、`NoneType`、`int`、`float`、`str` です。
5. よくある間違いは `"1" + "2"` を計算だと思うことです。これは文字列結合なので、結果は `"12"` です。

</details>

---

## まとめ

| 型 | キーワード | 例 | 用途 |
|------|--------|------|------|
| **整数** | `int` | `42`, `-10`, `0` | カウント、インデックス |
| **浮動小数点数** | `float` | `3.14`, `-0.5` | 正確な数値、科学計算 |
| **文字列** | `str` | `"hello"`, `'world'` | テキストデータ |
| **ブール値** | `bool` | `True`, `False` | 条件分岐 |
| **空値** | `NoneType` | `None` | 「値がない」ことを表す |

:::tip 核心の理解
Python では、**すべてがオブジェクト**です。数値もオブジェクト、文字列もオブジェクト、さらに `True` や `None` もオブジェクトです。それぞれのオブジェクトには型（`type`）があり、型によってできる操作が決まります。
:::
