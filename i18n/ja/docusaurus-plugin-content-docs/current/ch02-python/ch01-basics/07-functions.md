---
title: "2.1.7 関数の基礎"
sidebar_position: 7
description: "関数の定義、引数、戻り値、スコープを理解する"
---

# 2.1.7 関数の基礎

![関数の呼び出し、引数、スコープの図](/img/course/ch02-function-call-scope-ja.webp)

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

この節では、くり返し使うロジックを関数にまとめる方法を学びます。関数は、「スクリプトを書く」から「保守しやすいプログラムを書く」へ進むための重要なステップです。さらに、これから学ぶデータ処理の流れ、モデル学習の流れ、Web API のロジックを組み立てる土台にもなります。

## 学習目標

- 関数とは何か、なぜ必要なのかを理解する
- 関数の定義と呼び出しを身につける
- 引数（位置引数、デフォルト引数、キーワード引数）を理解する
- 戻り値の使い方を身につける
- 変数のスコープを理解する

---

## なぜ関数が必要なのか？

たとえば、データ処理スクリプトの中で API の平均レイテンシを何回も計算する場面を考えてみましょう。

```python
# 1回目の計算
api_latencies = [120, 95, 240, 180, 310]
total1 = sum(api_latencies)
avg1 = total1 / len(api_latencies)
print(f"平均レイテンシ: {avg1:.1f} ms")

# 2回目の計算（同じロジックをもう一度書く）
worker_latencies = [80, 76, 95, 110, 140, 90]
total2 = sum(worker_latencies)
avg2 = total2 / len(worker_latencies)
print(f"平均レイテンシ: {avg2:.1f} ms")

# 3回目の計算（さらにもう一度……）
batch_latencies = [450, 510, 480, 530, 470]
total3 = sum(batch_latencies)
avg3 = total3 / len(batch_latencies)
print(f"平均レイテンシ: {avg3:.1f} ms")
```

同じロジックを 3 回も書いています。もし後で計算方法を変えたくなったら（たとえば最大値と最小値の測定値を除くなど）、3 か所を直す必要があります。

関数を使うと、こうなります。

```python
def calculate_average(values):
    """平均値を計算する"""
    return sum(values) / len(values)

# これなら1行でOK
print(f"平均レイテンシ: {calculate_average([120, 95, 240, 180, 310]):.1f} ms")
print(f"平均レイテンシ: {calculate_average([80, 76, 95, 110, 140, 90]):.1f} ms")
print(f"平均レイテンシ: {calculate_average([450, 510, 480, 530, 470]):.1f} ms")
```

**関数の主な価値：**

| メリット | 説明 |
|------|------|
| **再利用** | 1回書けば、何度でも使える |
| **抽象化** | 複雑なロジックを関数名の後ろに隠せる。呼び出す側は「何をするか」だけ分かればよい |
| **保守性** | 修正が必要なとき、1か所だけ直せばよい |
| **読みやすさ** | 関数名そのものがコメントの役割を持つ。`calculate_average(latencies_ms)` は一目で分かる |

---

## 関数の定義と呼び出し

### 基本構文

```python
def greet(name):
    """誰かにあいさつする"""  # docstring（ドキュメンテーション文字列）、関数の説明
    print(f"こんにちは、{name}！プロジェクト作業スペースへようこそ！")

# 関数を呼び出す
greet("Mina")     # こんにちは、Mina！プロジェクト作業スペースへようこそ！
greet("Kai")      # こんにちは、Kai！プロジェクト作業スペースへようこそ！
```

構文のポイント：
- `def` キーワードは「関数を定義する」という意味
- `greet` は関数名（変数と同じく、小文字 + アンダースコアが基本）
- `(name)` は引数リスト
- `:` のコロンを忘れない
- 関数の中身はインデントが必要
- `"""..."""` は docstring で、関数の役割を説明する

### 引数がない関数

```python
def say_hello():
    print("Hello, World!")

say_hello()  # Hello, World!
```

### 複数の引数を持つ関数

```python
def add(a, b):
    result = a + b
    print(f"{a} + {b} = {result}")

add(3, 5)    # 3 + 5 = 8
add(10, 20)  # 10 + 20 = 30
```

---

## 戻り値

関数は `return` を使って、結果を呼び出し元に返せます。

```python
def add(a, b):
    return a + b

# 関数の戻り値を変数に入れる
result = add(3, 5)
print(result)       # 8

# そのまま使うこともできる
print(add(10, 20))  # 30

# 式の中でも使える
total = add(1, 2) + add(3, 4)
print(total)  # 10
```

### 複数の値を返す

```python
def get_min_max(numbers):
    """リストの最小値と最大値を返す"""
    return min(numbers), max(numbers)

# タプルのアンパックで受け取る
smallest, largest = get_min_max([3, 1, 4, 1, 5, 9])
print(f"最小値: {smallest}, 最大値: {largest}")
# 最小値: 1, 最大値: 9
```

### `return` がない関数

関数に `return` 文がない場合、または `return` の後ろに値がない場合、関数は `None` を返します。

```python
def greet(name):
    print(f"こんにちは、{name}！")
    # return なし

result = greet("Mina")   # 表示: こんにちは、Mina！
print(result)            # None
```

### `return` のもう1つの使い方: 早く関数を終える

```python
def divide(a, b):
    if b == 0:
        print("エラー：割る数は 0 にできません！")
        return None   # 関数を途中で終了
    return a / b

print(divide(10, 3))   # 3.333...
print(divide(10, 0))   # エラー：割る数は 0 にできません！ その後 None を返す
```

---

## 引数のくわしい説明

### 位置引数

順番どおりに渡す引数です。

```python
def describe_task(task, owner):
    print(f"{task} は {owner} が担当します")

describe_task("ログイン API", "Mina")   # ログイン API は Mina が担当します
describe_task("Mina", "ログイン API")   # Mina は ログイン API が担当します —— 順番が逆！
```

### キーワード引数

引数名を指定して値を渡します。順番を気にしなくて大丈夫です。

```python
def describe_task(task, owner):
    print(f"{task} は {owner} が担当します")

# キーワード引数なら順番は関係ない
describe_task(owner="Mina", task="ログイン API")   # ログイン API は Mina が担当します
describe_task(task="ダッシュボード UI", owner="Kai")   # ダッシュボード UI は Kai が担当します
```

### デフォルト引数

引数に初期値を与えて、呼び出し時に省略できるようにします。

```python
def train_model(epochs=10, lr=0.001, batch_size=32):
    print(f"学習パラメータ: epochs={epochs}, lr={lr}, batch_size={batch_size}")

# すべてデフォルト値を使う
train_model()
# 学習パラメータ: epochs=10, lr=0.001, batch_size=32

# 一部だけ変更する
train_model(epochs=50)
# 学習パラメータ: epochs=50, lr=0.001, batch_size=32

train_model(epochs=100, lr=0.01)
# 学習パラメータ: epochs=100, lr=0.01, batch_size=32
```

:::caution デフォルト引数の落とし穴
![ミュータブルなデフォルト引数の落とし穴図解](/img/course/ch02-mutable-default-trap-ja.webp)

デフォルト値は、関数が定義されたときに決まります。リストや辞書のような可変オブジェクトをデフォルト値にしないでください。

```python
# 間違った書き方 ❌
def add_item(item, items=[]):
    items.append(item)
    return items

print(add_item("a"))  # ['a']
print(add_item("b"))  # ['a', 'b'] —— バグ！前回の 'a' が残っている

# 正しい書き方 ✅
def add_item(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items
```
:::

### `*args`: 任意個の位置引数を受け取る

```python
def calculate_sum(*numbers):
    """任意個の数字の合計を計算する"""
    total = 0
    for num in numbers:
        total += num
    return total

print(calculate_sum(1, 2))           # 3
print(calculate_sum(1, 2, 3, 4, 5))  # 15
print(calculate_sum(10))             # 10
```

### `**kwargs`: 任意個のキーワード引数を受け取る

```python
def print_info(**info):
    """任意個の情報を表示する"""
    for key, value in info.items():
        print(f"{key}: {value}")

print_info(name="ログイン API", owner="Mina", status="進行中")
# name: ログイン API
# owner: Mina
# status: 進行中
```

### 引数の順番ルール

いろいろな引数を組み合わせるときの順番は次のとおりです。

```python
def func(pos_arg, default_arg=10, *args, **kwargs):
    print(f"pos_arg={pos_arg}")
    print(f"default_arg={default_arg}")
    print(f"args={args}")
    print(f"kwargs={kwargs}")

func(1, 2, 3, 4, name="test")
# pos_arg=1
# default_arg=2
# args=(3, 4)
# kwargs={'name': 'test'}
```

---

## 変数のスコープ

変数の「スコープ」とは、その変数が**有効な範囲**のことです。

### ローカル変数 vs グローバル変数

```python
# グローバル変数: 関数の外で定義
message = "私はグローバル変数です"

def my_function():
    # ローカル変数: 関数の中で定義
    local_var = "私はローカル変数です"
    print(message)      # グローバル変数を読み取れる
    print(local_var)    # ローカル変数を読み取れる

my_function()
print(message)          # グローバル変数にアクセスできる
# print(local_var)      # エラー！ローカル変数は関数の外では存在しない
```

### 同じ名前の変数

```python
x = 10  # グローバル変数

def my_function():
    x = 20  # これは新しいローカル変数。グローバル変数の変更ではない
    print(f"関数内の x: {x}")  # 20

my_function()
print(f"関数外の x: {x}")    # 10（グローバル変数は変更されていない）
```

### `global` キーワード

どうしても関数内でグローバル変数を変更したい場合に使います（通常はおすすめしません）。

```python
count = 0

def increment():
    global count   # グローバル変数 count を使うと宣言する
    count += 1

increment()
increment()
increment()
print(count)  # 3
```

:::tip ベストプラクティス
できるだけ**グローバル変数を使わない**ようにしましょう。関数は、引数でデータを受け取り、戻り値で結果を返す形にすると、テストしやすく、理解もしやすくなります。
:::

---

## docstring

よい関数には、分かりやすい説明文を付けましょう。

```python
def calculate_bmi(weight, height):
    """
    体格指数（BMI）を計算する。

    引数:
        weight (float): 体重、単位は kg
        height (float): 身長、単位は m

    戻り値:
        float: BMI の値

    例:
        >>> calculate_bmi(70, 1.75)
        22.857142857142858
    """
    return weight / (height ** 2)

# 関数の説明を見る
help(calculate_bmi)
```

---

## 総合例

### 例 1: API レイテンシ分析ツール

```python
def analyze_latencies(latencies_ms, service="不明なサービス"):
    """
    API レイテンシのリストを分析して、統計情報を返す。

    引数:
        latencies_ms: レイテンシのリスト（ミリ秒）
        service: サービス名
    戻り値:
        統計情報を含む辞書
    """
    if not latencies_ms:
        return {"error": "レイテンシリストが空です"}

    avg = sum(latencies_ms) / len(latencies_ms)
    slow = [ms for ms in latencies_ms if ms >= 200]
    normal = [ms for ms in latencies_ms if ms < 200]

    return {
        "service": service,
        "count": len(latencies_ms),
        "average_ms": round(avg, 1),
        "max_ms": max(latencies_ms),
        "min_ms": min(latencies_ms),
        "slow_rate": f"{len(slow) / len(latencies_ms):.1%}",
        "slow_requests": len(slow),
        "normal_requests": len(normal)
    }

def print_report(stats):
    """整形したレイテンシレポートを表示する"""
    print(f"\n{'='*30}")
    print(f"  {stats['service']} レイテンシレポート")
    print(f"{'='*30}")
    print(f"  サンプル数:       {stats['count']}")
    print(f"  平均レイテンシ:   {stats['average_ms']} ms")
    print(f"  最大レイテンシ:   {stats['max_ms']} ms")
    print(f"  最小レイテンシ:   {stats['min_ms']} ms")
    print(f"  遅いリクエスト率: {stats['slow_rate']}")
    print(f"  遅いリクエスト数: {stats['slow_requests']}")
    print(f"  通常リクエスト数: {stats['normal_requests']}")
    print(f"{'='*30}")

# 使用例
login_latencies = [120, 95, 240, 180, 310, 88, 160, 205]
worker_latencies = [450, 510, 480, 530, 470, 620, 390]

login_stats = analyze_latencies(login_latencies, "ログイン API")
worker_stats = analyze_latencies(worker_latencies, "バックグラウンド Worker")

print_report(login_stats)
print_report(worker_stats)
```

### 例 2: 簡単なパスワード生成器

```python
import random
import string

def generate_password(length=12, use_upper=True, use_digits=True, use_special=True):
    """
    ランダムなパスワードを生成する。

    引数:
        length: パスワードの長さ。デフォルトは 12
        use_upper: 大文字を含めるか
        use_digits: 数字を含めるか
        use_special: 特殊文字を含めるか
    """
    chars = string.ascii_lowercase  # 小文字

    if use_upper:
        chars += string.ascii_uppercase
    if use_digits:
        chars += string.digits
    if use_special:
        chars += "!@#$%^&*"

    password = ''.join(random.choice(chars) for _ in range(length))
    return password

# いろいろな種類のパスワードを生成
print(f"デフォルトのパスワード: {generate_password()}")
print(f"英字のみ:   {generate_password(length=8, use_digits=False, use_special=False)}")
print(f"超強力パスワード: {generate_password(length=20)}")
```

---

## 実践演習

### 練習 1: 温度変換関数

摂氏と華氏を相互に変換する 2 つの関数を書いてみましょう。

```python
def celsius_to_fahrenheit(celsius):
    """摂氏 → 華氏: F = C × 9/5 + 32"""
    return celsius * 9 / 5 + 32

def fahrenheit_to_celsius(fahrenheit):
    """華氏 → 摂氏: C = (F - 32) × 5/9"""
    return (fahrenheit - 32) * 5 / 9

# テスト
print(celsius_to_fahrenheit(100))  # 212.0 が出力されるはず
print(fahrenheit_to_celsius(32))   # 0.0 が出力されるはず
```

### 練習 2: リスト統計関数

数値リストを受け取り、最大値、最小値、平均値、中央値を返す関数を書いてみましょう。

```python
def list_stats(numbers):
    """
    リストの統計情報を返す。
    max()、min()、sum() などの組み込み関数は使わず、自分で実装すること！
    """
    if not numbers:
        return None

    maximum = numbers[0]
    minimum = numbers[0]
    total = 0
    for value in numbers:
        if value > maximum:
            maximum = value
        if value < minimum:
            minimum = value
        total += value

    sorted_numbers = sorted(numbers)
    n = len(sorted_numbers)
    if n % 2 == 1:
        median = sorted_numbers[n // 2]
    else:
        median = (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2

    average = total / len(numbers)
    return {
        "max": maximum,
        "min": minimum,
        "average": average,
        "median": median,
    }

# テスト
stats = list_stats([3, 1, 4, 1, 5, 9, 2, 6, 5])
print(stats)
```

### 練習 3: 数当てゲーム（関数版）

前の数当てゲームを、関数を使う形に書き換えてみましょう。

```python
def guess_number_game(min_val=1, max_val=100, max_attempts=7):
    """数当てゲーム"""
    import random

    target = random.randint(min_val, max_val)
    print(f"{min_val} から {max_val} までの数字を予想してください")
    for attempt in range(1, max_attempts + 1):
        raw = input(f"{attempt}/{max_attempts} 回目: ")
        if not raw.isdigit():
            print("整数を入力してください。")
            continue

        guess = int(raw)
        if guess == target:
            print("正解です！")
            return True
        if guess < target:
            print("小さすぎます")
        else:
            print("大きすぎます")
    print(f"ゲーム終了。答えは {target} でした。")
    return False

# ゲームを実行
guess_number_game()
guess_number_game(1, 50, 5)  # 範囲を狭くして、回数を少なくする
```

安定してテストしたい場合は、`target = random.randint(min_val, max_val)` を一時的に `target = 42` に変えてください。関数の動きが確認できたら、ランダム版に戻します。

<details>
<summary>参考実装と解説</summary>

1. 温度変換テストでは、`100` C が `212.0` F、`32` F が `0.0` C になります。`37` C のような往復テストも追加すると安心です。
2. `list_stats([3, 1, 4, 1, 5, 9, 2, 6, 5])` は、最大値 `9`、最小値 `1`、平均 `4.0`、中央値 `4` を返します。
3. 空リストに `None` を返す設計は、呼び出し側が確認するなら妥当です。別案として `ValueError` を送出する方法もあります。
4. ゲーム関数は、成功時に `True`、試行回数切れで `False` を返すと、テストコードで結果を確認できます。
5. ユーザー操作そのものが目的でない限り、良い関数は隠れた入力や出力を減らします。純粋関数のほうがテストしやすいです。

</details>

---

## まとめ

| 概念 | 説明 | 例 |
|------|------|------|
| **関数の定義** | `def 関数名(引数):` | `def add(a, b):` |
| **戻り値** | `return 値` | `return a + b` |
| **デフォルト引数** | 引数に初期値がある | `def f(x=10):` |
| **キーワード引数** | 名前を指定して渡す | `f(x=5, y=10)` |
| **`*args`** | 任意個の位置引数を受け取る | `def f(*args):` |
| **`**kwargs`** | 任意個のキーワード引数を受け取る | `def f(**kwargs):` |
| **ローカル変数** | 関数内で定義され、関数外では使えない | — |
| **グローバル変数** | 関数外で定義され、関数内から読める | — |

:::tip 核心の理解
関数はプログラミングの**基本ブロック**です。よいコードは、小さな関数を組み合わせて作ります。各関数は1つの仕事だけを担当し、その仕事をきちんとやるのが理想です。もし関数が 20 行を超えるなら、もっと小さな関数に分けられないか考えてみましょう。
:::
