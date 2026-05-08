---
title: "2.1.7 関数の基礎"
sidebar_position: 7
description: "関数の定義、引数、戻り値、スコープを理解する"
---

# 2.1.7 関数の基礎

![関数の呼び出し、引数、スコープの図](/img/course/ch02-function-call-scope-ja.webp)

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

たとえば、データ処理スクリプトの中で平均値を何回も計算する場面を考えてみましょう。

```python
# 1回目の計算
scores1 = [85, 92, 78, 95, 88]
total1 = sum(scores1)
avg1 = total1 / len(scores1)
print(f"平均点: {avg1:.1f}")

# 2回目の計算（同じロジックをもう一度書く）
scores2 = [90, 85, 92, 88, 95, 87]
total2 = sum(scores2)
avg2 = total2 / len(scores2)
print(f"平均点: {avg2:.1f}")

# 3回目の計算（さらにもう一度……）
scores3 = [75, 80, 68, 72, 88]
total3 = sum(scores3)
avg3 = total3 / len(scores3)
print(f"平均点: {avg3:.1f}")
```

同じロジックを 3 回も書いています。もし後で計算方法を変えたくなったら（たとえば最高点と最低点を除くなど）、3 か所を直す必要があります。

関数を使うと、こうなります。

```python
def calculate_average(scores):
    """平均点を計算する"""
    return sum(scores) / len(scores)

# これなら1行でOK
print(f"平均点: {calculate_average([85, 92, 78, 95, 88]):.1f}")
print(f"平均点: {calculate_average([90, 85, 92, 88, 95, 87]):.1f}")
print(f"平均点: {calculate_average([75, 80, 68, 72, 88]):.1f}")
```

**関数の主な価値：**

| メリット | 説明 |
|------|------|
| **再利用** | 1回書けば、何度でも使える |
| **抽象化** | 複雑なロジックを関数名の後ろに隠せる。呼び出す側は「何をするか」だけ分かればよい |
| **保守性** | 修正が必要なとき、1か所だけ直せばよい |
| **読みやすさ** | 関数名そのものがコメントの役割を持つ。`calculate_average(scores)` は一目で分かる |

---

## 関数の定義と呼び出し

### 基本構文

```python
def greet(name):
    """誰かにあいさつする"""  # docstring（ドキュメンテーション文字列）、関数の説明
    print(f"こんにちは、{name}！Python の学習へようこそ！")

# 関数を呼び出す
greet("小明")     # こんにちは、小明！Python の学習へようこそ！
greet("小紅")     # こんにちは、小紅！Python の学習へようこそ！
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

result = greet("小明")   # 表示: こんにちは、小明！
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
def describe_pet(animal, name):
    print(f"私は{animal}を飼っています。名前は{name}です")

describe_pet("猫", "ミミ")   # 私は猫を飼っています。名前はミミです
describe_pet("ミミ", "猫")   # 私はミミを飼っています。名前は猫です —— 順番が逆！
```

### キーワード引数

引数名を指定して値を渡します。順番を気にしなくて大丈夫です。

```python
def describe_pet(animal, name):
    print(f"私は{animal}を飼っています。名前は{name}です")

# キーワード引数なら順番は関係ない
describe_pet(name="ミミ", animal="猫")   # 私は猫を飼っています。名前はミミです
describe_pet(animal="犬", name="ワンちゃん")   # 私は犬を飼っています。名前はワンちゃんです
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

print_info(name="小明", age=20, city="北京")
# name: 小明
# age: 20
# city: 北京
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

### 例 1: 成績分析ツール

```python
def analyze_scores(scores, subject="不明な科目"):
    """
    1組の成績を分析して、統計情報を返す。

    引数:
        scores: 成績のリスト
        subject: 科目名
    戻り値:
        統計情報を含む辞書
    """
    if not scores:
        return {"error": "成績リストが空です"}

    avg = sum(scores) / len(scores)
    passed = [s for s in scores if s >= 60]
    failed = [s for s in scores if s < 60]

    return {
        "subject": subject,
        "count": len(scores),
        "average": round(avg, 1),
        "max": max(scores),
        "min": min(scores),
        "pass_rate": f"{len(passed) / len(scores):.1%}",
        "passed": len(passed),
        "failed": len(failed)
    }

def print_report(stats):
    """整形した成績レポートを表示する"""
    print(f"\n{'='*30}")
    print(f"  {stats['subject']} 成績分析レポート")
    print(f"{'='*30}")
    print(f"  受験人数: {stats['count']}")
    print(f"  平均点:   {stats['average']}")
    print(f"  最高点:   {stats['max']}")
    print(f"  最低点:   {stats['min']}")
    print(f"  合格率:   {stats['pass_rate']}")
    print(f"  合格者数: {stats['passed']}")
    print(f"  不合格:   {stats['failed']}")
    print(f"{'='*30}")

# 使用例
math_scores = [85, 92, 45, 78, 95, 55, 88, 72, 60, 98]
english_scores = [70, 55, 88, 45, 92, 78, 65, 82, 90, 58]

math_stats = analyze_scores(math_scores, "数学")
english_stats = analyze_scores(english_scores, "英語")

print_report(math_stats)
print_report(english_stats)
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
