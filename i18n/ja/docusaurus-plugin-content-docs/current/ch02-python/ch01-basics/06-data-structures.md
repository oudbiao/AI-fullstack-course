---
title: "2.1.6 データ構造"
sidebar_position: 6
description: "Python の4大データ構造：リスト、タプル、辞書、集合をマスターする"
---

# 2.1.6 データ構造

![Python データ構造比較図](/img/course/ch02-data-structures-comparison-ja.webp)

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

この節では、ひとまとまりのデータをどう整理するかを学びます。リスト、タプル、辞書、集合は、この先のクローラー、データ分析、機械学習のサンプル処理、API レスポンスの解析までずっと使います。大事なのは、それぞれの構造が何を入れるのに向いているか、どんなときにどれを使うかを知ることです。

## 学習目標

- リスト（list）の作成と基本操作を理解する
- タプル（tuple）の特徴と使いどころを理解する
- 辞書（dict）のキー・バリュー操作を身につける
- 集合（set）の重複削除と集合演算を知る
- 場面に応じて適切なデータ構造を選べるようになる

---

## なぜデータ構造が必要なの？

ここまで学んだ変数は、1つの値しか入れられません。でも実際の場面では、**ひとまとまりのデータ**を扱うことがよくあります。

- 100回分の API レイテンシ
- あるモデルのすべてのパラメータ
- ユーザーの個人情報（名前、年齢、メールアドレス……）

データ構造は、**複数のデータを整理して保存する**ための入れ物です。

Python には 4 種類の組み込みデータ構造があります。

| データ構造 | 記号 | 順序あり | 可変 | 重複可 | 典型的な用途 |
|---------|------|------|------|---------|---------|
| **リスト** list | `[]` | ✅ | ✅ | ✅ | 順序のあるデータの集合 |
| **タプル** tuple | `()` | ✅ | ❌ | ✅ | 変更しないデータ |
| **辞書** dict | `{}` | ✅ | ✅ | キーは重複不可 | キー・バリューの対応 |
| **集合** set | `{}` | ❌ | ✅ | ❌ | 重複削除、集合演算 |

---

## リスト（list）—— 最もよく使うデータ構造

リストは、**伸び縮みする引き出し**のようなものです。中にいろいろなものを入れられて、いつでも追加・削除・変更できます。

### リストを作る

```python
# リストを作成
latencies_ms = [120, 95, 240, 180, 310]
features = ["ログイン API", "RAG デモ", "グラフビュー"]
mixed = [1, "hello", 3.14, True]   # 型を混ぜることもできる（おすすめはしない）
empty = []                          # 空のリスト

print(type(latencies_ms))  # <class 'list'>
print(len(latencies_ms))   # 5
```

### 要素にアクセスする（インデックス）

```python
service_queue = ["ログイン API", "検索 API", "Worker", "ダッシュボード", "ドキュメントサイト"]
#                 0             1          2         3                4
#                -5            -4         -3        -2               -1

print(service_queue[0])     # ログイン API（最初のサービス）
print(service_queue[2])     # Worker（3番目のサービス）
print(service_queue[-1])    # ドキュメントサイト（最後のサービス）
print(service_queue[-2])    # ダッシュボード（後ろから2番目のサービス）
```

### スライス

```python
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:5])    # [2, 3, 4]（インデックス 2 から 4）
print(numbers[:3])     # [0, 1, 2]（最初の3つ）
print(numbers[7:])     # [7, 8, 9]（インデックス 7 から最後まで）
print(numbers[::2])    # [0, 2, 4, 6, 8]（1つおきに取り出す）
print(numbers[::-1])   # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]（逆順）
```

### 要素を変更する

```python
latencies_ms = [120, 95, 240, 180, 310]

# 1つの要素を変更
latencies_ms[2] = 210
print(latencies_ms)  # [120, 95, 210, 180, 310]

# 複数の要素を変更（スライスを使う）
latencies_ms[1:3] = [100, 180]
print(latencies_ms)  # [120, 100, 180, 180, 310]
```

### 要素を追加する

```python
tasks = ["ログインフォームを作る", "API テストを書く"]

# 末尾に追加
tasks.append("エラー状態を追加する")
print(tasks)  # ['ログインフォームを作る', 'API テストを書く', 'エラー状態を追加する']

# 指定した位置に挿入
tasks.insert(1, "認証フローをレビューする")
print(tasks)  # ['ログインフォームを作る', '認証フローをレビューする', 'API テストを書く', 'エラー状態を追加する']

# 複数の要素を追加
tasks.extend(["README を更新する", "デモを録画する"])
print(tasks)  # ['ログインフォームを作る', '認証フローをレビューする', 'API テストを書く', 'エラー状態を追加する', 'README を更新する', 'デモを録画する']
```

### 要素を削除する

```python
tasks = ["ログインフォームを作る", "API テストを書く", "エラー状態を追加する", "認証フローをレビューする", "デモを録画する"]

# 値で削除（最初に一致したものを削除）
tasks.remove("エラー状態を追加する")
print(tasks)  # ['ログインフォームを作る', 'API テストを書く', '認証フローをレビューする', 'デモを録画する']

# インデックスで削除
deleted = tasks.pop(1)     # インデックス 1 の要素を削除して返す
print(deleted)             # API テストを書く
print(tasks)               # ['ログインフォームを作る', '認証フローをレビューする', 'デモを録画する']

# 最後の要素を削除
last = tasks.pop()
print(last)    # デモを録画する

# インデックスで削除（返り値は不要）
del tasks[0]
print(tasks)  # ['認証フローをレビューする']
```

### リストでよく使う操作

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5]

# ソート
numbers.sort()
print(numbers)  # [1, 1, 2, 3, 4, 5, 5, 6, 9]

# 降順ソート
numbers.sort(reverse=True)
print(numbers)  # [9, 6, 5, 5, 4, 3, 2, 1, 1]

# 元のリストを変更しないソート
original = [3, 1, 4, 1, 5]
sorted_list = sorted(original)
print(original)    # [3, 1, 4, 1, 5]（元のリストは変わらない）
print(sorted_list) # [1, 1, 3, 4, 5]

# 反転
numbers = [1, 2, 3, 4, 5]
numbers.reverse()
print(numbers)  # [5, 4, 3, 2, 1]

# 検索
print(numbers.index(3))    # 2（要素 3 のインデックス）
print(numbers.count(5))    # 1（要素 5 の出現回数）
print(3 in numbers)        # True

# 集計
latencies_ms = [120, 95, 240, 180, 310]
print(len(latencies_ms))    # 5
print(sum(latencies_ms))    # 945
print(max(latencies_ms))    # 310
print(min(latencies_ms))    # 95
print(sum(latencies_ms) / len(latencies_ms))  # 189.0（平均レイテンシ）
```

### リスト内包表記（とても Python らしい書き方！）

リスト内包表記は、新しいリストを簡潔に作る方法です。

```python
# ふつうの書き方
squares = []
for i in range(1, 6):
    squares.append(i ** 2)
print(squares)  # [1, 4, 9, 16, 25]

# リスト内包表記（一行で書ける！）
squares = [i ** 2 for i in range(1, 6)]
print(squares)  # [1, 4, 9, 16, 25]

# 条件付きのリスト内包表記
even_squares = [i ** 2 for i in range(1, 11) if i % 2 == 0]
print(even_squares)  # [4, 16, 36, 64, 100]

# 実践例：機能 slug を正規化する
raw_slugs = ["  Login API  ", "RAG DEMO", "  Chart View "]
clean_slugs = [slug.strip().lower().replace(" ", "-") for slug in raw_slugs]
print(clean_slugs)  # ['login-api', 'rag-demo', 'chart-view']
```

:::tip リスト内包表記の形
`[式 for 変数 in イテラブル if 条件]`

日本語にすると、「条件を満たす各要素について式を計算して、新しいリストに入れる」という意味です。
:::

---

## タプル（tuple）—— 変更できないリスト

タプルとリストはほとんど同じですが、違いは1つです。**タプルは作成後に変更できません。**

### タプルを作る

```python
# 丸括弧で作成
point = (3, 4)
colors = ("赤", "緑", "青")
single = (42,)          # 要素が1つだけのときは、カンマが必要！
empty = ()

# 実は丸括弧は省略できる
coordinates = 3, 4      # これもタプル
print(type(coordinates)) # <class 'tuple'>
```

### タプルの操作

```python
colors = ("赤", "緑", "青", "黄", "紫")

# アクセス（リストと同じ）
print(colors[0])     # 赤
print(colors[-1])    # 紫
print(colors[1:3])   # ('緑', '青')

# 走査
for color in colors:
    print(color)

# 検索
print(len(colors))          # 5
print("赤" in colors)       # True
print(colors.count("赤"))   # 1
print(colors.index("青"))   # 2

# ただし変更はできない！
# colors[0] = "黒"  # エラー！TypeError: 'tuple' object does not support item assignment
```

### タプルのアンパック

```python
# タプルの値をそれぞれ別の変数に代入する
point = (10, 20)
x, y = point
print(f"x={x}, y={y}")  # x=10, y=20

# 関数が複数の値を返すとき、実際にはタプルが返っている
def get_task_and_hours():
    return "ログイン API", 8

task, hours = get_task_and_hours()
print(f"{task}, {hours}時間")  # ログイン API, 8時間

# * を使って余った値をまとめる
first, *rest = [1, 2, 3, 4, 5]
print(first)  # 1
print(rest)   # [2, 3, 4, 5]
```

### いつタプルを使う？

- データを変更されたくないとき（例：座標、RGB の色の値）
- 辞書のキーに使いたいとき（リストは辞書のキーにできないが、タプルはできる）
- 関数で複数の値を返したいとき

---

## 辞書（dict）—— キー・バリューで保存する

辞書は、Python の**とても重要なデータ構造の1つ**です。**キー（key）** で **値（value）** を探します。まるで実際の辞書で、単語から意味を引くようなものです。

### 辞書を作る

```python
# 波括弧で作成
task = {
    "name": "ログイン API",
    "owner": "Mina",
    "status": "進行中",
    "hours": [2, 3, 3]
}

# 空の辞書
empty = {}

# dict() で作成
config = dict(learning_rate=0.001, epochs=100, batch_size=32)
print(config)  # {'learning_rate': 0.001, 'epochs': 100, 'batch_size': 32}

print(type(task))  # <class 'dict'>
```

### 値にアクセスする

```python
task = {"name": "ログイン API", "owner": "Mina", "status": "進行中"}

# 方法1：[] でアクセス
print(task["name"])   # ログイン API
# print(task["deadline"])  # エラー！KeyError: 'deadline'

# 方法2：.get() でアクセス（より安全）
print(task.get("owner"))    # Mina
print(task.get("deadline"))   # None（存在しないときは None を返し、エラーにならない）
print(task.get("deadline", "未設定"))  # 未設定（存在しないときはデフォルト値を返す）
```

:::tip .get() の使用がおすすめ
キーがあるか分からないときは、`[]` よりも `.get()` のほうが安全です。プログラムが止まるのを防げます。
:::

### 追加と変更

```python
task = {"name": "ログイン API", "status": "未着手"}

# 新しいキー・バリューを追加
task["owner"] = "Mina"
task["repo"] = "portfolio-api"

# 既存の値を変更
task["status"] = "進行中"

print(task)
# {'name': 'ログイン API', 'status': '進行中', 'owner': 'Mina', 'repo': 'portfolio-api'}

# まとめて更新
task.update({"status": "完了", "hours": 8})
print(task)
```

### 削除

```python
task = {"name": "ログイン API", "status": "完了", "owner": "Mina"}

# 指定したキーを削除
del task["owner"]
print(task)  # {'name': 'ログイン API', 'status': '完了'}

# pop：削除して値を返す
status = task.pop("status")
print(status)  # 完了
print(task)    # {'name': 'ログイン API'}
```

### 辞書を走査する

```python
task_hours = {"ログイン API": 8, "RAG デモ": 12, "グラフビュー": 5}

# キーを走査
for task in task_hours:
    print(task)

# 値を走査
for hours in task_hours.values():
    print(hours)

# キー・バリューを走査（最もよく使う）
for task, hours in task_hours.items():
    print(f"{task}: {hours} 時間")

# 出力:
# ログイン API: 8 時間
# RAG デモ: 12 時間
# グラフビュー: 5 時間
```

### 辞書内包表記

```python
# 数字から平方への対応表を作る
squares = {x: x**2 for x in range(1, 6)}
print(squares)  # {1: 1, 2: 4, 3: 9, 4: 16, 5: 25}

# 辞書を絞り込む
task_hours = {"ログイン API": 8, "バグ修正": 3, "RAG デモ": 12, "ドキュメント": 2}
large_tasks = {name: hours for name, hours in task_hours.items() if hours >= 8}
print(large_tasks)  # {'ログイン API': 8, 'RAG デモ': 12}
```

### 実践例：文字の出現回数を数える

```python
text = "hello world"
char_count = {}

for char in text:
    if char in char_count:
        char_count[char] += 1
    else:
        char_count[char] = 1

print(char_count)
# {'h': 1, 'e': 1, 'l': 3, 'o': 2, ' ': 1, 'w': 1, 'r': 1, 'd': 1}
```

---

## 集合（set）—— 重複削除の強い味方

集合は、**順序がなく、重複しない**要素の集まりです。

### 集合を作る

```python
# 波括弧で作成
task_tags = {"api", "ui", "testing", "api"}  # 重複は自動で削除される
print(task_tags)  # {'testing', 'ui', 'api'}（順序は変わることがある）

# リストから作成（重複を削除！）
modules = ["api", "api", "ui", "worker", "ui", "db"]
unique_modules = set(modules)
print(unique_modules)  # {'api', 'db', 'ui', 'worker'}（順序は変わることがある）

# 注意：空の集合は set() を使う。{} ではない
empty_set = set()     # 空集合
empty_dict = {}       # これは空の辞書！

print(type(task_tags))   # <class 'set'>
```

### 集合の操作

```python
a = {1, 2, 3, 4, 5}
b = {4, 5, 6, 7, 8}

# 積集合（両方にあるもの）
print(a & b)          # {4, 5}
print(a.intersection(b))

# 和集合（まとめて、重複なし）
print(a | b)          # {1, 2, 3, 4, 5, 6, 7, 8}
print(a.union(b))

# 差集合（a にはあるが b にはないもの）
print(a - b)          # {1, 2, 3}
print(a.difference(b))

# 対称差集合（それぞれにしかないもの）
print(a ^ b)          # {1, 2, 3, 6, 7, 8}
```

### 実践での使い方

```python
# 例：フロントエンドとバックエンドの両方に関わるタスクを探す
frontend_tasks = {"ログイン UI", "グラフビュー", "設定ページ", "テーマ切替"}
backend_tasks = {"ログイン API", "グラフビュー", "監査ログ", "設定ページ"}

both = frontend_tasks & backend_tasks
print(f"両方に関わるタスク: {sorted(both)}")

only_frontend = frontend_tasks - backend_tasks
print(f"フロントエンドのみのタスク: {sorted(only_frontend)}")

all_tasks = frontend_tasks | backend_tasks
print(f"関連タスクすべて: {sorted(all_tasks)}")
```

---

## データ構造の選び方ガイド

| 目的 | おすすめ | 理由 |
|------|------|------|
| 順序のある集合で、追加・削除・変更をしたい | **リスト** | いちばん汎用的な入れ物 |
| データを変更したくない | **タプル** | 変更不可で安全 |
| キーで値を探したい | **辞書** | O(1) で高速に検索できる |
| 重複を取り除きたい | **集合** | 自動で重複を削除する |
| 出現回数を数えたい | **辞書** | キーを要素、値をカウントにできる |
| 要素が存在するか調べたい | **集合/辞書** | リストよりずっと速い |

---

## 実践練習

### 練習 1：API レイテンシ集計

```python
latencies_ms = [120, 95, 240, 180, 310, 150, 88, 205, 260, 170]

# 1. 最大レイテンシ、最小レイテンシ、平均レイテンシを計算する
# 2. 200 ms を超えるレイテンシをすべて取り出す（リスト内包表記を使う）
# 3. レイテンシを高い順に並べる
```

### 練習 2：サービス担当者ディレクトリ

辞書を使って、簡単なサービス担当者ディレクトリを作ってみましょう。

```python
owners = {}

# 1. 3つのサービスを追加する（サービス名 → 担当者メール）
# 2. あるサービス担当者のメールを調べる
# 3. あるサービス担当者のメールを変更する
# 4. 1つのサービスを削除する
# 5. すべてのサービス担当者を表示する
```

### 練習 3：イベント語の出現回数を数える

```python
text = "api error api timeout worker error api"

# それぞれのイベント語が何回出てくるかを数える
# ヒント：まず split() でリストに分けてから、辞書で数える
```

### 練習 4：リストの重複削除（順序を保つ）

```python
numbers = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]

# 重複を取り除くが、元の順序は保つ
# 期待する出力: [3, 1, 4, 5, 9, 2, 6]
# ヒント：すでに出てきた要素を集合で記録する
```

<details>
<summary>参考実装と解説</summary>

1. レイテンシの統計は、最大値 `310`、最小値 `88`、平均 `181.8` です。`200` ms 超は `[240, 310, 205, 260]`、降順では `[310, 260, 240, 205, ...]` から始まります。
2. サービス担当者ディレクトリはキーで追加、更新、削除します。例: `owners["ログイン API"] = "api-owner@example.com"`。
3. サンプルのイベント語頻度では、`api` が `3`、`error` が `2`、`timeout` と `worker` が 1 回です。
4. 順序を保った重複削除の結果は `[3, 1, 4, 5, 9, 2, 6]` です。`seen` セットと結果リストを組み合わせます。
5. 順序が必要ならリスト、検索なら辞書、所属判定や重複削除なら集合、固定レコードならタプルを選びます。

</details>

---

## まとめ

| データ構造 | 作り方 | 特徴 | よくある場面 |
|---------|---------|------|---------|
| **リスト** | `[1, 2, 3]` | 順序あり、可変、重複可 | ひとまとまりの同種データを保存する |
| **タプル** | `(1, 2, 3)` | 順序あり、変更不可 | 座標、複数の戻り値 |
| **辞書** | `{"a": 1}` | キー・バリュー、キー重複不可 | 設定、対応関係 |
| **集合** | `{1, 2, 3}` | 順序なし、重複なし | 重複削除、集合演算 |

:::tip 核心の理解
データ構造を選ぶのは、収納道具を選ぶのと似ています。リストは**引き出し**（順番に並べる）、辞書は**ラベル付きの棚**（ラベルで探す）、集合は**ふるい**（自動で重複を取る）、タプルは**密封袋**（入れたあとに変更しない）です。道具を正しく選べば、作業はぐっと楽になります。
:::
