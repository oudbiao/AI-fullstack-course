---
title: "1.5 フロー制御"
sidebar_position: 5
description: "条件分岐とループ構造を身につける"
---

# フロー制御

![Python フロー制御の実行パス図](/img/course/ch02-control-flow-paths-ja.png)

## この節の位置づけ

この節では、プログラムに「判断させる」ことと「繰り返し実行させる」ことを学びます。条件分岐とループは、あらゆる自動化スクリプト、データ処理フロー、モデル学習コードの土台です。これらを身につけると、コードはもう上から下へ順番に実行されるだけではなくなります。

## 学習目標

- `if/elif/else` による条件分岐を身につける
- `for` ループと `while` ループを身につける
- `break`、`continue` を使ってループを制御できるようになる
- 入れ子のロジックを含むプログラムを書けるようになる

---

## フロー制御とは？

ここまでに書いてきたコードは、すべて**上から下へ1行ずつ実行**されるものでした。でも実際のプログラムでは、判断したり、繰り返したりする必要があります。これがフロー制御です。

朝、家を出るときの判断を思い浮かべてみましょう。

```
もし 雨が降っていたら:
    傘を持つ
それ以外で もし 日差しが強ければ:
    帽子をかぶる
それ以外:
    そのまま出かける
```

これが**条件分岐**です。

次に、単語を覚えるときのことを考えてみましょう。

```
100回繰り返す:
    新しい単語を見る
    覚える
```

これが**ループ**です。

---

## 条件分岐: if / elif / else

### 基本の if

```python
temperature = 35

if temperature > 30:
    print("今日はとても暑いです。熱中症に注意！")
```

**文法ルール：**
1. `if` の後に条件式を書く
2. 条件の後には**コロン `:`** を付ける（初心者がよく忘れます）
3. 条件が成り立つときに実行するコードは**4スペースでインデント**する

### if...else

```python
age = 15

if age >= 18:
    print("あなたは成人です")
    print("この映画を見られます")
else:
    print("あなたは未成年です")
    print("保護者の同伴が必要です")
```

### if...elif...else

`elif` は "else if" の略で、複数の条件を順番に確認するときに使います。

```python
score = 85

if score >= 90:
    grade = "A（優秀）"
elif score >= 80:
    grade = "B（良い）"
elif score >= 70:
    grade = "C（普通）"
elif score >= 60:
    grade = "D（合格）"
else:
    grade = "F（不合格）"

print(f"あなたの点数: {score} 点、評価: {grade}")
# 出力: あなたの点数: 85 点、評価: B（良い）
```

:::caution 実行順に注意
Python は上から下へ順番に各条件をチェックします。**いずれかの条件が成り立ったら、対応するコードブロックを実行して、その後ろにあるすべての elif と else をスキップします**。そのため、条件の順番はとても重要です。

```python
score = 95

# 間違った順番 ❌
if score >= 60:
    print("合格")      # 95 >= 60 が成り立つので、ここがすぐ実行される
elif score >= 90:
    print("優秀")      # 実行されない！

# 正しい順番 ✅：厳しい条件から順に
if score >= 90:
    print("優秀")      # 95 >= 90 が成り立つので、ここが実行される
elif score >= 60:
    print("合格")
```
:::

### 条件分岐の省略形

```python
# 三項演算子（1行で簡単な if-else を書く）
age = 20
status = "成人" if age >= 18 else "未成年"
print(status)  # 成人

# 同じ意味
if age >= 18:
    status = "成人"
else:
    status = "未成年"
```

### 入れ子の if

条件の中にさらに条件を書けます。

```python
has_ticket = True
age = 15

if has_ticket:
    if age >= 18:
        print("入場してください")
    else:
        print("未成年は保護者の同伴が必要です")
else:
    print("まずチケットを購入してください")
```

ただし、入れ子が深すぎると読みづらくなるので、通常は3階層を超えないようにするのがおすすめです。

---

## for ループ

`for` ループは、シーケンス（リスト、文字列、範囲など）に含まれる各要素を**順番にたどる**ために使います。

### リストをたどる

```python
fruits = ["りんご", "バナナ", "オレンジ", "ぶどう"]

for fruit in fruits:
    print(f"私は{fruit}が好きです")

# 出力:
# 私はりんごが好きです
# 私はバナナが好きです
# 私はオレンジが好きです
# 私はぶどうが好きです
```

理解のしかたとしては、`for fruit in fruits` は「fruits の中の1つ1つの fruit について、下のコードを実行する」という意味です。

### 文字列をたどる

```python
word = "Python"

for char in word:
    print(char, end=" ")

# 出力: P y t h o n
```

### range() 関数

`range()` は数字の並びを作る関数で、`for` ループの相棒としてよく使われます。

```python
# range(5) は 0, 1, 2, 3, 4 を作る
for i in range(5):
    print(i, end=" ")
# 出力: 0 1 2 3 4

# range(start, stop) は start から stop-1 まで
for i in range(1, 6):
    print(i, end=" ")
# 出力: 1 2 3 4 5

# range(start, stop, step) は step 付き
for i in range(0, 10, 2):
    print(i, end=" ")
# 出力: 0 2 4 6 8

# 逆順
for i in range(5, 0, -1):
    print(i, end=" ")
# 出力: 5 4 3 2 1
```

### 実践例: 1 から 100 までの合計を求める

```python
total = 0
for i in range(1, 101):
    total += i
print(f"1 から 100 までの合計は: {total}")  # 5050
```

### enumerate()：インデックスと値を同時に取得する

```python
students = ["張三", "李四", "王五"]

# 普通の書き方
for i in range(len(students)):
    print(f"{i+1}番目: {students[i]}")

# より Pythonic な書き方：enumerate を使う
for i, name in enumerate(students):
    print(f"{i+1}番目: {name}")

# 開始番号を指定する
for i, name in enumerate(students, start=1):
    print(f"{i}番目: {name}")
```

---

## while ループ

`while` ループは、**条件が成り立っている間**ずっと実行され、条件が成り立たなくなると止まります。

### 基本の使い方

```python
count = 0

while count < 5:
    print(f"現在のカウント: {count}")
    count += 1   # 条件を更新するのを忘れないでください！

print("ループ終了")

# 出力:
# 現在のカウント: 0
# 現在のカウント: 1
# 現在のカウント: 2
# 現在のカウント: 3
# 現在のカウント: 4
# ループ終了
```

:::caution 無限ループに注意！
条件変数を更新し忘れると、ループはいつまでも終わりません。

```python
# 無限ループの例（実行しないでください！）
count = 0
while count < 5:
    print(count)
    # count += 1 を忘れているので、count はずっと 0 のまま。ループは終わらない
```

もし無限ループに入ってしまったら、`Ctrl+C` で強制終了してください。
:::

### while がよく使われる場面

`while` は、**ループ回数が決まっていない**場合に向いています。

```python
# 場面: 数当てゲーム
import random

target = random.randint(1, 100)
guess = 0
attempts = 0

print("1 から 100 までの数字を1つ考えました。当ててみてください！")

while guess != target:
    guess = int(input("あなたの予想: "))
    attempts += 1

    if guess < target:
        print("小さすぎます！")
    elif guess > target:
        print("大きすぎます！")
    else:
        print(f"おめでとう！{attempts}回で正解しました")
```

### for と while はどう選ぶ？

| 場面 | おすすめ | 理由 |
|------|------|------|
| リスト/文字列をたどる | `for` | 自然に向いている |
| 回数が決まっているループ | `for + range()` | 簡潔でわかりやすい |
| 回数が決まっていないループ | `while` | 柔軟に制御できる |
| ある条件が成り立つのを待つ | `while` | 直感的 |

**経験則: `for` が使えるなら `for` を使いましょう。より安全です（無限ループになりにくいです）。**

---

## break と continue

### break: ループをすぐ終了する

```python
# 最初の偶数を見つけたら止める
numbers = [1, 3, 7, 4, 9, 2]

for num in numbers:
    if num % 2 == 0:
        print(f"最初の偶数を見つけました: {num}")
        break
    print(f"{num} は偶数ではありません。続けて探します...")

# 出力:
# 1 は偶数ではありません。続けて探します...
# 3 は偶数ではありません。続けて探します...
# 7 は偶数ではありません。続けて探します...
# 最初の偶数を見つけました: 4
```

### continue: 今回のループだけ飛ばして、次へ進む

```python
# 奇数だけを表示して、偶数は飛ばす
for i in range(1, 11):
    if i % 2 == 0:
        continue   # 偶数をスキップ
    print(i, end=" ")

# 出力: 1 3 5 7 9
```

### break と continue の違い

```python
# break: ループを完全に抜ける
for i in range(10):
    if i == 5:
        break       # 5 になったらループ全体を終了
    print(i, end=" ")
# 出力: 0 1 2 3 4

# continue: 今回だけ飛ばして、次へ進む
for i in range(10):
    if i == 5:
        continue    # 5 を飛ばして、6, 7, 8, 9 へ進む
    print(i, end=" ")
# 出力: 0 1 2 3 4 6 7 8 9
```

---

## ループの else

Python のループには、少し特別な `else` があります。ループが**正常に終了したとき**、つまり `break` で止められなかったときに実行されます。

```python
# ある数が素数かどうかを調べる
num = 17

for i in range(2, num):
    if num % i == 0:
        print(f"{num} は素数ではありません。{i} で割り切れます")
        break
else:
    # ループが break で終わらなかったので、因数が見つからなかった
    print(f"{num} は素数です")

# 出力: 17 は素数です
```

---

## 入れ子ループ

ループの中に、さらにループを書けます。

```python
# 九九表を表示する
for i in range(1, 10):
    for j in range(1, i + 1):
        print(f"{j}×{i}={i*j}", end="\t")
    print()   # 各行の終わりで改行
```

出力：

```
1×1=1
1×2=2	2×2=4
1×3=3	2×3=6	3×3=9
...
1×9=9	2×9=18	3×9=27	...	9×9=81
```

---

## 総合演習

### 例1: AI モデル学習の流れをシミュレートする

```python
import random

print("=== モデル学習開始 ===")
print(f"{'Epoch':<10}{'Loss':<15}{'Accuracy':<15}{'Status'}")
print("-" * 50)

loss = 2.5
accuracy = 0.10

for epoch in range(1, 21):
    # 学習を模擬する: 損失は徐々に下がり、精度は徐々に上がる
    loss *= random.uniform(0.85, 0.95)
    accuracy = min(accuracy + random.uniform(0.03, 0.06), 1.0)

    # 学習状態を判断する
    if accuracy >= 0.95:
        status = "✅ 達成"
    elif accuracy >= 0.80:
        status = "📈 良好"
    else:
        status = "🔄 学習中"

    print(f"{epoch:<10}{loss:<15.4f}{accuracy:<15.2%}{status}")

    # 精度が 98% に達したら早期終了
    if accuracy >= 0.98:
        print(f"\n早期終了！第 {epoch} エポックで目標精度に到達しました")
        break
else:
    print(f"\n学習完了！最終精度: {accuracy:.2%}")
```

### 例2: パスワード強度チェッカー

```python
password = input("パスワードを入力してください: ")

has_upper = False    # 大文字があるか
has_lower = False    # 小文字があるか
has_digit = False    # 数字があるか
has_special = False  # 特殊文字があるか

for char in password:
    if char.isupper():
        has_upper = True
    elif char.islower():
        has_lower = True
    elif char.isdigit():
        has_digit = True
    else:
        has_special = True

# 強度スコアを計算する
score = 0
if len(password) >= 8:
    score += 1
if has_upper:
    score += 1
if has_lower:
    score += 1
if has_digit:
    score += 1
if has_special:
    score += 1

# 結果を出力する
print(f"\nパスワード強度: {'★' * score}{'☆' * (5 - score)} ({score}/5)")

if score <= 2:
    print("弱いパスワードです！強化をおすすめします")
elif score <= 4:
    print("中くらいの強さです")
else:
    print("強いパスワードです！")
```

---

## やってみよう

### 練習1: FizzBuzz

これは定番のプログラミング面接問題です。

1 から 50 までの数字を表示してください。ただし、
- 3 で割り切れるなら "Fizz" を表示する
- 5 で割り切れるなら "Buzz" を表示する
- 3 と 5 の両方で割り切れるなら "FizzBuzz" を表示する
- それ以外は数字そのものを表示する

```python
for i in range(1, 51):
    if i % 15 == 0:
        print("FizzBuzz")
    elif i % 3 == 0:
        print("Fizz")
    elif i % 5 == 0:
        print("Buzz")
    else:
        print(i)
```

ヒント: まず 15 で割り切れるか（3 と 5 の公倍数）を判定し、その後で 3 と 5 を判定します。

### 練習2: 数当てゲーム（回数制限あり）

数当てゲームを改良して、最大 7 回までしか予想できないようにしましょう。7 回を超えたら失敗です。

```python
import random
target = random.randint(1, 100)
max_attempts = 7

for attempt in range(1, max_attempts + 1):
    guess = target if attempt == 1 else 1
    print(f"{attempt} 回目: {guess}")
    if guess == target:
        print("正解です！")
        break
    elif guess < target:
        print("小さすぎます")
    else:
        print("大きすぎます")
else:
    print(f"失敗です。答えは {target} でした。")
```

### 練習3: 三角形を描く

ループを使って次の図形を表示してください。

```
*
**
***
****
*****
```

次に、逆三角形も表示してみましょう。

```
*****
****
***
**
*
```

### 練習4: 素数を求める

1 から 100 までの素数をすべて表示してください。

ヒント: 素数とは、1 より大きい自然数で、1 と自分自身でしか割り切れない数です。

---

## まとめ

| 文法 | 用途 | 重要ポイント |
|------|------|--------|
| `if/elif/else` | 条件分岐 | 条件は上から順にチェック。コロンとインデントを忘れない |
| `for...in` | シーケンスをたどる | `range()`、リスト、文字列と一緒に使う |
| `while` | 条件ループ | 条件の更新を忘れず、無限ループを避ける |
| `break` | ループを終了する | ループ全体をすぐ抜ける |
| `continue` | 今回を飛ばす | 現在の反復を飛ばして次へ進む |
| `range()` | 数字の並びを作る | `range(start, stop, step)` |

:::tip コア理解
フロー制御はプログラミングの**骨組み**です。変数はデータ、演算子は操作、そしてフロー制御は「どんな条件で何をするか」「何回やるか」を決めます。フロー制御を身につけると、ロジックのあるプログラムが書けるようになります。
:::
