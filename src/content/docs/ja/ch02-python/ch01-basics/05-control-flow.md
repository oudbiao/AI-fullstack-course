---
title: "2.1.5 フロー制御"
description: "条件分岐とループ構造を身につける"
sidebar:
  order: 5
---
![Python フロー制御の実行パス図](/img/course/ch02-control-flow-paths-ja.webp)

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
failed_tests = 3

if failed_tests > 0:
    print("リリースを止めて、失敗したテストを確認します。")
```

**文法ルール：**
1. `if` の後に条件式を書く
2. 条件の後には**コロン `:`** を付ける（初心者がよく忘れます）
3. 条件が成り立つときに実行するコードは**4スペースでインデント**する

### if...else

```python
all_checks_passed = False

if all_checks_passed:
    print("ビルドはデプロイできます")
    print("リリースノートを書きます")
else:
    print("ビルドはレビュー状態のままにします")
    print("失敗したチェックを先に直します")
```

### if...elif...else

`elif` は "else if" の略で、複数の条件を順番に確認するときに使います。

```python
latency_ms = 185

if latency_ms < 100:
    status = "高速"
elif latency_ms < 200:
    status = "正常"
elif latency_ms < 500:
    status = "遅め"
else:
    status = "重大"

print(f"API レイテンシ: {latency_ms} ms、状態: {status}")
# 出力: API レイテンシ: 185 ms、状態: 正常
```

:::caution[実行順に注意]
Python は上から下へ順番に各条件をチェックします。**いずれかの条件が成り立ったら、対応するコードブロックを実行して、その後ろにあるすべての elif と else をスキップします**。そのため、条件の順番はとても重要です。

```python
latency_ms = 95

# 間違った順番 ❌
if latency_ms < 500:
    print("要確認")     # 95 < 500 が成り立つので、ここがすぐ実行される
elif latency_ms < 100:
    print("高速")       # 実行されない！

# 正しい順番 ✅：厳しい条件から順に
if latency_ms < 100:
    print("高速")       # 95 < 100 が成り立つので、ここが実行される
elif latency_ms < 500:
    print("要確認")
```
:::
### 条件分岐の省略形

```python
# 三項演算子（1行で簡単な if-else を書く）
latency_ms = 185
status = "予算内" if latency_ms <= 200 else "要確認"
print(status)  # 予算内

# 同じ意味
if latency_ms <= 200:
    status = "予算内"
else:
    status = "要確認"
```

### 入れ子の if

条件の中にさらに条件を書けます。

```python
has_approval = True
all_tests_passed = False

if has_approval:
    if all_tests_passed:
        print("このビルドをデプロイします")
    else:
        print("テストスイートの通過を待ちます")
else:
    print("先にリリース承認を依頼します")
```

ただし、入れ子が深すぎると読みづらくなるので、通常は3階層を超えないようにするのがおすすめです。

---

## for ループ

`for` ループは、シーケンス（リスト、文字列、範囲など）に含まれる各要素を**順番にたどる**ために使います。

### リストをたどる

```python
services = ["ログイン API", "検索 API", "Worker", "ダッシュボード"]

for service in services:
    print(f"{service} をチェックします")

# 出力:
# ログイン API をチェックします
# 検索 API をチェックします
# Worker をチェックします
# ダッシュボード をチェックします
```

理解のしかたとしては、`for service in services` は「services の中の1つ1つの service について、下のコードを実行する」という意味です。

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

### 実践例: レビュー時間を合計する

```python
total_minutes = 0
for day in range(1, 6):
    total_minutes += 30
print(f"5日分のレビュー時間: {total_minutes} 分")  # 150
```

### enumerate()：インデックスと値を同時に取得する

```python
tasks = ["ログインフォーム設計", "API エンドポイント実装", "スモークテスト作成"]

# 普通の書き方
for i in range(len(tasks)):
    print(f"タスク {i+1}: {tasks[i]}")

# より Pythonic な書き方：enumerate を使う
for i, task in enumerate(tasks):
    print(f"タスク {i+1}: {task}")

# 開始番号を指定する
for i, task in enumerate(tasks, start=1):
    print(f"タスク {i}: {task}")
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

:::caution[無限ループに注意！]
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
# 場面: バックグラウンドジョブの完了を待つ
job_status = "queued"
poll_count = 0

while job_status != "finished":
    poll_count += 1
    print(f"{poll_count} 回目のポーリング: {job_status}")

    if poll_count == 1:
        job_status = "running"
    elif poll_count == 2:
        job_status = "finished"

print(f"ジョブは {poll_count} 回のポーリング後に完了しました")
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
# 最初の遅いリクエストを見つけたら止める
latencies_ms = [120, 145, 310, 180, 260]

for latency_ms in latencies_ms:
    if latency_ms > 250:
        print(f"最初の遅いリクエスト: {latency_ms} ms")
        break
    print(f"{latency_ms} ms は範囲内です。続けて確認します...")

# 出力:
# 120 ms は範囲内です。続けて確認します...
# 145 ms は範囲内です。続けて確認します...
# 最初の遅いリクエスト: 310 ms
```

### continue: 今回のループだけ飛ばして、次へ進む

```python
# 遅いリクエストだけを表示し、正常なものは飛ばす
latencies_ms = [95, 210, 180, 260, 130]

for latency_ms in latencies_ms:
    if latency_ms <= 200:
        continue   # 正常なリクエストをスキップ
    print(latency_ms, end=" ")

# 出力: 210 260
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
# 必須レビューが不足していないか調べる
completed_checks = ["unit-test", "lint", "api-test"]
required_check = "security-review"

for check in completed_checks:
    if check == required_check:
        print(f"{required_check} は完了しています")
        break
else:
    # ループが break で終わらなかったので、必須チェックは見つからなかった
    print(f"{required_check} が不足しています")

# 出力: security-review が不足しています
```

---

## 入れ子ループ

ループの中に、さらにループを書けます。

```python
# モジュール/チェックの表を表示する
modules = ["API", "UI", "DB"]
checks = ["lint", "test"]

for module in modules:
    for check in checks:
        print(f"{module}:{check}", end="\t")
    print()   # 各モジュールの終わりで改行
```

出力：

```
API:lint	API:test
UI:lint	UI:test
DB:lint	DB:test
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

### 練習1: リリースチェックラベル

分岐の順番を、小さなリリースチェックのラベル付けで練習します。

1 から 50 までのサンプル番号を表示してください。ただし、
- 15 で割り切れるなら "FullCheck" を表示する
- 3 で割り切れるなら "Lint" を表示する
- 5 で割り切れるなら "Test" を表示する
- それ以外は数字そのものを表示する

```python
for i in range(1, 51):
    if i % 15 == 0:
        print("FullCheck")
    elif i % 3 == 0:
        print("Lint")
    elif i % 5 == 0:
        print("Test")
    else:
        print(i)
```

ヒント: まず 15 で割り切れるかを判定し、その後で 3 と 5 を判定します。

### 練習2: レイテンシ告警ループ（サンプル数制限あり）

最大 7 個のレイテンシサンプルを確認し、しきい値を超えたらすぐ停止しましょう。

```python
latencies_ms = [120, 180, 260, 140, 310, 190, 170]
threshold_ms = 250
max_samples = 7

for sample_no, latency_ms in enumerate(latencies_ms[:max_samples], start=1):
    print(f"サンプル {sample_no}: {latency_ms} ms")

    if latency_ms <= threshold_ms:
        print("正常")
        continue

    print("告警: レイテンシがしきい値を超えました")
    break
else:
    print("確認したサンプルはすべてしきい値内でした。")
```

:::tip[デバッグで疲れないために]
まず小さな固定リストを使い、遅い値の位置を変えながら、正常パス、告警パス、すべて正常のパスを確認しましょう。
:::
### 練習3: デプロイ進捗バーを表示する

ループを使って次の進捗表示を出してください。

```
#
##
###
####
#####
```

次に、カウントダウンの進捗バーも表示してみましょう。

```
#####
####
###
##
#
```

### 練習4: 失敗したチェックを見つける

ステータスが `"passed"` ではないチェック名をすべて表示してください。

```python
checks = [
    ("lint", "passed"),
    ("unit-test", "failed"),
    ("api-test", "passed"),
    ("security-review", "failed"),
]

for check_name, status in checks:
    if status == "passed":
        continue
    print(f"{check_name}: {status}")
```

<details>
<summary>参考実装と解説</summary>

1. リリースチェックラベルでは、先に `15` で割り切れるかを判定します。そうしないと `15` が先に `Lint` や `Test` として出力される可能性があります。
2. レイテンシリストでは、固定データを使って、正常パス、告警パス、遅い値を取り除いたすべて正常のパスを確認します。
3. 進捗バーは `for n in range(1, 6): print("#" * n)` で表示できます。カウントダウンには逆向きの `range` を使います。
4. 失敗チェックの抽出では `"passed"` に対して `continue` を使い、失敗または未通過のステータスだけを表示します。
5. 境界ミスに注意します。`range(1, 51)` は `50` を含み、`range(1, 50)` は含みません。

</details>

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

:::tip[コア理解]
フロー制御はプログラミングの**骨組み**です。変数はデータ、演算子は操作、そしてフロー制御は「どんな条件で何をするか」「何回やるか」を決めます。フロー制御を身につけると、ロジックのあるプログラムが書けるようになります。
:::