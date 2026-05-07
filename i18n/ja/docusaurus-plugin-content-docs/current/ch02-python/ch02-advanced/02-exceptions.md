---
title: "2.2.2 例外処理"
sidebar_position: 2
description: "Python の例外処理メカニズムを身につけて、プログラムをより堅牢にしよう"
---

# 2.2.2 例外処理

![例外処理の実行フロー図](/img/course/ch02-exception-flow-ja.png)

## この節の位置づけ

この節では、エラーが起きてもプログラムがすぐにクラッシュしないようにします。例外処理は、ファイルの読み書き、ネットワークリクエスト、API 呼び出し、データクリーニング、モデル推論で何度も登場します。ここでは、エラーを事前に予測し、捕捉し、復旧できる形で対処することを学びます。

## 学習目標

- 例外とは何か、なぜ例外処理が必要なのかを理解する
- `try/except/else/finally` の使い方を身につける
- 異なる種類の例外を捕捉できるようになる
- すぐにクラッシュしない、堅牢なプログラムを書けるようになる

---

## 例外とは？

例外とは、プログラム実行中に起きる**エラー**のことです。例外処理がないプログラムは、エラーが起きるとそのままクラッシュします。

```python
# これらのコードはすべてプログラムをクラッシュさせます
print(10 / 0)           # ZeroDivisionError: ゼロ除算
print(int("abc"))        # ValueError: 変換できない
print([1, 2, 3][10])     # IndexError: インデックスが範囲外
print({"a": 1}["b"])     # KeyError: キーが存在しない

# プログラムがクラッシュすると、この後のコードは実行されません
print("この行は絶対に実行されません")
```

実際のプログラムでは、エラーは**避けられません**。ユーザーが不正なデータを入力することもありますし、ファイルが存在しないこともあります。ネットワークが切れることもあります。例外処理を使うと、こうした問題に**丁寧に対応**でき、プログラムを直接クラッシュさせずに済みます。

---

## よくある例外の種類

| 例外の種類 | 発生する場面 | 例 |
|---------|---------|------|
| `ZeroDivisionError` | ゼロ除算 | `1 / 0` |
| `TypeError` | 型が合わない操作 | `"hello" + 5` |
| `ValueError` | 値が不正 | `int("abc")` |
| `IndexError` | リストのインデックスが範囲外 | `[1, 2][5]` |
| `KeyError` | 辞書にキーが存在しない | `{"a": 1}["b"]` |
| `FileNotFoundError` | ファイルが存在しない | `open("存在しない.txt")` |
| `AttributeError` | 属性が存在しない | `"hello".foo()` |
| `NameError` | 変数が定義されていない | `print(xyz)` |
| `ImportError` | import に失敗する | `import 存在しないモジュール` |

---

## try / except の基本

`try/except` の流れは、**まずコードを試し、エラーが起きたら代わりの処理を実行する**、というものです。

```python
try:
    number = int(input("数字を入力してください: "))
    print(f"入力された数字: {number}")
except ValueError:
    print("入力が無効です！数字を入力してください。")

print("プログラムは続行します...")  # 例外があってもなくても、この行は実行されます
```

実行例：

```
# 正常入力
数字を入力してください: 42
入力された数字: 42
プログラムは続行します...

# 数字以外を入力
数字を入力してください: abc
入力が無効です！数字を入力してください。
プログラムは続行します...
```

ポイントは、**`try/except` があれば、エラーが起きてもプログラムはクラッシュしない**ことです。

---

## 異なる種類の例外を捕捉する

### 複数の例外をそれぞれ捕捉する

```python
def safe_divide(a, b):
    try:
        result = a / b
        return result
    except ZeroDivisionError:
        print("エラー：ゼロで割ることはできません！")
        return None
    except TypeError:
        print("エラー：数値を渡してください！")
        return None

print(safe_divide(10, 3))    # 3.333...
print(safe_divide(10, 0))    # エラー：ゼロで割ることはできません！ → None
print(safe_divide("10", 3))  # エラー：数値を渡してください！ → None
```

### 複数の例外をまとめて捕捉する

```python
try:
    # エラーが起きる可能性があるコード
    value = int(input("数字を入力してください: "))
    result = 100 / value
    print(f"結果: {result}")
except (ValueError, ZeroDivisionError) as e:
    print(f"エラーが発生しました: {e}")
```

### 例外情報を取得する

```python
try:
    number = int("abc")
except ValueError as e:
    print(f"例外の種類: {type(e).__name__}")  # ValueError
    print(f"例外メッセージ: {e}")                 # invalid literal for int() with base 10: 'abc'
```

### すべての例外を捕捉する（注意して使う）

```python
try:
    # いくつかのコード
    result = risky_operation()
except Exception as e:
    print(f"予期しないエラーが発生しました: {type(e).__name__}: {e}")
```

:::caution `except Exception` を乱用しない
すべての例外を捕捉すると便利そうに見えますが、**本当のバグを隠してしまう**ことがあります。できるだけ**具体的な例外の種類**を捕捉し、`except Exception` は最終手段として外側で使いましょう。

```python
# よくない例 ❌
try:
    do_something()
except:  # KeyboardInterrupt まで含めてすべて捕捉してしまう
    pass   # しかも何もしない！

# よい例 ✅
try:
    do_something()
except ValueError:
    handle_value_error()
except FileNotFoundError:
    handle_file_not_found()
except Exception as e:
    logging.error(f"予期しないエラー: {e}")
```
:::

---

## try / except / else / finally

完全な例外処理の構造は、次の4つの部分からなります。

```python
try:
    # 試して実行するコード
    file = open("data.txt", "r")
    content = file.read()
except FileNotFoundError:
    # エラーが起きたときに実行
    print("ファイルが存在しません！")
else:
    # エラーがなかったときに実行
    print(f"ファイルの内容: {content}")
finally:
    # エラーの有無にかかわらず実行（通常はリソースの片付けに使う）
    print("処理が完了しました")
```

| 節 | 実行されるタイミング | 用途 |
|------|---------|------|
| `try` | いつも実行 | エラーが起きるかもしれないコードを置く |
| `except` | エラーが起きたときだけ実行 | エラーを処理する |
| `else` | エラーが起きなかったときだけ実行 | 成功後の処理を置く |
| `finally` | エラーの有無にかかわらず実行 | リソースの片付け（ファイルを閉じる、接続を切る） |

### finally の典型的な使い方

```python
file = None
try:
    file = open("data.txt", "r")
    data = file.read()
    # データを処理...
except FileNotFoundError:
    print("ファイルが存在しません")
finally:
    if file:
        file.close()   # エラーの有無にかかわらず、ファイルは閉じる
        print("ファイルを閉じました")
```

:::tip よりよい方法：with 文
後の「ファイル操作」の章で `with` 文を学びます。`with` 文はリソースの解放を自動で行ってくれるので、`finally` よりも簡潔です。
:::

---

## 例外を投げる

例外を処理するだけでなく、**自分で例外を投げる**こともできます。これは、ありえない状態や不正な状態を見つけたときに、呼び出し元へ「問題がある」と伝えるためです。

### `raise` 文

```python
def set_age(age):
    if not isinstance(age, int):
        raise TypeError("年齢は整数でなければなりません")
    if age < 0 or age > 150:
        raise ValueError(f"年齢 {age} は不適切です。0〜150 の範囲である必要があります")
    return age

# 正常な使用
print(set_age(25))      # 25

# 例外を発生させる
try:
    set_age(-5)
except ValueError as e:
    print(f"エラー: {e}")  # エラー: 年齢 -5 は不適切です。0〜150 の範囲である必要があります

try:
    set_age("二十")
except TypeError as e:
    print(f"エラー: {e}")  # エラー: 年齢は整数でなければなりません
```

### 独自の例外を作る

組み込みの例外では足りない場合は、自分で定義できます。

```python
class InsufficientFundsError(Exception):
    """残高不足の例外"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"残高不足：現在の残高は {balance}、引き出そうとした金額は {amount} です")

class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance

    def withdraw(self, amount):
        if amount > self.balance:
            raise InsufficientFundsError(self.balance, amount)
        self.balance -= amount
        return self.balance

# 使用例
account = BankAccount(1000)
try:
    account.withdraw(1500)
except InsufficientFundsError as e:
    print(f"取引失敗: {e}")
    print(f"現在の残高: {e.balance}, 依頼金額: {e.amount}")
```

---

## 実践パターン

### パターン 1：LBYL と EAFP

Python コミュニティでは、**EAFP**（Easier to Ask Forgiveness than Permission、先にやってから考える）が、**LBYL**（Look Before You Leap、先に確認してから実行する）より好まれます。

```python
# LBYL スタイル（先に確認してから操作）—— Python らしくない
if key in my_dict:
    value = my_dict[key]
else:
    value = default_value

# EAFP スタイル（先に操作し、エラーが起きたら処理）—— より Python らしい
try:
    value = my_dict[key]
except KeyError:
    value = default_value

# もちろん、辞書にはもっとよい書き方もあります
value = my_dict.get(key, default_value)
```

### パターン 2：再試行メカニズム

```python
import time

def fetch_data_with_retry(url, max_retries=3):
    """再試行付きでデータを取得する"""
    for attempt in range(1, max_retries + 1):
        try:
            print(f"{attempt} 回目の試行...")
            # ネットワークリクエストを模擬
            import random
            if random.random() < 0.5:
                raise ConnectionError("ネットワーク接続に失敗しました")
            return "取得したデータ"
        except ConnectionError as e:
            print(f"  失敗: {e}")
            if attempt < max_retries:
                wait = attempt * 2  # 待ち時間を徐々に長くする
                print(f"  {wait} 秒後に再試行します...")
                time.sleep(wait)
            else:
                print("  すべての再試行に失敗しました！")
                raise  # 最後の再試行も失敗したら例外を投げる

try:
    data = fetch_data_with_retry("https://api.example.com")
    print(f"成功: {data}")
except ConnectionError:
    print("最終的にデータ取得に失敗しました")
```

### パターン 3：安全なユーザー入力

```python
def get_number(prompt, min_val=None, max_val=None):
    """ユーザー入力の数字を安全に取得する"""
    while True:
        try:
            value = float(input(prompt))
            if min_val is not None and value < min_val:
                print(f"{min_val} 以上の数を入力してください")
                continue
            if max_val is not None and value > max_val:
                print(f"{max_val} 以下の数を入力してください")
                continue
            return value
        except ValueError:
            print("有効な数字を入力してください！")

# 使用例
age = get_number("年齢を入力してください: ", min_val=0, max_val=150)
print(f"あなたの年齢は: {age}")
```

---

## 総合例：安全な成績管理システム

```python
class GradeManager:
    def __init__(self):
        self.students = {}

    def add_student(self, name, score):
        """学生の成績を追加する"""
        if not isinstance(name, str) or not name.strip():
            raise ValueError("学生名は空にできません")
        if not isinstance(score, (int, float)):
            raise TypeError(f"成績は数値である必要があります。受け取った型: {type(score).__name__}")
        if not 0 <= score <= 100:
            raise ValueError(f"成績 {score} は範囲外です（0〜100）")

        self.students[name] = score
        print(f"✅ 追加成功: {name} - {score} 点")

    def get_average(self):
        """平均点を取得する"""
        if not self.students:
            raise RuntimeError("学生データがないため、平均点を計算できません")
        return sum(self.students.values()) / len(self.students)

    def get_student(self, name):
        """学生の成績を検索する"""
        if name not in self.students:
            raise KeyError(f"学生が見つかりません: {name}")
        return self.students[name]

# 使用
gm = GradeManager()

# 学生を安全に追加する
test_data = [
    ("張三", 85),
    ("李四", 92),
    ("王五", "優秀"),  # 型エラー
    ("趙六", 150),     # 範囲エラー
    ("", 80),          # 名前が空
    ("銭七", 78),
]

for name, score in test_data:
    try:
        gm.add_student(name, score)
    except (ValueError, TypeError) as e:
        print(f"❌ 追加失敗: {e}")

# 検索
print(f"\n平均点: {gm.get_average():.1f}")

try:
    print(gm.get_student("孫八"))
except KeyError as e:
    print(f"検索失敗: {e}")
```

---

## 手を動かしてみよう

### 練習 1：安全な計算機

```python
def safe_calculator(inputs=None):
    """不正入力とゼロ除算を処理できる、安全な四則演算機。"""
    inputs = iter(inputs or ["10", "0", "/", "n"])

    while True:
        try:
            a = float(next(inputs) if inputs else input("1つ目の数値: "))
            b = float(next(inputs) if inputs else input("2つ目の数値: "))
            op = next(inputs) if inputs else input("演算子（+、-、*、/）: ")

            if op == "+":
                result = a + b
            elif op == "-":
                result = a - b
            elif op == "*":
                result = a * b
            elif op == "/":
                result = a / b
            else:
                raise ValueError(f"未対応の演算子です: {op}")

            print(f"結果: {result}")
        except ZeroDivisionError:
            print("ゼロで割ることはできません。")
        except ValueError as error:
            print(f"入力が不正です: {error}")
        except StopIteration:
            break

        again = next(inputs, "n") if inputs else input("続けますか？(y/n): ")
        if again.lower() != "y":
            break

safe_calculator()
```

### 練習 2：ファイル読み取り器

```python
def read_file_safely(filename):
    """ファイルの内容を安全に読み取る。"""
    try:
        with open(filename, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        print(f"ファイルが見つかりません: {filename}")
    except PermissionError:
        print(f"読み取り権限がありません: {filename}")
    except OSError as error:
        print(f"読み取りに失敗しました: {error}")
    return None

content = read_file_safely("test.txt")
if content:
    print(content)
```

### 練習 3：一括型変換

```python
def convert_to_numbers(data_list):
    """文字列を数値に変換し、失敗理由も残す。"""
    numbers = []
    errors = []
    for item in data_list:
        try:
            numbers.append(float(item))
        except ValueError:
            numbers.append(None)
            errors.append(f"{item} を変換できません")
    return numbers, errors

values, errors = convert_to_numbers(["10", "20.5", "abc", "30", "xyz"])
print(values)
print(errors)
```

---

## まとめ

| 文法 | 役割 | 使う場面 |
|------|------|---------|
| `try` | エラーが起きるかもしれないコードを囲む | エラーの可能性があるところ全般 |
| `except` | 例外を捕捉して処理する | 対象の例外を指定して処理するとき |
| `else` | 例外がなかったときに実行する | 成功後の処理 |
| `finally` | 必ず実行する | リソースの片付け |
| `raise` | 自分で例外を投げる | 入力が不正、状態が不正なとき |
| 独自例外 | 業務に合った例外を作る | 組み込み例外では説明しきれないとき |

:::tip コアの理解
例外処理の本質は、**起こりうる問題を予測し、対処法を用意しておくこと**です。よいプログラムとは、エラーが起きないプログラムではなく、エラーが起きたときに**丁寧に処理できる**プログラムです。ユーザーにわかりやすいメッセージを出し、エラー情報を記録し、必要なら自動で再試行する。これが、初心者とプロの大きな違いです。
:::
