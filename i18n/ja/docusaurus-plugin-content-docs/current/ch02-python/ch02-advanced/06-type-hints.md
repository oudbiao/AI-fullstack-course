---
title: "2.2.6 型注釈とコード品質"
sidebar_position: 6
description: "Python の型注釈とコード品質ツールを身につける"
---

# 2.2.6 型注釈とコード品質

![型注釈とコード品質のフローチャート](/img/course/ch02-type-hints-quality-flow-ja.webp)

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
パターン：class、exception、file IO、functional pipeline、generator、またはtype hint
コード成果物：最小限の実行可能な例と、現実的なユースケース 1 つ
出力：印字されたオブジェクト状態、捕捉したエラー、保存したファイル、yieldされた値、または型チェックのメモ
失敗確認：隠れた変更、副作用を飲み込む例外、ファイルパスの問題、lazy iterator の混同、または誤解を招く注釈
期待される成果: デバッグメモを含む小さな高度Python例
```

## この節の位置づけ

この節では、「コードが動く」から一歩進んで、「コードを保守しやすくする」ことに注目します。型注釈、フォーマッタ、コードチェックツールを使うと、プロジェクトが大きくなったときや、複数人で協力するとき、複雑なライブラリを使うときにミスを減らせます。また、未来の自分もコードをすばやく理解しやすくなります。

## 学習目標

- なぜ型注釈が必要なのかを理解する
- Python の型注釈の基本構文を身につける
- よく使うコード品質ツール（linter、formatter）を知る
- 高品質なコードを書く習慣をつける

---

## なぜ型注釈が必要なのか？

Python は動的型付け言語です。変数を宣言するときに型を書く必要はありません。これはとても柔軟ですが、一方で次のような問題もあります。

```python
def calculate_total(items, tax):
    return sum(items) * (1 + tax)

# 使うときには、推測する必要がある:
# items は何？ リスト？ タプル？
# tax は何？ 0.1？ それとも "10%"？
# 戻り値は何？ 数字？ 文字列？
```

プロジェクトが大きくなると、型情報のないコードは**道標のない高速道路**のようなものです。頼れるのは勘だけになってしまいます。

型注釈の役割は次のとおりです。

| メリット | 説明 |
|------|------|
| **自己文書化** | 関数が何を受け取り、何を返すかが一目でわかる |
| **IDE のスマート補完** | VS Code でより正確な自動補完が使える |
| **静的チェック** | 実行前に型の間違いを見つけられる |
| **チーム開発** | コミュニケーションコストを下げ、コード自体が意図を伝えてくれる |

---

## 基本的な型注釈

### 変数注釈

```python
# 基本型
name: str = "小明"
age: int = 25
height: float = 1.75
is_student: bool = True

# Python は型注釈を強制しない
# 次のコードはエラーにならないが、静的チェックツールは警告する
age: int = "二十五"  # 型注釈では int だが、実際には str を代入している
```

:::info 型注釈は「提案」
Python の型注釈は**実行時に強制されません**。型が一致しなくても、プログラムは動きます。これは主に**開発者とツール**のための情報です。実際の型チェックには mypy などの静的解析ツールを使います。
:::

### 関数注釈

```python
def greet(name: str) -> str:
    """
    name: str  → 引数 name の型は str
    -> str     → 戻り値の型は str
    """
    return f"こんにちは、{name}！"

def calculate_bmi(weight: float, height: float) -> float:
    """BMI を計算する"""
    return weight / (height ** 2)

def train_model(epochs: int = 10, lr: float = 0.001) -> None:
    """None を返す関数"""
    print(f"{epochs} エポックを学習します。学習率は {lr} です")
```

型注釈があると、VS Code のスマート補完がとても正確になります。`greet(` と入力したときに、引数の型が `str` だとすぐにわかります。

---

## 複合型の型注釈

### リストと辞書

```python
# Python 3.9+：組み込み型をそのまま使える
scores: list[int] = [85, 92, 78]
student: dict[str, int] = {"张三": 85, "李四": 92}
coordinates: tuple[float, float] = (3.14, 2.71)
unique_ids: set[int] = {1, 2, 3}

# Python 3.8 以前：typing から import する必要がある
from typing import List, Dict, Tuple, Set

scores: List[int] = [85, 92, 78]
student: Dict[str, int] = {"张三": 85, "李四": 92}
```

### Optional: None になる可能性がある値

```python
from typing import Optional

def find_student(name: str) -> Optional[dict]:
    """学生を探す。見つからなければ None を返す"""
    students = {"张三": {"age": 20}, "李四": {"age": 21}}
    return students.get(name)

# Python 3.10+ では、より簡潔に書ける
def find_student(name: str) -> dict | None:
    students = {"张三": {"age": 20}, "李四": {"age": 21}}
    return students.get(name)
```

### Union: 複数の型を受け付ける

```python
from typing import Union

def process(data: Union[str, list]) -> str:
    """文字列またはリストを受け取る"""
    if isinstance(data, list):
        return ", ".join(str(item) for item in data)
    return data

# Python 3.10+ の簡潔な書き方
def process(data: str | list) -> str:
    if isinstance(data, list):
        return ", ".join(str(item) for item in data)
    return data
```

### Callable: 関数の型

```python
from typing import Callable

def apply_func(func: Callable[[int, int], int], a: int, b: int) -> int:
    """関数を引数として受け取る"""
    return func(a, b)

result = apply_func(lambda x, y: x + y, 3, 5)  # 8
```

### さらにいろいろな型注釈

```python
from typing import Any, Literal

# Any：任意の型
def log(message: Any) -> None:
    print(message)

# Literal：特定の値だけを受け取る
def set_mode(mode: Literal["train", "eval", "test"]) -> None:
    print(f"モード: {mode}")

set_mode("train")   # ✅
set_mode("play")    # 静的チェックで警告される
```

---

## 型注釈の実践

### 関数に型注釈を追加する

```python
def analyze_scores(
    scores: list[float],
    subject: str = "不明",
    pass_line: float = 60.0
) -> dict[str, float | int | str]:
    """成績を分析し、統計情報を返す"""
    if not scores:
        return {"error": "成績リストが空です"}

    return {
        "subject": subject,
        "count": len(scores),
        "average": sum(scores) / len(scores),
        "max": max(scores),
        "min": min(scores),
        "pass_count": sum(1 for s in scores if s >= pass_line)
    }
```

### クラスに型注釈を追加する

```python
class DataProcessor:
    def __init__(self, name: str, data: list[dict[str, Any]]) -> None:
        self.name: str = name
        self.data: list[dict[str, Any]] = data
        self._processed: bool = False

    def filter_by(self, key: str, value: Any) -> list[dict[str, Any]]:
        """条件に応じてデータをフィルタする"""
        return [item for item in self.data if item.get(key) == value]

    def get_column(self, key: str) -> list[Any]:
        """ある列を取り出す"""
        return [item[key] for item in self.data if key in item]
```

---

## コード品質ツール

良いコードは、動くだけではなく、**読みやすく、ルールがそろっていて、バグが少ない**ことも大切です。ここでは、そのためのツールを紹介します。

### コード整形: black

`black` は Python で最もよく使われるコードフォーマッタです。コードを自動で統一されたスタイルに整えてくれます。

```bash
# インストール
pip install black

# 1ファイルを整形
black my_script.py

# ディレクトリ全体を整形
black src/

# 変更せずにチェックだけする
black --check my_script.py
```

整形前：

```python
x = {  'a':37,'b':42,
'c':927}
y = 'hello ''world'
z = 'hello '+'world'
a = [1,2,3,4,5,]
```

整形後：

```python
x = {"a": 37, "b": 42, "c": 927}
y = "hello " "world"
z = "hello " + "world"
a = [1, 2, 3, 4, 5]
```

### コードチェック: ruff

`ruff` は新世代の Python linter で、とても高速です。よくある問題をたくさん見つけてくれます。

```bash
# インストール
pip install ruff

# コードをチェック
ruff check my_script.py

# 自動修正
ruff check --fix my_script.py

# 整形（ruff は black の代わりにも使える）
ruff format my_script.py
```

### 型チェック: mypy

```bash
# インストール
pip install mypy

# 型をチェック
mypy my_script.py
```

```python
# example.py
def add(a: int, b: int) -> int:
    return a + b

result = add("hello", "world")  # mypy がエラーを出す: 引数の型が違う！
```

```bash
$ mypy example.py
example.py:4: error: Argument 1 to "add" has incompatible type "str"; expected "int"
```

### VS Code との連携

VS Code で次の拡張機能を入れると、コード品質の問題を**リアルタイム**で確認できます。

| 拡張機能 | 機能 |
|------|------|
| **Pylance** | 型チェックとスマート補完（VS Code の標準おすすめ） |
| **Ruff** | リアルタイムのコードチェック、必要ならフォーマットも担当 |
| **Black Formatter** | チームが Black を標準フォーマッタにしている場合の保存時フォーマット |

新しいプロジェクトでは、Ruff に lint と format の両方を任せるとツールチェーンがシンプルになります。VS Code の設定に以下を追加します。

```json
{
    "editor.formatOnSave": true,
    "editor.defaultFormatter": "charliermarsh.ruff",
    "[python]": {
        "editor.defaultFormatter": "charliermarsh.ruff"
    }
}
```

チームですでに Black を標準にしている場合は、Black Formatter をデフォルトフォーマッタにし、Ruff は lint と import 整理だけに使いましょう。同じ Python ファイルに対して Ruff と Black の両方をデフォルトフォーマッタにしないでください。

---

## Python コーディング規約（PEP 8）

PEP 8 は Python の公式コーディング規約です。特に大切なポイントは次のとおりです。

### 命名規則

```python
# 変数と関数: 小文字 + アンダースコア（snake_case）
student_name = "张三"
def calculate_average(scores):
    return sum(scores) / len(scores)

# クラス: 先頭を大文字（PascalCase）
class DataProcessor:
    def __init__(self, source: str):
        self.source = source

# 定数: すべて大文字 + アンダースコア
MAX_RETRY = 3
DEFAULT_TIMEOUT = 30
API_BASE_URL = "https://api.example.com"

# "非公開" 属性: 先頭にアンダースコア
class MyClass:
    def __init__(self):
        self._internal_state = None
```

### 空行とスペース

```python
# 関数の間は空行 2 行
def function_one():
    return "function one"


def function_two():
    return "function two"


# クラスの間は空行 2 行
class ClassOne:
    value = 1


class ClassTwo:
    value = 2

# 演算子の前後にスペースを入れる
x = 1 + 2       # ✅
x = 1+2         # ❌

# カンマの後ろにスペースを入れる
items = [1, 2, 3]     # ✅
items = [1,2,3]       # ❌

# 関数引数のデフォルト値の前後にはスペースを入れない。
# 2 つ目は構文としては動きますが、PEP 8 では推奨されません。
def func(x=10):       # ✅
    return x

def func_not_recommended(x = 10):  # ❌ スタイル上は非推奨
    return x
```

### 1 行の長さ

```python
# 1 行は 79 文字以内（チームのルールによっては 88 や 120 文字）

# 長い行は括弧で改行する
result = (
    first_variable
    + second_variable
    + third_variable
)

# 引数が多い関数
def complex_function(
    param1: str,
    param2: int,
    param3: float = 0.0,
    param4: bool = True,
) -> dict:
    return {
        "param1": param1,
        "param2": param2,
        "param3": param3,
        "param4": param4,
    }
```

---

## docstring を書く

よい docstring があると、ほかの人も未来の自分もコードをすばやく理解できます。

```python
def train_model(
    data: list[dict],
    epochs: int = 100,
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> dict[str, float]:
    """
    モデルを学習し、学習指標を返す。

    Args:
        data: 学習データのリスト。各要素は1つのサンプル辞書
        epochs: 学習回数。デフォルトは 100
        learning_rate: 学習率。デフォルトは 0.001
        batch_size: バッチサイズ。デフォルトは 32

    Returns:
        学習指標を含む辞書。例:
        {"accuracy": 0.95, "loss": 0.05}

    Raises:
        ValueError: data が空のとき
        RuntimeError: GPU が使えないとき

    Example:
        >>> result = train_model(data, epochs=50)
        >>> print(result["accuracy"])
        0.95
    """
    if not data:
        raise ValueError("学習データは空にできません")
    # 実際の学習処理の代わりに、最小限の評価指標を返す
    total = sum(len(str(item)) for item in data)
    accuracy = min(0.95, 0.6 + total / 1000)
    return {"accuracy": accuracy, "loss": 1 - accuracy}
```

---

## 手を動かしてみよう

### 練習 1: 古いコードに型注釈を追加する

次のコードに、完全な型注釈を追加してください。

```python
def process_students(students, min_score):
    results = []
    for student in students:
        if student["score"] >= min_score:
            results.append({
                "name": student["name"],
                "score": student["score"],
                "passed": True
            })
    return results

def calculate_stats(numbers):
    if not numbers:
        return None
    return {
        "mean": sum(numbers) / len(numbers),
        "max": max(numbers),
        "min": min(numbers),
        "count": len(numbers)
    }
```

### 練習 2: コード品質ツールをインストールして使う

```bash
# 1. ruff をインストールする
pip install ruff

# 2. 形式に問題がある Python ファイルを作る

# 3. ruff check を実行して問題を確認する

# 4. ruff format を実行して自動整形する

# 5. 整形前後の差分を比べる
```

### 練習 3: 高品質なコードを書く

学んだルールをすべて使って、次の「よくない」コードを書き直してください。

```python
# よくないコード
def f(l,n):
 r=[]
 for x in l:
  if x>n:r.append(x)
 return r

def g(d):
 s=0
 for k in d:s+=d[k]
 return s/len(d)
```

条件:
1. わかりやすい名前にする
2. 型注釈を追加する
3. docstring を追加する
4. PEP 8 に従う

---

<details>
<summary>参考実装と解説</summary>

1. 既存関数には、たとえば `process_students(students: list[dict[str, int]], min_score: int) -> list[dict[str, object]]` や `calculate_stats(numbers: Sequence[float]) -> dict[str, float] | None` のように、入力構造と空リスト時の戻り値がわかる型を付けます。
2. `ruff` はまず `ruff check` で問題を確認し、そのあと `ruff format` で整形し、最後に差分を見る流れが扱いやすいです。lint と整形を分けると、何が直ったかを説明しやすくなります。
3. 書き直し後のコードは、意味のある関数名、型注釈、docstring、PEP 8 の空白を満たす必要があります。平均値を返す関数では、空入力でゼロ除算しないようにすることも大切です。

</details>

## まとめ

| ツール/概念 | 役割 | おすすめ度 |
|-----------|------|---------|
| **型注釈** | 引数と戻り値の型を示す | 強く推奨 |
| **PEP 8** | Python のコード規約 | 必ず守る |
| **black / ruff format** | コードを自動整形する | 強く推奨 |
| **ruff** | コード品質をチェックする | 強く推奨 |
| **mypy** | 静的型チェックを行う | 推奨 |
| **docstring** | ドキュメント文字列 | 公開関数には必須 |

:::tip 核心の理解
コードは人が読むために書き、ついでに機械が実行します。型注釈やコード規約は、コードを速く動かすためのものではありませんが、コードを**理解しやすく、保守しやすく、協力しやすく**してくれます。AI プロジェクトでは、1 人が書いたコードを複数人で使ったり変更したりすることがよくあります。だからこそ、今からよい習慣を身につけましょう。
:::
