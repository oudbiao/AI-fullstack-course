---
title: "2.2.1 オブジェクト指向プログラミング"
description: "Python のオブジェクト指向プログラミングの核となる概念を身につける"
sidebar:
  order: 1
---

# 2.2.1 オブジェクト指向プログラミング

![クラス、オブジェクト、属性、メソッドの関係図](/img/course/ch02-oop-class-object-map-ja.webp)

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

この節では、Python で複雑なコードを整理するための重要な方法を紹介します。オブジェクト指向は最初から完璧に理解する必要はありませんが、クラス、オブジェクト、属性、メソッドを理解しておくと、後で出てくるモデルクラス、データセットクラス、API サービスクラス、そしてサードパーティライブラリのソースコードが読みやすくなります。

## 学習目標

- オブジェクト指向プログラミング（OOP）の基本思想を理解する
- クラスとオブジェクトの定義と使い方を身につける
- 属性とメソッドを理解する
- 継承、カプセル化の基本的な使い方を身につける
- よく使うマジックメソッドを知る

---

## なぜオブジェクト指向が必要なのか？

たとえば、AI ポートフォリオアプリ用の小さなプロジェクトトラッカーを作っていて、各機能の担当者と作業記録を残したいとします。

```python
# 変数と辞書で書く方法
feature1_name = "ログイン API"
feature1_owner = "Mina"
feature1_hours = [2, 5, 1]

feature2_name = "RAG デモ"
feature2_owner = "Kai"
feature2_hours = [3, 4, 2]

# あるいは辞書を使う
feature1 = {"name": "ログイン API", "owner": "Mina", "hours": [2, 5, 1]}
feature2 = {"name": "RAG デモ", "owner": "Kai", "hours": [3, 4, 2]}

# 合計作業時間を計算する関数
def total_hours(feature):
    return sum(feature["hours"])
```

このような書き方にはいくつか問題があります。
- データと操作が**分離**されている（機能データは辞書の中、計算関数は外側）
- **制約**がない（誰でも辞書に変なキーを追加したり、必要なキーを消したりできる）
- 機能の属性が増えるほど、コードが**どんどん散らかる**

オブジェクト指向プログラミングの考え方は、**データと操作をひとまとめにして、「オブジェクト」にする**ことです。

```python
class FeatureTask:
    def __init__(self, name, owner, hours):
        self.name = name
        self.owner = owner
        self.hours = hours

    def total_hours(self):
        return sum(self.hours)

# 機能タスクのオブジェクトを作成
task1 = FeatureTask("ログイン API", "Mina", [2, 5, 1])
task2 = FeatureTask("RAG デモ", "Kai", [3, 4, 2])

# データと操作がひとつにまとまっていて、自然に使える
print(f"{task1.name} の合計時間: {task1.total_hours():.1f}")
print(f"{task2.name} の合計時間: {task2.total_hours():.1f}")
```

---

## クラスとオブジェクトの基本概念

身近な例で考えてみましょう。

- **クラス（Class）** = 設計図/テンプレート。たとえば「スマートフォン」はひとつの概念・カテゴリ
- **オブジェクト（Object/Instance）** = 設計図から作られた実体。たとえば「あなたが持っている iPhone 15」

```
クラス：FeatureTask（機能タスクのテンプレート）
    └── 属性：name, owner, hours
    └── メソッド：total_hours(), is_over_budget()

オブジェクト（インスタンス）：
    └── task1 = FeatureTask("ログイン API", "Mina", [2, 5, 1])
    └── task2 = FeatureTask("RAG デモ", "Kai", [3, 4, 2])
```

---

## クラスを定義する

### いちばん簡単なクラス

```python
class Dog:
    """1匹の犬"""

    def __init__(self, name, breed):
        """初期化メソッド。オブジェクト作成時に自動で呼ばれる"""
        self.name = name      # インスタンス属性
        self.breed = breed    # インスタンス属性

    def bark(self):
        """メソッド：犬が鳴く"""
        print(f"{self.name} が言う: ワンワンワン！")

    def info(self):
        """メソッド：情報を表示する"""
        print(f"名前: {self.name}, 品種: {self.breed}")

# オブジェクトを作成（インスタンス化）
my_dog = Dog("福ちゃん", "ゴールデンレトリバー")
your_dog = Dog("クロ", "ラブラドール")

# 属性にアクセス
print(my_dog.name)     # 福ちゃん
print(your_dog.breed)  # ラブラドール

# メソッドを呼び出す
my_dog.bark()      # 福ちゃん が言う: ワンワンワン！
your_dog.info()    # 名前: クロ, 品種: ラブラドール
```

### 重要ポイントの解説

**1. `__init__` メソッド（コンストラクタ）**

`__init__` は、オブジェクトを作成したときに**自動で呼び出される**メソッドで、オブジェクトの属性を初期化するために使います。

```python
my_dog = Dog("福ちゃん", "ゴールデンレトリバー")
# Python は内部で次のことを行う：
# 1. 新しい Dog オブジェクトを作成する
# 2. __init__(self, "福ちゃん", "ゴールデンレトリバー") を呼ぶ
# 3. self.name = "福ちゃん"
# 4. self.breed = "ゴールデンレトリバー"
# 5. このオブジェクトを my_dog に返す
```

**2. `self` とは何か？**

`self` は**そのオブジェクト自身**を表します。`my_dog.bark()` を呼ぶと、Python は自動的に `my_dog` を `self` として `bark` メソッドに渡します。

```python
my_dog.bark()
# 次と同じ意味
Dog.bark(my_dog)
```

つまり、`self.name` は「このオブジェクトの name」という意味です。

:::tip[self の名前について]
`self` はあくまで慣習（convention）です。`this` や別の名前にすることもできますが、**強くおすすめするのは `self`** です。これは Python プログラマ全員の約束ごとです。
:::
---

## 属性とメソッド

### インスタンス属性 vs クラス属性

```python
class FeatureTask:
    # クラス属性：すべてのインスタンスで共有される
    project = "AI ポートフォリオ"
    task_count = 0

    def __init__(self, name, owner):
        # インスタンス属性：各インスタンスごとに独立
        self.name = name
        self.owner = owner
        FeatureTask.task_count += 1  # タスクを1つ作るたびにカウントを1増やす

t1 = FeatureTask("ログイン API", "Mina")
t2 = FeatureTask("RAG デモ", "Kai")

# クラス属性はクラス名でもインスタンスでもアクセスできる
print(FeatureTask.project)     # AI ポートフォリオ
print(t1.project)              # AI ポートフォリオ
print(FeatureTask.task_count)  # 2

# インスタンス属性はそれぞれのインスタンスに属する
print(t1.name)   # ログイン API
print(t2.owner)  # Kai
```

### メソッド

```python
class Circle:
    def __init__(self, radius):
        self.radius = radius

    def area(self):
        """面積を計算する"""
        return 3.14159 * self.radius ** 2

    def perimeter(self):
        """周囲長を計算する"""
        return 2 * 3.14159 * self.radius

    def scale(self, factor):
        """半径を拡大・縮小する"""
        self.radius *= factor  # 属性を変更する

c = Circle(5)
print(f"面積: {c.area():.2f}")       # 78.54
print(f"周囲長: {c.perimeter():.2f}")   # 31.42

c.scale(2)  # 半径は 10 に変わる
print(f"拡大後の面積: {c.area():.2f}") # 314.16
```

---

## マジックメソッド（ダブルアンダースコアメソッド）

Python では、`__` で始まり `__` で終わるメソッドをマジックメソッド（Magic Methods）と呼びます。これを使うと、クラスを組み込み型のように扱えます。

### `__str__`：print の出力を定義する

```python
class FeatureTask:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner

    def __str__(self):
        return f"FeatureTask({self.name}, owner={self.owner})"

task = FeatureTask("ログイン API", "Mina")
print(task)  # FeatureTask(ログイン API, owner=Mina)
# __str__ がなければ、print は <__main__.FeatureTask object at 0x...> を出力する
```

### `__repr__`：開発者向けの表示を定義する

```python
class FeatureTask:
    def __init__(self, name, owner):
        self.name = name
        self.owner = owner

    def __repr__(self):
        return f"FeatureTask('{self.name}', '{self.owner}')"

task = FeatureTask("ログイン API", "Mina")
print(repr(task))   # FeatureTask('ログイン API', 'Mina')
# 対話モードで task と直接入力しても同じように表示される
```

### `__len__`：len() の動作を定義する

```python
class Playlist:
    def __init__(self, name, songs):
        self.name = name
        self.songs = songs

    def __len__(self):
        return len(self.songs)

my_playlist = Playlist("お気に入り音楽", ["曲A", "曲B", "曲C"])
print(len(my_playlist))  # 3
```

### `__eq__`：`==` の動作を定義する

```python
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

p1 = Point(3, 4)
p2 = Point(3, 4)
p3 = Point(1, 2)

print(p1 == p2)  # True
print(p1 == p3)  # False
```

---

## 継承

継承を使うと、すでにあるクラスをもとに新しいクラスを作り、**コードを再利用**できます。

### 基本的な継承

```python
# 親クラス（基底クラス）
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        print(f"{self.name} が声を出しました")

    def info(self):
        print(f"{self.name}, {self.age}歳")

# 子クラス（派生クラス）
class Dog(Animal):
    def __init__(self, name, age, breed):
        super().__init__(name, age)  # 親クラスの __init__ を呼ぶ
        self.breed = breed

    def speak(self):  # 親クラスのメソッドを上書きする
        print(f"{self.name} が言う: ワンワンワン！")

    def fetch(self):  # 子クラスだけのメソッド
        print(f"{self.name} がボールを取ってきた！")

class Cat(Animal):
    def speak(self):  # 親クラスのメソッドを上書きする
        print(f"{self.name} が言う: ニャーニャー～")

# 使用例
dog = Dog("ポチ", 3, "ゴールデンレトリバー")
cat = Cat("ミミ", 2)

dog.info()     # ポチ, 3歳（Animal から継承）
dog.speak()    # ポチ が言う: ワンワンワン！（Dog 独自の実装）
dog.fetch()    # ポチ がボールを取ってきた！（Dog だけのメソッド）

cat.info()     # ミミ, 2歳
cat.speak()    # ミミ が言う: ニャーニャー～
```

### `super()` の役割

`super()` は親クラスのメソッドを呼ぶために使います。いちばんよくある使い方は `__init__` の中です。

```python
class Animal:
    def __init__(self, name):
        self.name = name

class Dog(Animal):
    def __init__(self, name, breed):
        super().__init__(name)   # 親クラスに name の初期化を任せる
        self.breed = breed       # 自分で breed を初期化する
```

### `isinstance()` で型を確認する

```python
dog = Dog("ポチ", 3, "ゴールデンレトリバー")

print(isinstance(dog, Dog))     # True —— Dog である
print(isinstance(dog, Animal))  # True —— 継承しているので Animal でもある
print(isinstance(dog, Cat))     # False —— Cat ではない
```

---

## カプセル化

カプセル化の考え方は、**内部の詳細を隠し、必要なインターフェースだけを公開する**ことです。

### プライベート属性（慣習）

Python には本当の意味でのプライベート属性はありませんが、**命名の慣習**があります。

```python
class BankAccount:
    def __init__(self, owner, balance=0):
        self.owner = owner
        self._balance = balance  # 先頭にアンダースコア： "内部で使う" という慣習

    def deposit(self, amount):
        if amount > 0:
            self._balance += amount
            print(f"{amount} 円を入金しました。残高: {self._balance}")

    def withdraw(self, amount):
        if 0 < amount <= self._balance:
            self._balance -= amount
            print(f"{amount} 円を引き出しました。残高: {self._balance}")
        else:
            print("残高不足です！")

    def get_balance(self):
        return self._balance

account = BankAccount("ポートフォリオ口座", 1000)
account.deposit(500)     # 500 円を入金しました。残高: 1500
account.withdraw(200)    # 200 円を引き出しました。残高: 1300
print(account.get_balance())  # 1300

# 技術的には _balance に直接アクセスできますが、推奨される方法ではありません
# print(account._balance)  # 使えるが、そうすべきではない
```

| 命名の慣習 | 意味 | 例 |
|---------|------|------|
| `name` | 公開属性 | `self.name` |
| `_name` | 内部使用（慣習） | `self._balance` |
| `__name` | 名前の変換（強い隠蔽） | `self.__secret` |

---

## 総合例：AI モデルマネージャー

```python
class AIModel:
    """AI モデルの基底クラス"""
    model_count = 0

    def __init__(self, name, version="1.0"):
        self.name = name
        self.version = version
        self.is_trained = False
        self._accuracy = 0.0
        self._history = []
        AIModel.model_count += 1

    def train(self, epochs=10):
        """モデルを学習する（シミュレーション）"""
        import random
        print(f"{self.name} v{self.version} の学習を開始...")
        for epoch in range(1, epochs + 1):
            acc = min(0.5 + epoch * 0.05 + random.uniform(-0.02, 0.02), 1.0)
            self._history.append(acc)
            if epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch}/{epochs} - Accuracy: {acc:.2%}")
        self._accuracy = self._history[-1]
        self.is_trained = True
        print(f"学習完了！最終精度: {self._accuracy:.2%}")

    def predict(self, data):
        """予測"""
        if not self.is_trained:
            print("エラー：モデルがまだ学習されていません！")
            return None
        print(f"{self.name} が {len(data)} 件のデータを予測しています...")
        return [f"予測結果_{i}" for i in range(len(data))]

    def __str__(self):
        status = "学習済み" if self.is_trained else "未学習"
        return f"Model({self.name} v{self.version}, {status}, acc={self._accuracy:.2%})"


class ImageClassifier(AIModel):
    """画像分類モデル"""
    def __init__(self, name, version="1.0", num_classes=10):
        super().__init__(name, version)
        self.num_classes = num_classes

    def predict(self, images):
        if not self.is_trained:
            print("エラー：モデルがまだ学習されていません！")
            return None
        print(f"{len(images)} 枚の画像を分類しています（{self.num_classes} クラス）...")
        import random
        return [random.randint(0, self.num_classes - 1) for _ in images]


# 使用例
model = ImageClassifier("ResNet-50", "2.0", num_classes=100)
print(model)  # Model(ResNet-50 v2.0, 未学習, acc=0.00%)

model.train(epochs=10)
predictions = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])
print(f"予測クラス: {predictions}")
print(model)
print(f"現在のモデル総数: {AIModel.model_count}")
```

---

## ハンズオン練習

### 練習 1：書籍管理

`Book` クラスを作成してください。

```python
class Book:
    def __init__(self, title, author, pages):
        self.title = title
        self.author = author
        self.pages = pages
        self.current_page = 0

    def read(self, pages):
        self.current_page = min(self.current_page + pages, self.pages)

    def progress(self):
        return self.current_page / self.pages * 100

    def __str__(self):
        return f"{self.title} / {self.author}: {self.current_page}/{self.pages} ページ"

# テスト
book = Book("Python入門", "コースチーム", 300)
book.read(50)
print(f"{book.progress():.1f}%")  # 16.7%
print(book)
```

### 練習 2：簡単なショッピングカート

```python
class Product:
    def __init__(self, name, price):
        self.name = name
        self.price = price

class ShoppingCart:
    def __init__(self):
        self.items = {}

    def add(self, product, quantity=1):
        self.items[product.name] = self.items.get(product.name, [product, 0])
        self.items[product.name][1] += quantity

    def remove(self, product_name):
        self.items.pop(product_name, None)

    def total(self):
        return sum(product.price * quantity for product, quantity in self.items.values())

    def __str__(self):
        lines = [f"{product.name} x {quantity}" for product, quantity in self.items.values()]
        return "\n".join(lines) or "カートは空です"

cart = ShoppingCart()
cart.add(Product("キーボード", 199), 2)
cart.add(Product("マウス", 99), 1)
print(cart)
print(f"合計: {cart.total()}")
```

### 練習 3：動物園

継承を使って実装してください。

```python
class Animal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    def speak(self):
        return "..."

class Dog(Animal):
    def speak(self):
        return "ワンワン"

class Cat(Animal):
    def speak(self):
        return "ニャーニャー"

class Duck(Animal):
    def speak(self):
        return "ガーガー"

animals = [Dog("クロ", 3), Cat("ミミ", 2), Duck("ダック", 1)]
for animal in animals:
    print(f"{animal.name}: {animal.speak()}")
```

<details>
<summary>参考実装と解説</summary>

1. `Book` は現在ページを内部状態として保持し、`progress()` で読書進捗を計算できれば合格です。300 ページ中 50 ページ読んだ直後は約 `16.7%`、文字列表現には `50/300` が出るはずです。
2. `ShoppingCart` は商品オブジェクトと数量を一緒に保存し、`total()` で価格と数量を掛け合わせます。存在しない商品を `remove()` しても壊れず、空のときは明確なメッセージを返せると実用的です。
3. `Animal` は共通フィールドと仮の `speak()` を持ち、各サブクラスは鳴き声だけを上書きします。ループで名前と鳴き声がそれぞれ変われば、継承とポリモーフィズムが働いています。

</details>

---

## まとめ

| 概念 | 説明 | 構文 |
|------|------|------|
| **クラス** | オブジェクトのテンプレート/設計図 | `class MyClass:` |
| **オブジェクト** | クラスのインスタンス | `obj = MyClass()` |
| **`__init__`** | コンストラクタ。属性を初期化する | `def __init__(self):` |
| **self** | 現在のオブジェクト自身を指す | `self.name = name` |
| **継承** | 子クラスが親クラスのコードを再利用する | `class Dog(Animal):` |
| **super()** | 親クラスのメソッドを呼ぶ | `super().__init__()` |
| **マジックメソッド** | オブジェクトのふるまいをカスタマイズする | `__str__`, `__len__`, `__eq__` |

:::tip[核心の理解]
オブジェクト指向の核心は、**データと振る舞いをひとまとめにする**ことです。クラスはテンプレート、オブジェクトは実体です。継承でコードを再利用し、カプセル化で詳細を隠します。AI 開発ではクラスをよく使います。たとえば PyTorch のモデル定義は `nn.Module` を継承したクラスですし、学習ループで扱うのもオブジェクトです。
:::