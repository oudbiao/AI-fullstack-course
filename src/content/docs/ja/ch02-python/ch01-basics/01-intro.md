---
title: "2.1.1 Python の概要"
description: "Python 言語の特徴、活用分野、開発環境を理解する"
sidebar:
  order: 1
---

# 2.1.1 Python の概要

![Python から AI アプリケーションへのワークフロー](/img/course/ch02-python-ai-workflow-ja.webp)

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

この節は、Python 学習の入り口です。いきなり複雑な文法を覚える必要はありません。まずは、Python がなぜ AI に向いているのか、何ができるのかを理解し、最初のプログラムを実際に動かして、「コードで現実の問題を解決できる」という最初の実感をつかみましょう。

## 学習目標

- Python とは何か、なぜこれほど人気なのかを理解する
- AI 分野での Python の中心的な役割を理解する
- あなたの最初の Python プログラムを書いて実行する
- Python コードの基本構造を理解する

---

## なぜ Python を学ぶのか？

プログラミング言語が道具だとしたら、Python は**スイスアーミーナイフ**のようなものです。何でもできて、しかもすぐ使い始められます。

まずは、いくつかのデータを見てみましょう。

| 観点 | 説明 |
|------|------|
| **人気度** | 長年にわたり TIOBE のプログラミング言語ランキングで 1 位を維持 |
| **AI の第一選択** | ほぼすべての AI / 機械学習フレームワーク（PyTorch、TensorFlow）が Python を中心に使っている |
| **就職市場** | データサイエンス、AI エンジニア、バックエンド開発の必須スキル |
| **学習のしやすさ** | 文法が自然言語に近く、初心者でも始めやすい言語のひとつ |

一言でまとめると、**AI をやりたいなら、Python が唯一の出発点です。**

---

## Python とは何か？

Python は、Guido van Rossum（グイド・ヴァンロッサム）によって 1991 年に公開された**高水準プログラミング言語**です。

「高水準」とはどういう意味でしょうか？
プログラミング言語がハードウェアから離れていて、人間の言葉に近いほど「高水準」です。比べてみましょう。

```
# 機械語（2進数、コンピュータが直接実行する）
10110000 01100001

# C 言語（多くの細かい部分を手動で管理する必要がある）
#include <stdio.h>
int main() {
    printf("Hello World\n");
    return 0;
}

# Python（シンプルでわかりやすい）
print("Hello World")
```

同じ「1 文を表示する」処理でも、Python は **1 行**で済みますが、C 言語では 5 行必要です。これが Python の設計思想です。**シンプルで洗練されており、文法の細かさではなく問題解決に集中できる**ようになっています。

### Python の主な特徴

| 特徴 | 説明 | あなたにとっての利点 |
|------|------|------------------|
| **文法がシンプル** | 波かっこではなくインデントを使い、英語に近い書き方 | 早く学べて、書く量も少ない |
| **インタプリタ言語** | 書いたらすぐ実行でき、コンパイルが不要 | デバッグしやすく、すぐ結果が見える |
| **動的型付け** | 変数の型宣言が不要 | コードが短く、柔軟 |
| **エコシステムが豊富** | 40 万以上のサードパーティライブラリがある | 誰かが作った便利なものをすぐ使える |
| **マルチプラットフォーム** | Windows、macOS、Linux で動く | 1 つのコードをいろいろな環境で実行できる |

---

## Python で何ができるのか？

Python の活用範囲はとても広いです。ここでは、特に重要な分野をいくつか見てみましょう。

### AI と機械学習（このコースの中心）

:::tip[この例を実行する前に scikit-learn をインストールしてください]
Colab や Jupyter で以下のコードを実行する前に、まずインストールしてください（1 回だけで OK）:
```bash
!pip install scikit-learn
```
ローカルのターミナルや Conda 環境では次を使います: `pip install scikit-learn`
:::
```python
# 数行でシンプルな線形回帰モデルを学習する（サンプルデータ、直接実行できます）
import numpy as np
from sklearn.linear_model import LinearRegression

X_train = np.array([[1], [2], [3], [4], [5]])   # 特徴量
y_train = np.array([2, 4, 6, 8, 10])             # ラベル（y ≈ 2*x）

model = LinearRegression()
model.fit(X_train, y_train)
# 学習後は model.predict() で予測できます
```

主なフレームワーク: PyTorch、TensorFlow、scikit-learn、Hugging Face Transformers

### データ分析と可視化

```python
import pandas as pd
import matplotlib.pyplot as plt

# サンプルデータ（実際のプロジェクトでは pd.read_csv("sales.csv") で自分のファイルを読み込めます）
data = pd.DataFrame({"month": ["1月", "2月", "3月"], "revenue": [100, 150, 120]})

# 1 行でグラフを描画
data.plot(x="month", y="revenue", kind="bar")
plt.show()
```

主なライブラリ: pandas、NumPy、Matplotlib、Seaborn

### Web バックエンド開発

Python を使うと、API を提供する Web バックエンドをすばやく作れます。たとえば次のようになります。

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def say_hello():
    return {"message": "こんにちは、世界！"}
```

**サービスを起動してアクセスする:**

1. まず上のコードをファイル（例: `main.py`）として保存し、ターミナルでそのディレクトリに移動して次を実行します。
   ```bash
   pip install fastapi uvicorn
   uvicorn main:app --reload
   ```
2. ターミナルに `Uvicorn running on http://127.0.0.1:8000` と表示されたら、ブラウザで次を開きます。
   - **http://127.0.0.1:8000/hello** → `{"message":"こんにちは、世界！"}` が返ります
   - **http://127.0.0.1:8000/docs** → 自動生成された API ドキュメントページ。ここからそのまま試せます

主なフレームワーク: FastAPI、Django、Flask

### 自動化スクリプト

```python
import os

# 例: フォルダ内の画像を一括リネームする（まずテスト用ディレクトリを作ってから実行し、FileNotFoundError を避けましょう）
os.makedirs("photos", exist_ok=True)
for i in range(3):
    open(f"photos/old_{i}.jpg", "w").close()   # 3 つの空ファイルを作って例にする

for i, filename in enumerate(os.listdir("photos/")):
    new_name = f"photo_{i+1}.jpg"
    os.rename(f"photos/{filename}", f"photos/{new_name}")

# 結果を確認（実際のプロジェクトではテスト用ディレクトリを削除してもよい: os.removedirs など）
print(os.listdir("photos/"))   # ['photo_1.jpg', 'photo_2.jpg', 'photo_3.jpg']
```

### Web スクレイピング

```python
# 先にインストール: !pip install beautifulsoup4
from bs4 import BeautifulSoup

# サンプル HTML を使って解析を示す（外部ネットワークに依存せず、直接実行できます）
html = """
<html><body>
  <h1>Python を学ぼう</h1>
  <p>1 つ目の段落</p>
  <p>2 つ目の段落</p>
</body></html>
"""
soup = BeautifulSoup(html, "html.parser")
title = soup.find("h1").text
paragraphs = soup.find_all("p")
print(f"Web ページのタイトル: {title}")
print(f"段落の数: {len(paragraphs)} 個")
```

---

## 最初の Python プログラムを書こう

### 方法 1: ターミナルで Python の対話モードを使う

ターミナルを開いて（第 1 ステップで学びました）、次を入力します。

```bash
python
```

次のようなプロンプトが表示されます。

```
Python 3.11.5 (main, Sep 11 2023, 08:31:25)
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

`>>>` は Python の対話プロンプトで、コマンド入力を待っていることを意味します。

次を試してみましょう。

```python
>>> print("Hello, World!")
Hello, World!

>>> 1 + 1
2

>>> "AI" * 3
'AIAIAI'

>>> len("Python")
6
```

対話モードを終了するには、`exit()` を入力するか、`Ctrl+D` を押します。

:::tip[対話モードの使いどころ]
対話モードは**すぐに試したいとき**にとても便利です。たとえば、ある関数の使い方がよくわからないときに、まず対話モードで試してみて、問題なければファイルに書く、という使い方ができます。
:::
### 方法 2: VS Code で書いて実行する

1. VS Code を開きます（第 1 ステップですでにインストール済み）
2. 新しいファイル `hello.py` を作成します（拡張子が `.py` であることに注意）
3. 次のコードを入力します。

```python
# これが私の最初の Python プログラムです
print("Hello, World!")
print("私は Python を学んでいます！")
print("1 + 1 =", 1 + 1)
```

4. ファイルを保存します（`Ctrl+S` / `Cmd+S`）
5. ターミナルで次を実行します。

```bash
python hello.py
```

出力:

```
Hello, World!
私は Python を学んでいます！
1 + 1 = 2
```

おめでとうございます。あなたの最初の Python プログラムができました！

### 方法 3: Jupyter Notebook で実行する

第 1 ステップですでに Jupyter をインストールしています。次で起動します。

```bash
jupyter notebook
```

新しい Notebook を作成し、コードセルに `print("Hello from Jupyter!")` と入力して `Shift+Enter` で実行します。

:::note[3 つの方法はどう選ぶ？]
- **対話モード**: 少しのコードをすぐ試したいとき
- **VS Code + .py ファイル**: 本格的なプロジェクトコードを書くとき
- **Jupyter Notebook**: データ分析、学習用の実験（このコースでは主にこれを使います）
:::
---

## Python コードの基本ルール

本格的に学ぶ前に、まず最も基本的なルールを確認しましょう。

### インデントが重要

Python では、他の言語のように波かっこ `{}` を使わず、**インデント**（通常は 4 スペース）でコードブロックを表します。

```python
# 正しい ✅
if True:
    print("4 スペースでインデントされています")
    print("同じコードブロックです")
```

次の例は意図的に間違っています。実行すると `IndentationError` が発生します。

```text
if True:
print("インデントがないので Python はエラーになります")
```

:::caution[注意]
インデントミスは初心者が最もよくやるミスです。VS Code は自動でインデントを手伝ってくれますが、コードをコピペしたときは、インデントが正しいか必ず確認しましょう。
:::
### コメントは `#` を使う

```python
# これはコメントです。Python は無視します
print("この行は実行されます")  # 行末にもコメントを書けます

# 複数行コメントは、# で始まる行を並べます
# 1 行目のコメント
# 2 行目のコメント
```

コメントは人間のために書くものです。あなた自身や他の人がコードを理解しやすくなります。良いコメントは、**何をしたか**ではなく、**なぜそうしたか**を説明します。

### 大文字と小文字を区別する

```python
service_name = "ログイン API"
Service_Name = "検索 API"
SERVICE_NAME = "Worker"
# これらは 3 つの別々の変数です！

print(service_name)   # ログイン API
Print(service_name)   # エラー！Python に Print はなく、print だけです
```

### ファイルの末尾は `.py`

Python スクリプトの拡張子は `.py` です。たとえば `hello.py`、`train.py`、`model.py` です。

---

## Python 2 それとも Python 3？

短く答えると、**Python 3 を使ってください。Python 2 は使わないでください。**

Python 2 は 2020 年 1 月 1 日に正式にサポート終了しました。新しいプロジェクトや現代的なライブラリは、すべて Python 3 のみをサポートしています。このコースでは **Python 3.10 以上** を使います。

Python のバージョンを確認しましょう。

```bash
python --version
# Python 3.10.x 以上が表示されるはずです
```

もし `Python 2.x` と表示されたら、`python3` コマンドを使うか、第 1 ステップで設定した conda 環境が正しく有効になっているか確認してください。

---

## ハンズオン演習

### 演習 1: Hello World の発展版

`about_me.py` というファイルを作成し、自己紹介を出力してみましょう。

```python
print("=== 自己紹介 ===")
print("名前：[あなたの名前]")
print("目標：AI エンジニアになること")
print("学習中：Python プログラミング")
print("=" * 20)
```

実行して、出力を確認してください。内容を変更して、もっと情報を追加してみましょう。

### 演習 2: Python を電卓として使う

Python の対話モードで、次の計算を試してみましょう。

```python
>>> 100 + 200
>>> 10 * 3.14
>>> 2 ** 10        # ** は累乗、2 の 10 乗
>>> 17 / 5         # 割り算
>>> 17 // 5        # 切り捨て除算（小数部分を捨てる）
>>> 17 % 5         # 余り
```

それぞれの結果を記録して、なぜそうなるのか考えてみましょう。

### 演習 3: print() を調べる

次のコードを試して、`print()` のさまざまな使い方を観察しましょう。

```python
print("Hello")
print("Hello", "World")           # 複数の引数はカンマで区切る
print("Hello", "World", sep="-")  # - でつなぐ
print("Hello", end=" ")           # 改行しない
print("World")
print("価格:", 99.9, "円")
```

<details>
<summary>参考実装と解説</summary>

1. `about_me.py` はターミナルから実行でき、読みやすい複数行の自己紹介を表示できる状態にします。文面を変えるだけなら Python の構文を変える必要はありません。
2. 計算機の出力には、`300`、`31.400000000000002` または近い浮動小数点値、`1024`、`3.4`、`3`、`2` が含まれます。
3. `print("Hello", "World")` は空白を挿入します。`sep="-"` は区切り文字を変え、`end=" "` は次の `print()` を同じ行につなげます。
4. スクリプトが動かないときは、ファイル名、現在のフォルダ、Python 3 の実行環境、引用符や丸括弧の対応を確認します。
5. 証拠として、コードだけでなく実行コマンドと出力も残します。

</details>

---

## まとめ

| ポイント | 説明 |
|------|------|
| Python は AI 開発の第一選択 | ほとんどの AI フレームワークは Python を基盤にしている |
| 文法がシンプルで自然言語に近い | 学習のハードルが下がり、ロジックに集中できる |
| エコシステムが豊富 | 40 万以上のサードパーティライブラリがあり、ほとんどのニーズに既製の解決策がある |
| 実行方法は 3 種類ある | 対話モード、.py ファイル、Jupyter Notebook |
| インデントは Python の命 | 4 スペースでインデントし、Tab は使わない |

:::tip[学習のコツ]
プログラミングは**技能**です。見るだけでは身につきません。各レッスンの練習は、必ず自分で手を動かして打ち込んでください。コピー＆ペーストではなく、1 文字ずつ入力してみましょう。入力する中でミスをし、デバッグし、より深く理解できるようになります。
:::