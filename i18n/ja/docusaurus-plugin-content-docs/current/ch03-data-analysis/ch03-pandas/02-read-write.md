---
title: "3.2 データの読み書き"
sidebar_position: 10
description: "CSV、Excel、JSON などの形式のデータの読み書きを身につける"
---

# データの読み書き

:::tip この節の位置づけ
多くの初心者は、`read_csv()` を初めて学ぶと、こう感じます。

- ただファイルを読み込むだけじゃないの？

でも、実際の分析では、多くの問題はこの「読み込む」段階から始まります。

- 文字コードが合っていない
- 区切り文字が違う
- ヘッダーの判定を間違えた
- 日付が日付として解析されていない

だからこの節でいちばん大事なのは、パラメータを覚えることではなく、まず次の考え方を身につけることです。

> **データを読むことは機械的な取り込みではなく、「この表が自分の想定どおりに正しく読めているか」を確認する作業です。**
:::

## 学習目標

- CSV ファイルの読み込みと書き出しを身につける
- Excel、JSON などの形式の読み書きを理解する
- よく使うパラメータ設定を覚える
- 大きなファイルを分割して読むコツを理解する

---

## まずは全体の地図を作ろう

データの読み書きは、「まず読み込んで、それから正しく読めたか確認する」と考えると理解しやすいです。

![Pandas データ読み書きの初対面フロー](/img/course/ch03-pandas-read-write-first-look.png)

この節で本当に解決したいのは、次の 2 つです。

- データはどうやって読み込まれるのか
- 読み込んだあと、最初に何を確認すべきか

## CSV ファイルを読む

CSV（Comma-Separated Values）は、データ分析で最もよく使われるファイル形式です。

### 初心者にいちばん合うたとえ

データの読み書きは、次のように考えられます。

- 外のファイルを、自分の分析用の作業台に運び込む

このとき怖いのは、「運び込めないこと」よりも、

- 運び込めたけれど、形式が崩れてしまうこと

たとえば、次のようなことがあります。

- 日付がまだ文字列のまま
- 日本語が文字化けする
- 1 行目は本当はヘッダーではないのに、ヘッダーとして扱われてしまう

### 基本の読み込み

```python
import pandas as pd

# CSV ファイルを読み込む
df = pd.read_csv("titanic_sample.csv")
print(df.head())       # 最初の 5 行を見る
print(df.shape)        # 行数と列数
print(df.info())       # 列の情報を見る
```

この 1 行だけで、Pandas は自動的に次のことをやってくれます。

- ヘッダーを認識する（1 行目を列名として使う）
- 各列のデータ型を推測する
- 行インデックスを作る（0, 1, 2, ...）

### よく使うパラメータ

```python
# 区切り文字を指定する（tab やセミコロンで区切られているファイルもある）
df = pd.read_csv("data.tsv", sep="\t")         # tab 区切り
df = pd.read_csv("data.csv", sep=";")          # セミコロン区切り

# 文字コードを指定する（日本語・中国語ファイルでよくある問題）
df = pd.read_csv("chinese_data.csv", encoding="utf-8")
df = pd.read_csv("chinese_data.csv", encoding="gbk")     # 一部の Windows 出力ファイル

# ヘッダーがないファイル
df = pd.read_csv("no_header.csv", header=None)
df = pd.read_csv("no_header.csv", header=None, names=["col1", "col2", "col3"])

# ある列をインデックスにする
df = pd.read_csv("data.csv", index_col="id")
df = pd.read_csv("data.csv", index_col=0)      # 1 列目をインデックスにする

# 一部の列だけ読む
df = pd.read_csv("data.csv", usecols=["Name", "Age", "Fare"])

# 最初の 100 行だけ読む
df = pd.read_csv("data.csv", nrows=100)

# 欠損値の表記を指定する
df = pd.read_csv("data.csv", na_values=["NA", "N/A", "missing", "-"])

# データ型を指定する
df = pd.read_csv("data.csv", dtype={"Age": float, "Pclass": str})
```

### パラメータ早見表

| パラメータ | 役割 | 例 |
|------|------|------|
| `sep` | 区切り文字 | `sep="\t"` |
| `encoding` | 文字コード | `encoding="utf-8"` |
| `header` | ヘッダー行番号 | `header=None` |
| `names` | 列名を指定する | `names=["a","b"]` |
| `index_col` | インデックス列 | `index_col="id"` |
| `usecols` | 一部の列だけ読む | `usecols=["Name","Age"]` |
| `nrows` | 読む行数 | `nrows=100` |
| `skiprows` | 飛ばす行数 | `skiprows=5` |
| `na_values` | 欠損値の表記 | `na_values=["NA","-"]` |
| `dtype` | 型を指定する | `dtype={"Age": float}` |
| `parse_dates` | 日付列を解析する | `parse_dates=["date"]` |

### 初めて新しいファイルを読むときの、いちばん安定した順番

より安全なのは、次の順番です。

1. まずそのまま読み込む
2. まず `shape` を見る
3. 次に `head()` を見る
4. それから `info()` を見る
5. おかしいところがあれば、あとからパラメータを足す

最初からすべてのパラメータを書き込むより、この方が問題を見つけやすいです。

---

## CSV ファイルに書き出す

```python
# 基本の書き出し
df.to_csv("output.csv")

# インデックスを保存しない（通常はこちらをおすすめ）
df.to_csv("output.csv", index=False)

# 文字コードを指定する
df.to_csv("output.csv", index=False, encoding="utf-8-sig")  # Excel で扱いやすい UTF-8

# 区切り文字を指定する
df.to_csv("output.tsv", index=False, sep="\t")
```

:::tip 日本語 CSV が Excel で文字化けする？
保存するときに `encoding="utf-8-sig"`（BOM 付き UTF-8）を使うと、Excel でも日本語を正しく表示しやすくなります。
:::

---

## Excel ファイルの読み書き

```python
# Excel を読む（openpyxl ライブラリが必要：pip install openpyxl）
df = pd.read_excel("data.xlsx")

# 特定のシートを読む
df = pd.read_excel("data.xlsx", sheet_name="Sheet2")
df = pd.read_excel("data.xlsx", sheet_name=1)      # インデックスで指定

# すべてのシートを読む（辞書が返る）
all_sheets = pd.read_excel("data.xlsx", sheet_name=None)
for name, sheet_df in all_sheets.items():
    print(f"シート {name}: {sheet_df.shape}")

# Excel に書き出す
df.to_excel("output.xlsx", index=False)

# 複数シートに書き出す
with pd.ExcelWriter("output.xlsx") as writer:
    df1.to_excel(writer, sheet_name="販売データ", index=False)
    df2.to_excel(writer, sheet_name="ユーザーデータ", index=False)
```

---

## JSON ファイルの読み書き

```python
# JSON を読む
df = pd.read_json("data.json")

# さまざまな JSON 形式
# records 形式：[{"name": "张三", "age": 22}, {...}]
df = pd.read_json("data.json", orient="records")

# JSON に書き出す
df.to_json("output.json", orient="records", force_ascii=False, indent=2)
# force_ascii=False：日本語をエスケープしない
# indent=2：見やすく整形する
```

---

## 大きなファイルを扱う：分割読み込み

ファイルが大きすぎて、一度にメモリへ読み込めないときは、分割して読み込みます。

```python
# 分割読み込み：毎回 1000 行ずつ読む
chunks = pd.read_csv("huge_file.csv", chunksize=1000)

# 1 つずつ処理する
results = []
for chunk in chunks:
    # 各ブロックに処理を行う
    filtered = chunk[chunk["Age"] > 30]
    results.append(filtered)

# すべてのブロックを結合する
df_final = pd.concat(results, ignore_index=True)
print(f"合計 {len(df_final)} 件を抽出しました")
```

```python
# もう 1 つのよくある使い方：大きなファイルの総行数を数える
total_rows = sum(len(chunk) for chunk in pd.read_csv("huge_file.csv", chunksize=10000))
print(f"総行数: {total_rows}")
```

### 初心者がまず覚えるとよい判断表

| 症状 | まず疑うとよいもの |
|---|---|
| ファイルを読むと文字化けする | `encoding` |
| 1 列に全部まとまってしまう | `sep` |
| 日付をそのまま時間分析できない | `parse_dates` |
| メモリが足りない | `chunksize` |
| 1 行目の扱いを間違えた | `header / names` |

この表は初心者にとても役立ちます。なぜなら、「ファイルが読めない・読めていない」を、よくある入口の問題に分けて考えられるからです。

---

## ほかの形式

```python
# HTML の表を読む（lxml または html5lib が必要）
# tables = pd.read_html("https://example.com/data.html")

# クリップボードから読む（Excel からコピーしたあと）
# df = pd.read_clipboard()

# Parquet を読む（効率のよい列指向ストレージ形式。大規模データでよく使う）
# df = pd.read_parquet("data.parquet")

# SQL データベースを読む（第 4 章で詳しく扱う）
# import sqlite3
# conn = sqlite3.connect("database.db")
# df = pd.read_sql("SELECT * FROM users", conn)
```

---

## 実践：データを読み込んで最初に確認する

```python
import pandas as pd

# 売上データがあるとする
# ファイルがなければ、まずサンプルデータを作る
data = {
    "日付": ["2024-01-15", "2024-01-16", "2024-01-17", "2024-01-18", "2024-01-19"],
    "商品": ["リンゴ", "牛乳", "パン", "リンゴ", "牛乳"],
    "数量": [50, 30, 45, 60, 25],
    "単価": [5.5, 8.0, 3.5, 5.5, 8.0],
    "销售额": [275.0, 240.0, 157.5, 330.0, 200.0]
}
df = pd.DataFrame(data)

# CSV として保存
df.to_csv("sales.csv", index=False)

# もう一度読み込む
df = pd.read_csv("sales.csv")

# 標準的な「初対面」フロー
print("=== データの形状 ===")
print(df.shape)

print("\n=== 最初の数行 ===")
print(df.head())

print("\n=== データの情報 ===")
print(df.info())

print("\n=== 統計サマリー ===")
print(df.describe())
```

### この小さな実践で、まず何を学ぶべき？

いちばん大事なのは、`read_*` 関数名を 1 つ覚えることではありません。  
新しいデータを受け取ったときの、いちばん安定した 3 ステップを覚えることです。

1. `shape` を見る
2. `head()` を見る
3. `info()` を見る

この 3 つがスムーズにできれば、CSV、Excel、JSON を読むときも、ずっと安定します。

---

## まとめ

| 操作 | 読み込み | 書き出し |
|------|------|------|
| CSV | `pd.read_csv()` | `df.to_csv()` |
| Excel | `pd.read_excel()` | `df.to_excel()` |
| JSON | `pd.read_json()` | `df.to_json()` |
| SQL | `pd.read_sql()` | `df.to_sql()` |
| Parquet | `pd.read_parquet()` | `df.to_parquet()` |

:::tip データを受け取ったら最初にやること
必ず次の 3 行を実行しましょう。
```python
print(df.shape)
df.info()
df.head()
```
:::

## この節でぜひ持ち帰ってほしいこと

- データの読み書きは、「ファイルを取り込む」ことではなく、データが正しく読み込めたかを確認すること
- 初めてファイルを受け取ったら、まず `shape / head / info` を見る
- 読み込みの問題の多くは、`encoding / sep / header / parse_dates / chunksize` という種類のパラメータに関係している

---

## 手を動かして練習しよう

### 練習 1：CSV を作成して読み書きする

```python
# 1. 10 人の学生情報を含む DataFrame を作る（名前、年齢、成績）
# 2. CSV ファイルとして保存する（インデックスなし）
# 3. その CSV ファイルをもう一度読み込む
# 4. 読み込み前後でデータが一致していることを確認する
```

### 練習 2：実データを読む

[Kaggle](https://www.kaggle.com/datasets) から、興味のある小さなデータセット（CSV 形式）をダウンロードし、`pd.read_csv()` で読み込んで、「初対面」の流れを実行してみましょう。

### 練習 3：文字コードの問題を扱う

```python
# 日本語を含むデータを作り、utf-8 と gbk でそれぞれ保存する
# そのあと、別々の文字コードで読み込んで、文字化けの違いを観察する
```
