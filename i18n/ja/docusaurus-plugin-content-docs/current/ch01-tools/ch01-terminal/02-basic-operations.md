---
title: "ターミナルの基本操作"
sidebar_position: 2
description: "コアコマンド、パスの考え方、パイプ、環境変数を身につける"
---

# ターミナルの基本操作

![ターミナルのパスとコマンド実行の関係図](/img/course/ch01-terminal-path-command-map-ja.png)

## この節の位置づけ

この節から、いよいよ本格的にターミナルを使っていきます。まずは「今どのディレクトリにいるのか」を判断できるようになり、そのうえでよく使うファイル・ディレクトリ・パス・パイプ・環境変数の操作を身につけます。これは、後で Python を実行したり、依存関係を管理したり、Git を使ったりするための土台になります。

## 学習目標

- 日常操作の 90% をカバーする 10+ 個の基本コマンドを身につける
- 絶対パスと相対パスを理解する
- パイプとリダイレクトの使い方を学ぶ
- 環境変数の考え方を理解する

---

## ターミナルを開く

まず、ターミナルを見つけて開きましょう。

| オペレーティングシステム | 開き方 |
|---------|---------|
| **Windows** | "PowerShell" または "Windows Terminal" を検索して開く |
| **macOS** | `Command + 空格` で "Terminal" を検索し、Enter で開く |
| **Linux** | `Ctrl + Alt + T` |

ウィンドウが開き、点滅するカーソルが表示されていれば、そこがターミナルです。ここにコマンドを入力していきます。

:::info Windows ユーザー向けの選択
Windows には複数のターミナルがあります。**Windows Terminal** の使用をおすすめします（Microsoft Store から無料でインストールできます）。その中で PowerShell のタブを選んでください。この講座のコマンドは macOS/Linux を中心に説明しますが、Windows でもほとんど同じです。違いがある箇所はその都度明記します。
:::

---

## 第1部：パス——今どこにいる？

コマンドラインにはグラフィカルな画面がないので、「どのフォルダの中のものを操作するか」を文字で伝える必要があります。これが**パス**です。

### 今どこにいる？

```bash
pwd
```

`pwd` = **P**rint **W**orking **D**irectory（現在の作業ディレクトリを表示）

出力は次のようになります。

```
/Users/zhangsan          # macOS
/home/zhangsan           # Linux
C:\Users\zhangsan        # Windows PowerShell
```

これが今いるフォルダで、**作業ディレクトリ**と呼びます。

### 絶対パス vs 相対パス

```
/Users/zhangsan/projects/ai-course/data/train.csv
```

これは**絶対パス**です。ルートディレクトリ `/` から始まり、ファイルの場所を完全に表しています。現実の住所でいうと、「中国北京市海淀区中関村大街1号」のような、全部書いた住所です。

```
data/train.csv
```

これは**相対パス**です。今いるフォルダを基準にしたパスです。もし今 `/Users/zhangsan/projects/ai-course/` にいるなら、`data/train.csv` は上の絶対パスと同じ場所を指します。現実でいえば「となりの建物の2階」のような言い方です。

### パスの中の特別な記号

| 記号 | 意味 | 例 |
|------|------|------|
| `/` | ルートディレクトリ（すべてのファイルの起点） | `cd /` |
| `~` | 現在のユーザーのホームディレクトリ | `cd ~` は `cd /Users/zhangsan` と同じ |
| `.` | 現在のディレクトリ | `./run.py` は現在のディレクトリにある run.py を表す |
| `..` | 1つ上のディレクトリ | `cd ..` で1つ上へ戻る |

理解を助ける練習です。

```bash
# もし /Users/zhangsan/projects/ai-course にいると仮定する

pwd                    # 出力: /Users/zhangsan/projects/ai-course
cd ..                  # 1つ上へ戻る
pwd                    # 出力: /Users/zhangsan/projects
cd ~                   # Home ディレクトリへ戻る
pwd                    # 出力: /Users/zhangsan
cd ~/projects/ai-course  # 絶対パス風に戻る
pwd                    # 出力: /Users/zhangsan/projects/ai-course
```

---

## 第2部：基本コマンド

以下のコマンドは、毎日のように使います。まずは実際に入力してみましょう。暗記する必要はありません。何度も使ううちに自然と覚えられます。

### 移動コマンド

#### `cd` — ディレクトリを切り替える

```bash
cd projects        # projects フォルダに入る
cd ..              # 1つ上へ戻る
cd ~               # Home ディレクトリへ戻る
cd ~/Desktop       # デスクトップへ移動
cd -               # 直前にいたディレクトリへ戻る（とても便利！）
```

#### `ls` — ファイルを一覧表示する

```bash
ls                 # 現在のディレクトリのファイルとフォルダを表示
ls -l              # 詳細表示（サイズ、日付、権限を表示）
ls -a              # 隠しファイル（. で始まるファイル）も表示
ls -la             # 2つを組み合わせる
ls projects/       # projects フォルダの中身を表示
```

:::note Windows PowerShell
PowerShell でも `ls` は使えます（`Get-ChildItem` の別名です）。`ls -la` は使えないので、隠しファイルを表示するには `ls -Force` を使います。
:::

### ファイルとフォルダの操作

#### `mkdir` — フォルダを作る

```bash
mkdir my-project               # フォルダを1つ作成
mkdir -p a/b/c                 # 複数階層のフォルダをまとめて作成
```

#### `touch` — 空ファイルを作る

```bash
touch hello.py                 # 空の Python ファイルを作成
touch README.md                # 空の Markdown ファイルを作成
```

:::note Windows
PowerShell には `touch` がないので、代わりに `New-Item hello.py` を使います。
:::

#### `cp` — コピーする

```bash
cp file.txt file_backup.txt          # ファイルをコピー
cp file.txt ~/Desktop/               # デスクトップへコピー
cp -r my-folder/ my-folder-backup/   # フォルダ全体をコピー（-r は再帰的）
```

#### `mv` — 移動 / 名前変更

```bash
mv old_name.py new_name.py       # ファイル名を変更
mv file.txt ~/Desktop/           # デスクトップへ移動
mv project/ ~/projects/          # フォルダを移動
```

#### `rm` — 削除する

```bash
rm file.txt                  # ファイルを削除
rm -r my-folder/             # フォルダとその中身をすべて削除
```

:::warning コマンドラインの削除にはゴミ箱がない
`rm` で削除したファイルはゴミ箱には入りません。直接消えます。実行する前に、本当に消してよいものか確認してください。習慣として、削除前に `ls` で一度確認しましょう。
:::

### ファイルの中身を見る

```bash
cat file.txt          # ファイル全体の内容を表示（小さいファイル向き）
head file.txt         # 最初の10行を表示
head -20 file.txt     # 最初の20行を表示
tail file.txt         # 最後の10行を表示
tail -f log.txt       # ファイル更新をリアルタイムで追う（ログ確認に便利）
```

### 検索

```bash
grep "error" log.txt              # "error" を含む行を検索
grep -r "import torch" ./         # 現在のディレクトリ配下のすべてのファイルから検索
grep -n "def train" model.py      # 行番号付きで検索結果を表示
```

`grep` は、今後のデバッグでとても役立つ味方です。たくさんのファイルの中から、ある関数や変数がどこで使われているかをすばやく見つけられます。

### そのほかの便利なコマンド

```bash
clear              # 画面をきれいにする（または Ctrl + L）
history            # これまで実行したコマンドを表示
which python       # python コマンドの場所を確認（環境の問題調査でよく使う）
echo "hello"       # 文字を表示する
```

---

## 第3部：パイプとリダイレクト

この2つは、コマンドラインを本当に強力にする仕組みです。

### パイプ `|`

パイプの意味は、「前のコマンドの出力を、次のコマンドの入力として渡す」ことです。

```bash
# すべてのファイルを一覧表示し、その中から .py ファイルを探す
ls -la | grep ".py"

# 履歴の中から使った git コマンドを探す
history | grep "git"

# 現在のディレクトリにある Python ファイルの数を数える
ls *.py | wc -l
```

パイプは、工場の流れ作業のようなものだと考えるとわかりやすいです。ある工程の出力が、次の工程の材料になります。

### リダイレクト `>` と `>>`

コマンドの出力を画面に表示する代わりに、ファイルに保存できます。

```bash
# ls の結果を filelist.txt に保存する（上書き）
ls -la > filelist.txt

# 結果をファイルの末尾に追加する（上書きしない）
echo "新しい1行" >> notes.txt

# Python スクリプトの出力をファイルに保存する
python train.py > training_log.txt
```

`>` は上書き、`>>` は追記です。実務では、学習ログの保存によく使います。

### 組み合わせて使う

```bash
# スクリプトを実行し、通常出力とエラー出力の両方をログファイルに保存する
python train.py > log.txt 2>&1

# 1つの Python ファイルに何行コードがあるか数える
cat model.py | wc -l

# "TODO" を含むファイルを見つけ、その件数を数える
grep -r "TODO" ./ | wc -l
```

---

## 第4部：環境変数

環境変数は、システムに保存されている「全体設定」のようなものです。多くのプログラムは、これを読み取って動作を決めます。

### 環境変数を見る

```bash
# すべての環境変数を表示
env

# 特定の環境変数の値を見る
echo $PATH
echo $HOME
```

### 最重要の環境変数：PATH

`PATH` は、ターミナルでコマンドを入力したとき、システムがどのディレクトリを見に行ってそのコマンドを探すかを決めます。

```bash
echo $PATH
# 出力例: /usr/local/bin:/usr/bin:/bin:/Users/zhangsan/miniconda3/bin
```

これらのパスは `:` で区切られています。`python` と入力すると、システムはこれらのディレクトリを順に探し、最初に見つかった `python` を実行します。

`command not found`（コマンドが見つからない）というエラーが出たら、多くの場合、そのプログラムが `PATH` に含まれるどのディレクトリにもないことが原因です。

### 環境変数を設定する

```bash
# 一時的に設定する（今開いているターミナルウィンドウでのみ有効）
export MY_API_KEY="your_api_key_here"
echo $MY_API_KEY    # 出力: your_api_key_here

# 確認：ターミナルを閉じて再度開くと MY_API_KEY は消える
```

```bash
# 永続的に設定する（設定ファイルに書き込む）
# macOS/Linux で zsh を使う場合：
echo 'export MY_API_KEY="your_api_key_here"' >> ~/.zshrc
source ~/.zshrc    # すぐに反映

# bash を使う場合：
echo 'export MY_API_KEY="your_api_key_here"' >> ~/.bashrc
source ~/.bashrc
```

:::info なぜ環境変数を知る必要があるの？
この先の学習では、API Key（たとえば OpenAI のキー）を環境変数に保存することがよくあります。こうすると、コードの中に直接書くよりずっと安全です。

```python
import os
api_key = os.environ.get("OPENAI_API_KEY")
```
:::

---

## 実践練習

ターミナルを開いて、次の操作を順番に行ってみましょう。

```bash
# 1. 今どこにいるか確認
pwd

# 2. Home ディレクトリへ移動
cd ~

# 3. 学習用プロジェクトフォルダを作成
mkdir -p ai-study/ch02-python/terminal-practice

# 4. そのフォルダへ移動
cd ai-study/ch02-python/terminal-practice

# 5. いくつかファイルを作成
touch hello.py notes.txt data.csv

# 6. 作成されたファイルを確認
ls -la

# 7. ファイルに少し書き込む
echo "print('Hello, AI!')" > hello.py
echo "1日目の学習メモ" > notes.txt

# 8. ファイルの内容を確認
cat hello.py
cat notes.txt

# 9. notes.txt をコピーしてバックアップを作る
cp notes.txt notes_backup.txt

# 10. バックアップが成功したか確認
ls

# 11. notes.txt に内容を追加
echo "cd, ls, mkdir, touch, cp, cat コマンドを学んだ" >> notes.txt
cat notes.txt

# 12. "AI" を含むファイルを検索
grep -r "AI" ./

# 13. 1つ上のディレクトリへ戻る
cd ..
pwd
```

すべての手順がうまくいけば、おめでとうございます。これでコマンドラインの最も重要な操作を身につけました。

---

## よく使うコマンド早見表

| コマンド | 用途 | よく使う引数 |
|------|------|---------|
| `pwd` | 現在のディレクトリを表示 | |
| `cd` | ディレクトリを切り替える | `..` 上へ、`~` Home、`-` 前回 |
| `ls` | ファイルを一覧表示 | `-l` 詳細、`-a` 隠しファイル |
| `mkdir` | フォルダを作る | `-p` 多階層作成 |
| `touch` | 空ファイルを作る | |
| `cp` | コピーする | `-r` フォルダをコピー |
| `mv` | 移動/名前変更 | |
| `rm` | 削除する | `-r` フォルダ削除 |
| `cat` | ファイルを見る | |
| `head` / `tail` | 先頭/末尾を見る | `-n 数字` で行数指定 |
| `grep` | テキスト検索 | `-r` 再帰的、`-n` 行番号 |
| `echo` | 文字を表示 | |
| `clear` | 画面を消去 | |
| `history` | コマンド履歴 | |
| `which` | コマンドの場所を確認 | |

:::tip 覚えられないときは？
この表を暗記する必要はありません。何度も使えば自然に覚えられます。最初のうちは印刷して画面の横に貼っておくか、スマホに保存してすぐ見られるようにすると便利です。多くのコマンドは `コマンド --help` と入力すれば使い方を確認できます。
:::
