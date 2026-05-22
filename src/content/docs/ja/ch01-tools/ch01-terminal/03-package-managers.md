---
title: "1.1.3 パッケージマネージャー"
description: "パッケージマネージャーでシステムソフトウェアと開発ツールをインストールする"
sidebar:
  order: 3
---
![パッケージマネージャーの依存関係インストールフロー図](/img/course/ch01-package-manager-flow-ja.webp)

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
コマンド：実行した正確なターミナルコマンド
作業ディレクトリ：pwd/現在のフォルダと重要ファイルを列挙
出力：コマンド出力または結果のスクリーンショットをコピーしたもの
失敗確認: 間違ったパス、コマンド不足、権限の問題、またはシェル不一致
期待される成果: コマンドと結果を並べて示す再現可能なターミナル操作
```

## この節の位置づけ

この節で解決するのは、「開発ツールをどうやってインストールして更新するか」です。パッケージマネージャーを開発者向けのアプリストアだと理解し、自分のOSに合わせて Homebrew、winget、apt などのツールを使えるようになり、今後の Git、Python、データベース、デプロイツールのインストールの土台を作ります。

## 学習目標

- パッケージマネージャーとは何か、なぜ必要なのかを理解する
- 自分のOSに応じたパッケージマネージャーの使い方を身につける
- パッケージマネージャーで AI 開発に必要な基本ツールをいくつかインストールする

---

## パッケージマネージャーとは？

スマホで App を入れたいときは、App Store やアプリストアを開いて、検索して、インストールを押します。

**パッケージマネージャーは、パソコン版の「アプリストア」ですが、コマンドラインで操作します。** これが次の3つを手伝ってくれます。

1. **ソフトウェアのインストール**——1行のコマンドで完了。サイトに行ってインストーラーをダウンロードする必要はありません
2. **ソフトウェアの更新**——1行のコマンドで、すべてのソフトウェアを最新版に更新できます
3. **依存関係の管理**——「Aを入れるには先にBが必要」といった依存関係を自動で処理します

OSごとに使うパッケージマネージャーは異なります。自分のシステムに合うものを見つけて、そのまま進めましょう。

---

## macOS：Homebrew

[Homebrew](https://brew.sh) は macOS で最も人気のあるパッケージマネージャーで、ほとんどの開発者が入れています。

### Homebrew のインストール

ターミナルを開いて、貼り付けて実行します。

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

インストールには数分かかることがあります。パスワードを求められたら、Mac のログインパスワードを入力してください（入力しても文字は表示されません。正常です）。

インストールが終わったら、確認します。

```bash
brew --version
# 出力例: Homebrew 4.x.x
```

:::note[中国国内のユーザー向け]
ダウンロードが遅い場合は、"Homebrew 清华镜像" や "Homebrew 中科大镜像" で検索して、国内の高速ミラーを使う方法があります。
:::
### Homebrew のよく使うコマンド

```bash
# ソフトウェアを検索
brew search git

# ソフトウェアをインストール
brew install git
brew install wget
brew install tree

# インストール済みのソフトウェアを確認
brew list

# すべてのソフトウェアを更新
brew update      # Homebrew 本体を更新
brew upgrade     # インストール済みのソフトウェアをすべて更新

# ソフトウェアをアンインストール
brew uninstall wget

# ソフトウェア情報を表示
brew info git
```

### Homebrew で AI 開発の基本ツールを入れる

```bash
# Git（バージョン管理。次の章で詳しく学びます）
brew install git

# tree（ディレクトリを木構造で表示。プロジェクト構造を確認するのに便利）
brew install tree

# wget（ファイルをダウンロードするツール）
brew install wget
```

tree を入れたら、試してみましょう。

```bash
cd ~/ai-study
tree
```

出力例：

```
.
└── ch01-tools
    └── terminal-practice
        ├── data.csv
        ├── hello.py
        ├── notes.txt
        └── notes_backup.txt
```

`ls` よりも、ディレクトリ全体の構造が直感的に見えます。

---

## Ubuntu/Debian Linux：apt

`apt` は Ubuntu と Debian 系 Linux に最初から入っているパッケージマネージャーで、追加インストールは不要です。

### apt のよく使うコマンド

```bash
# ソフトウェアソース情報を更新（インストール前に先に実行するのがおすすめ）
sudo apt update

# ソフトウェアをインストール
sudo apt install git
sudo apt install tree
sudo apt install wget
sudo apt install curl

# ソフトウェアを検索
apt search nodejs

# インストール済みのソフトウェアを確認
apt list --installed

# すべてのソフトウェアを更新
sudo apt update && sudo apt upgrade

# ソフトウェアをアンインストール
sudo apt remove wget
```

:::note[sudo について]
`sudo` は「管理者権限で実行する」という意味です。システム全体に関わるソフトウェアのインストールには管理者権限が必要なので、`apt install` の前に `sudo` を付けます。パスワードの入力を求められます。
:::
### apt で AI 開発の基本ツールを入れる

```bash
sudo apt update
sudo apt install -y git tree wget curl build-essential
```

`-y` は自動で確認するという意味で、手動で "Y" を入力する必要はありません。`build-essential` にはコンパイル用ツールが含まれていて、Python ライブラリのインストール時に必要になることがあります。

---

## Windows：winget と Scoop

Windows には、主なコマンドライン用パッケージマネージャーが2つあります。

### 方案1：winget（おすすめ、Windows 標準）

Windows 10 (1709+) と Windows 11 には `winget` が標準で入っています。PowerShell を開いて試してみましょう。

```powershell
winget --version
```

出力があれば、使える状態です。

```powershell
# ソフトウェアを検索
winget search vscode

# ソフトウェアをインストール
winget install Git.Git
winget install Microsoft.VisualStudioCode
winget install Python.Python.3.11

# すべてのソフトウェアを更新
winget upgrade --all

# インストール済みのソフトウェアを確認
winget list
```

### 方案2：Scoop（Linux に近い使い心地）

もっと「開発者向け」で使いたいなら、[Scoop](https://scoop.sh) を入れるのもおすすめです。

```powershell
# Scoop をインストール（PowerShell で実行）
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
irm get.scoop.sh | iex
```

```powershell
# 使い方
scoop install git
scoop install python
scoop install tree

# 更新
scoop update *
```

### AI 開発の基本ツールを入れる（winget）

```powershell
winget install Git.Git
winget install Python.Python.3.11
```

:::tip[Windows ユーザーへの追加アドバイス]
**Windows Terminal** を入れることを強くおすすめします（Microsoft Store で検索できます）。標準の PowerShell ウィンドウよりずっと使いやすく、タブ対応、よりきれいなフォント表示、コピー＆ペーストのしやすさなどがあります。
:::
---

## パッケージマネージャー vs pip/conda

後の章で `pip` や `conda` も学びますが、「それらもパッケージマネージャーなのに、何が違うの？」と疑問に思うかもしれません。

| ツール | 管理するもの | たとえ |
|------|---------|------|
| **brew / apt / winget** | OS レベルのソフトウェア（Git、Python、Node.js、Docker） | スマホのアプリストア |
| **pip** | Python ライブラリ（numpy、pandas、torch） | Python 専用のアプリストア |
| **conda** | Python 環境 + Python ライブラリ + 一部のシステムライブラリ | より強力な Python アプリストア |

簡単に言うと：

- Git、Docker、システムツールを入れる → **brew / apt / winget**
- Python ライブラリを入れる → **pip** または **conda**
- Python のバージョンと仮想環境を管理する → **conda**

この3つは役割が違うので、ぶつかりません。

---

## 実践練習

自分のOSに合わせて、次の練習をやってみましょう。

### macOS ユーザー

```bash
# 1. Homebrew をインストール（まだ入れていない場合）
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 2. tree と wget をインストール
brew install tree wget

# 3. tree で、前に作った ai-study ディレクトリ構造を見る
tree ~/ai-study

# 4. wget でファイルを1つダウンロードしてみる
wget https://raw.githubusercontent.com/plotly/datasets/master/iris.csv
cat iris.csv | head -5
```

### Ubuntu ユーザー

```bash
# 1. ソフトウェアソースを更新
sudo apt update

# 2. tree と wget をインストール
sudo apt install -y tree wget

# 3. tree でディレクトリを見る
tree ~/ai-study

# 4. テスト用ファイルをダウンロード
wget https://raw.githubusercontent.com/plotly/datasets/master/iris.csv
head -5 iris.csv
```

### Windows ユーザー

```powershell
# 1. winget が使えるか確認
winget --version

# 2. Git をインストール（次の章で必要）
winget install Git.Git

# 3. インストールを確認
git --version
```

---

## この章のセルフチェック

次の項目を確認して、ターミナルの基礎が身についているかチェックしましょう。

- [ ] ターミナルを開いて、自分がどのディレクトリにいるか分かる
- [ ] `cd`、`ls`、`mkdir`、`touch`、`cp`、`mv`、`rm` で基本的なファイル操作ができる
- [ ] 絶対パスと相対パスの違いが分かる
- [ ] パイプ `|` で2つのコマンドをつなげられる
- [ ] `>` または `>>` で出力をファイルに保存できる
- [ ] 自分のパッケージマネージャーでソフトウェアを1つインストールできる
- [ ] `echo $PATH` の意味が分かる

<details>
<summary>確認の考え方と解説</summary>

1. macOS なら Homebrew、Windows なら winget、Linux なら apt など、自分の OS に合うツールを使えていれば十分です。
2. `tree --version` や `tree .` が動けば、インストール確認として合格です。
3. `wget` で `iris.csv` を取得できたら、`head -5 iris.csv` で中身の先頭を確認します。
4. Windows で Git を入れた場合、`git --version` が表示されれば PATH も通っています。
5. `PATH` はシェルが実行ファイルを探す場所の一覧です。インストールしたのに command not found になるときは PATH とターミナル再起動を疑います。

</details>

:::tip[すべてチェックできましたか？]
あなたはもう、ターミナルとコマンドラインの核心となるスキルを身につけました。次は Git を学びましょう。開発者にとってもう1つの必須ツールです。
:::