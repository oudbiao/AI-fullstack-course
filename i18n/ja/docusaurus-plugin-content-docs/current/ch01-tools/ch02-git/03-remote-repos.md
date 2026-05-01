---
title: "リモートリポジトリ"
sidebar_position: 3
description: "コードを GitHub にプッシュして、リモートでの協力を学ぶ"
---

# リモートリポジトリ

![Git ローカル・リモート同期図](/img/course/ch01-git-remote-sync-ja.png)

## この節の位置づけ

この節では、ローカルの Git リポジトリを GitHub に接続します。リモートリポジトリが、バックアップ・共同作業・作品集の公開という3つの役割を同時に担う理由を理解し、push、pull、clone を使ってコードをクラウドに同期する方法を学びます。

## 学習目標

- GitHub でリポジトリを作成する
- SSH 接続を設定する（もうパスワード入力は不要）
- `git push`、`git pull`、`git clone` を身につける
- よい README.md を書く

---

## なぜリモートリポジトリが必要なの？

ここまで、あなたの Git の記録は自分のパソコンの中にしかありませんでした。もしパソコンのハードディスクが壊れたら、コードも履歴もすべて失われてしまいます。

**リモートリポジトリ**とは、コードのコピーをクラウド（通常は GitHub）に保存しておくことです。これには3つの大きなメリットがあります。

1. **バックアップ**——パソコンが壊れても大丈夫。コードはクラウドにある
2. **共同作業**——複数人で同じリポジトリにコードを送れる
3. **公開**——あなたの GitHub はコード作品集になる。就職面接でも見られる

---

## GitHub に登録する

1. [github.com](https://github.com) を開く
2. **Sign up** をクリックして、メールアドレスで登録する
3. ユーザー名は英語で、短くて覚えやすいものがおすすめ（たとえば `zhangsan-dev`）。これはプロジェクトのリンクに表示されます

:::info 中国国内のユーザー向け
GitHub のアクセスが遅い場合は、予備として [Gitee](https://gitee.com) も一緒に登録しておくとよいです。操作方法はほぼ同じです。ただし、メインは GitHub をおすすめします。GitHub は世界最大のオープンソースプラットフォームで、就職のときにも価値が高いです。
:::

---

## SSH 接続を設定する

GitHub にコードを push するたびに本人確認が必要です。SSH はいちばん便利な方法で、一度設定すれば、以後パスワードを入力する必要がありません。

### 1つ目の手順：SSH 鍵を作成する

```bash
ssh-keygen -t ed25519 -C "あなたのメール@example.com"
```

いくつか質問されますが、すべて Enter を押せばOKです（デフォルト値を使います）。

```
Enter file in which to save the key (/Users/あなたのユーザー名/.ssh/id_ed25519): [Enter]
Enter passphrase (empty for no passphrase): [Enter]
Enter same passphrase again: [Enter]
```

### 2つ目の手順：公開鍵をコピーする

```bash
# macOS
cat ~/.ssh/id_ed25519.pub | pbcopy

# Linux
cat ~/.ssh/id_ed25519.pub
# その後、出力された内容を手動でコピーする

# Windows PowerShell
Get-Content ~/.ssh/id_ed25519.pub | Set-Clipboard
```

出力は次のようになります（これは公開鍵なので、安全に共有できます）。

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI... あなたのメール@example.com
```

### 3つ目の手順：GitHub に追加する

1. [github.com/settings/keys](https://github.com/settings/keys) を開く
2. **New SSH key** をクリックする
3. Title に "My Laptop" と入力する（または何でもOK。どのPCか分かる名前にする）
4. Key 欄に、先ほどコピーした公開鍵を貼り付ける
5. **Add SSH key** をクリックする

### 4つ目の手順：接続を確認する

```bash
ssh -T git@github.com
```

次のように表示されれば成功です。

```
Hi zhangsan! You've successfully authenticated, but GitHub does not provide shell access.
```

これで設定完了です！

:::tip SSH 鍵の仕組み（読み物）
SSH 鍵は「2つの鍵」のペアです。
- **秘密鍵**（`id_ed25519`）はあなたのパソコンに保存され、絶対に他人に渡してはいけません
- **公開鍵**（`id_ed25519.pub`）は GitHub に登録します

あなたが push するたびに、GitHub は公開鍵を使って「この人は対応する秘密鍵を持っているか」を確認します。確認できれば操作を許可します。これはパスワード入力よりも安全で、しかも便利です。
:::

---

## リモートリポジトリを作成して push する

### 例：以前作った AI プロジェクトを GitHub に push する

#### 方法1：先に GitHub でリポジトリを作成してから、ローカルプロジェクトとつなぐ

**1つ目の手順：GitHub でリポジトリを作成する**

1. [github.com/new](https://github.com/new) を開く
2. Repository name に `ai-image-classifier` と入力する
3. Description に "CNN を使ったシンプルな画像分類プロジェクト" と入力する
4. **Public** を選ぶ（公開して、他の人にも作品を見てもらう）
5. "Add a README file" は **チェックしない**（ローカルにすでにあるため）
6. **Create repository** をクリックする

**2つ目の手順：ローカルリポジトリを GitHub に接続する**

GitHub にはコマンドが表示されますが、必要なのは "push an existing repository" の部分です。

```bash
cd ai-image-classifier

# リモートリポジトリを関連付ける（zhangsan はあなたの GitHub ユーザー名に置き換える）
git remote add origin git@github.com:zhangsan/ai-image-classifier.git

# ローカルコードを GitHub に push する
git push -u origin main
```

`git remote add origin` の意味は、リモートリポジトリに `origin` という名前を付けることです（これは慣例的な名前です）。URL はその後ろのものです。

`-u origin main` の意味は、ローカルの `main` ブランチとリモートの `main` ブランチを関連付けることです。以後は `git push` だけでよく、毎回長いコマンドを書く必要はありません。

**3つ目の手順：確認する**

GitHub のページを更新すると、コード、コミット履歴、README が表示されているはずです。

#### 方法2：空のリポジトリを clone してから、ファイルを追加する

まだローカルコードがない場合は、逆の順番で進めてもかまいません。

```bash
# GitHub から空のリポジトリ（または他人のプロジェクト）を clone する
git clone git@github.com:zhangsan/my-new-project.git
cd my-new-project

# その中でコードを書く...
echo "print('hello')" > main.py

# commit して push する
git add .
git commit -m "メインプログラムを追加"
git push
```

---

## 日常的な push と pull

リモートリポジトリをつないだら、日常の操作はとても簡単です。

### git push：ローカルの新しい commit をリモートへ送る

```bash
# 新しいコードを書いた
echo "新機能" >> src/utils.py
git add .
git commit -m "データ前処理関数を追加"

# GitHub に push する
git push
```

### git pull：リモートの更新をローカルに取り込む

```bash
# たとえば、別のPC上で（または同僚が）コードを修正して GitHub に push したとする
# 最新のコードを取り込む必要がある
git pull
```

### 実際の仕事での流れ

```bash
# 毎日の作業を始める前：まず最新コードを取り込む
git pull

# コードを書く、修正する...

# 1つの機能が終わったら：commit して push
git add .
git commit -m "データ拡張モジュールを完成"
git push

# さらにコードを書く...

# また1つの機能が終わった
git add .
git commit -m "学習ログ記録機能を追加"
git push
```

---

## git clone：他人のプロジェクトをダウンロードする

Git で最初に使うことが多い操作かもしれません。GitHub からオープンソースプロジェクトをダウンロードすることです。

```bash
# AI 関連のオープンソースプロジェクトを clone する
git clone git@github.com:ultralytics/yolov5.git
cd yolov5
ls
```

`git clone` は次の3つを行います。
1. プロジェクトと同じ名前のフォルダを作成する
2. すべてのコードと完全な履歴をダウンロードする
3. リモートリポジトリの関連付けを自動で設定する

### clone した後によく使う操作

```bash
# このプロジェクトの commit 履歴を見る
git log --oneline -10    # 直近10件を見る

# どんなブランチがあるか確認する
git branch -a

# リモートリポジトリのアドレスを確認する
git remote -v
```

---

## README.md をしっかり書く

GitHub の各プロジェクトのトップページには、自動で `README.md` の内容が表示されます。よい README は、あなたの作品集の顔になります。

### AI プロジェクト向け README テンプレート

```markdown
# プロジェクト名

このプロジェクトが何をするかを1文で紹介する。

## 📋 プロジェクト概要

プロジェクトの背景、解決したい問題、使っている方法を2〜3文で詳しく説明する。

## ✨ 主な特徴

- 特徴1：XXX
- 特徴2：XXX
- 特徴3：XXX

## 🛠️ 技術スタック

- Python 3.11
- PyTorch 2.0
- その他使っているライブラリ

## 🚀 クイックスタート

### 環境構築

​```bash
git clone git@github.com:yourname/project.git
cd project
pip install -r requirements.txt
​```

### 実行

​```bash
python src/train.py
​```

## 📊 実験結果

| モデル | 精度 | 学習時間 |
|------|:-----:|:------:|
| SimpleCNN | 85.2% | 10 min |
| ResNet18 | 92.7% | 30 min |

## 📁 プロジェクト構成

​```
project/
├── data/              # データファイル
├── models/            # 学習済みモデル
├── src/
│   ├── model.py       # モデル定義
│   ├── train.py       # 学習スクリプト
│   └── utils.py       # ユーティリティ関数
├── requirements.txt
└── README.md
​```

## 📄 License

MIT
```

### 例：私たちのプロジェクトの README を更新する

```bash
# 上のテンプレートを使って README を書く（内容は簡易版）
cat > README.md << 'READMEEOF'
# AI 画像分類器

CNN を使って CIFAR-10 データセットを画像分類する入門プロジェクトです。

## 技術スタック

- Python 3.11
- PyTorch 2.0

## クイックスタート

```bash
git clone git@github.com:zhangsan/ai-image-classifier.git
cd ai-image-classifier
pip install -r requirements.txt
python src/train.py
```

## プロジェクト構成

```
ai-image-classifier/
├── data/              # データファイル（gitignore 対象）
├── models/            # モデルの重み（gitignore 対象）
├── src/
│   ├── model.py       # CNN モデル定義
│   ├── train.py       # 学習スクリプト
│   └── utils.py       # ユーティリティ関数
├── .gitignore
├── requirements.txt
└── README.md
```
READMEEOF

git add README.md
git commit -m "README を充実：プロジェクト説明と使い方を追加"
git push
```

---

## よくある問題

### push が拒否される（rejected）

```
! [rejected]        main -> main (fetch first)
```

これは、リモートリポジトリにローカルにない commit があることを意味します（別のPCで変更した、または同僚が新しいコードを push した、など）。解決方法は次のとおりです。

```bash
git pull          # 先にリモートの更新を取得する
git push          # そのあとで push する
```

### clone が遅い

日本国内や中国国内では、GitHub プロジェクトの clone が遅いことがあります。対処法はいくつかあります。

```bash
# 方法1：最新バージョンだけを clone する（完全な履歴は取らない）。かなり速くなる
git clone --depth 1 git@github.com:xxx/yyy.git

# 方法2：ミラーを使って高速化する
# github.com をミラーサイトに置き換える（具体的なミラーURLは、最新の利用可能なものを検索してください）
```

### 間違ったリポジトリに push してしまった

```bash
# 現在関連付けられているリモートリポジトリを確認する
git remote -v

# リモートリポジトリのURLを変更する
git remote set-url origin git@github.com:正しいユーザー名/正しいリポジトリ名.git
```

---

## まとめ

| コマンド | 用途 | いつ使うか |
|------|------|----------|
| `git remote add origin URL` | リモートリポジトリを関連付ける | 新しいプロジェクトを初めて push する前 |
| `git push` | ローカルの commit をリモートに送る | 機能が完成したあと |
| `git pull` | リモートの更新をローカルに取り込む | 作業を始める前 |
| `git clone URL` | リモートリポジトリをローカルにダウンロードする | プロジェクトを初めて取得するとき |

日常の流れ：**pull → コードを書く → add → commit → push**。とてもシンプルです。
