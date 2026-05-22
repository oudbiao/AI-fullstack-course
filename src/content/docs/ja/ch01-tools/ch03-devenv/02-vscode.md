---
title: "1.3.2 VS Code の設定"
description: "VS Code を使いやすい AI 開発ツールに設定する"
sidebar:
  order: 2
---
![VS Code プロジェクト作業フロー図](/img/course/ch01-vscode-workspace-flow-ja.webp)

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
環境：Python/Node/エディタ/Notebook のバージョンと選択したインタプリタ／カーネル
検証記録：setup が動作することを示す 1 つの command または notebook cell
プロジェクトフォルダ：依存関係、スクリプト、Notebook を置く場所
失敗確認: 間違ったインタプリタ、パッケージ不足、古いカーネル、またはエディタのパス不一致
期待される成果: セットアップのスクリーンショットまたはターミナル出力と1件のフォールバックメモ
```

## この節の位置づけ

この節では、VS Code を Python と AI 学習に向いた開発ツールに設定します。エディタのインストール、拡張機能の設定、内蔵ターミナル、よく使うショートカットの設定まで進め、これからのコード演習に安定して使いやすい作業環境を作ります。

## 学習目標

- VS Code をインストールして日本語化する
- Python 開発に必要な拡張機能を入れる
- VS Code の内蔵ターミナルの使い方を学ぶ
- よく使う 10 個のショートカットを覚える
- AI 補助コーディングツールを知る

---

## なぜ VS Code を選ぶのか？

| エディタ | 長所 | 短所 |
|-------|------|------|
| **VS Code** | 無料、軽量、拡張が豊富、AI サポートが強い | 大規模プロジェクトでは PyCharm ほど賢くない場合がある |
| PyCharm | Python サポートが最強、リファクタリングが便利 | Community 版は無料だが機能が少ない、Professional 版は有料 |
| Vim/NeoVim | とても高速、玄人向け | 学習曲線が急 |

VS Code は今、世界で最も使われているコードエディタの 1 つで、Python と AI 開発のサポートがとても充実しています。初心者にとって、かなりおすすめの選択です。

---

## VS Code をインストールする

### macOS

```bash
# Homebrew でインストール（おすすめ）
brew install --cask visual-studio-code

# または公式サイトからダウンロード：https://code.visualstudio.com
```

インストールが終わったら、コマンドラインから起動できるように設定します。

1. VS Code を開く
2. `Cmd + Shift + P` を押して、"shell command" と入力する
3. **Shell Command: Install 'code' command in PATH** を選ぶ

これで、ターミナルから `code` コマンドを使ってファイルやフォルダを開けるようになります。

```bash
code .                  # VS Code で現在のフォルダを開く
code ~/projects         # 指定したフォルダを開く
code hello.py           # 指定したファイルを開く
```

### Windows

```powershell
# winget でインストール
winget install Microsoft.VisualStudioCode

# または公式サイトからダウンロード：https://code.visualstudio.com
```

インストール時に **"Add to PATH"** にチェックを入れておくと、ターミナルで `code` コマンドを使えます。

### Ubuntu

```bash
# 方法1：snap を使う（おすすめ）
sudo snap install code --classic

# 方法2：apt を使う
sudo apt install software-properties-common apt-transport-https wget
wget -q https://packages.microsoft.com/keys/microsoft.asc -O- | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://packages.microsoft.com/repos/vscode stable main"
sudo apt update
sudo apt install code
```

---

## 表示言語の設定

1. VS Code を開く
2. `Ctrl + Shift + X`（macOS では `Cmd + Shift + X`）を押して、拡張機能パネルを開く
3. **Japanese Language Pack** を検索する
4. **Install** をクリックしてインストールする
5. VS Code を再起動すると、画面が日本語になります

---

## 必須の拡張機能を入れる

拡張機能パネル（左側の四角いアイコン、または `Ctrl/Cmd + Shift + X`）を開いて、次の拡張機能を検索してインストールします。

### 必須拡張

| 拡張名 | 役割 | 検索キーワード |
|-------|------|----------|
| **Python** | Python の文法サポート、デバッグ、実行 | `ms-python.python` |
| **Pylance** | Python の賢い補完、型チェック | `ms-python.vscode-pylance` |
| **Jupyter** | VS Code で Notebook を実行する | `ms-toolsai.jupyter` |
| **GitLens** | Git 機能を強化し、誰がどの行を変更したか見やすくする | `eamodio.gitlens` |
| **Black Formatter** | Python コードの整形をそろえる | `ms-python.black-formatter` |

### おすすめ拡張

| 拡張名 | 役割 |
|-------|------|
| **autoDocstring** | Python 関数の docstring を自動生成する |
| **Ruff** | Python コードの高速チェックと import 整理を行う |
| **indent-rainbow** | インデントの階層を色で見分けやすくする |
| **Error Lens** | エラー情報をコード行の末尾に直接表示する |
| **Material Icon Theme** | ファイルアイコンを見やすくする |

---

## Python インタープリタを設定する

Python 拡張を入れたら、VS Code にどの Python 環境を使うか伝える必要があります。

1. `Ctrl/Cmd + Shift + P` を押してコマンドパネルを開く
2. **Python: Select Interpreter** と入力する
3. 以前作成した conda 環境（たとえば `ai-course`）を選ぶ

次のような候補一覧が表示されます。

```
Python 3.11.7 ('ai-course')    ~/miniconda3/envs/ai-course/bin/python
Python 3.12.1 ('base')         ~/miniconda3/bin/python
```

`ai-course` の方を選びます。

:::tip[自動検出]
VS Code の Python 拡張は、システム上のすべての Python 環境（conda や venv 環境を含む）を自動で検出します。もし使いたい環境が見つからない場合は、先にターミナルでその環境に `conda activate` してから、ターミナルで `code .` を入力して VS Code を開いてみてください。
:::
---

## 内蔵ターミナルを使う

VS Code にはターミナルが内蔵されているので、別のターミナルウィンドウを開く必要はありません。

### ターミナルを開く

```
ショートカット：Ctrl + `（キーボード左上、ESC の下にあるキー）
```

またはメニューから：**ターミナル → 新しいターミナル**

### 例：VS Code で開発の流れを一通りやってみる

```bash
# 1. ターミナルで環境を有効化する
conda activate ai-course

# 2. プロジェクトフォルダを作成する
mkdir my-first-project
cd my-first-project

# 3. VS Code でこのフォルダを開く（新しいウィンドウで開きます）
code .
```

そのあと、VS Code で次の操作をします。

1. 左側のファイルエクスプローラーで、新規ファイルアイコンをクリックして `hello.py` を作成する
2. 次のコードを書く

```python
name = input("あなたの名前は？")
print(f"こんにちは、{name}！AI の世界へようこそ 🤖")
```

3. 右上の **▶ 実行** ボタンをクリックする（または `Ctrl/Cmd + Shift + P` → "Run Python File"）
4. ターミナルに出る出力を見る

### ターミナルのコツ

- **複数ターミナル**：ターミナルパネル右上の `+` をクリックすると、複数のターミナルを開けます
- **分割表示**：左右に分割して、片方でコードを書き、もう片方でターミナルを見ることができます
- **ターミナルの種類**：bash、zsh、PowerShell など、いろいろな shell を選べます

---

## よく使うショートカット

全部覚える必要はありません。まずは最初の 5 個を覚えて、ほかは必要になったら確認しましょう。

### 基本操作

| 操作 | Windows/Linux | macOS |
|------|:---:|:---:|
| コマンドパネル（最重要！） | `Ctrl + Shift + P` | `Cmd + Shift + P` |
| ファイルを素早く開く | `Ctrl + P` | `Cmd + P` |
| ターミナルを開く/閉じる | `` Ctrl + ` `` | `` Ctrl + ` `` |
| 保存 | `Ctrl + S` | `Cmd + S` |
| 元に戻す | `Ctrl + Z` | `Cmd + Z` |

### コード編集

| 操作 | Windows/Linux | macOS |
|------|:---:|:---:|
| 現在の行をコピー | `Shift + Alt + ↓` | `Shift + Option + ↓` |
| 現在の行を移動 | `Alt + ↑/↓` | `Option + ↑/↓` |
| 現在の行を削除 | `Ctrl + Shift + K` | `Cmd + Shift + K` |
| 複数カーソル編集 | `Alt + クリック` | `Option + クリック` |
| コードコメント | `Ctrl + /` | `Cmd + /` |
| コード整形 | `Shift + Alt + F` | `Shift + Option + F` |

### 検索と移動

| 操作 | Windows/Linux | macOS |
|------|:---:|:---:|
| 全体検索 | `Ctrl + Shift + F` | `Cmd + Shift + F` |
| ファイル内検索 | `Ctrl + F` | `Cmd + F` |
| 置換 | `Ctrl + H` | `Cmd + Option + F` |
| 指定行へ移動 | `Ctrl + G` | `Ctrl + G` |

### 例：複数カーソル編集の強さ

5 個の変数名を `data1`、`data2`... から `dataset1`、`dataset2`... に変えたいとします。

```python
data1 = load("file1.csv")
data2 = load("file2.csv")
data3 = load("file3.csv")
data4 = load("file4.csv")
data5 = load("file5.csv")
```

操作：
1. 最初の `data` を選ぶ
2. `Ctrl/Cmd + D` を 5 回続けて押し、すべての `data` を順番に選ぶ
3. `dataset` と入力する

5 か所が同時に書き換わり、2 秒で終わります。

---

## AI 補助コーディングツール

今は VS Code でコードを書くときに助けてくれる AI ツールがたくさんあります。AI コースの学習者として、ぜひ知っておきましょう。

### GitHub Copilot

- 入力中にコードを自動補完する
- `Tab` を押すと提案を受け入れる
- 学生は無料で使える（GitHub Student Pack 経由）
- 拡張機能の検索キーワード：`GitHub.copilot`

### Codeium

- 無料の AI コード補完ツール
- Copilot に似た機能があり、個人ユーザーは完全無料
- 拡張機能の検索キーワード：`Codeium.codeium`

### 使い方のおすすめ

:::caution[学習者へのアドバイス]
学習段階では、**AI のコード補完に頼りすぎない** でください。これは計算機のようなものです。まだ暗算を学んでいないのに計算機だけ使っていると、数学はなかなか身につきません。

おすすめ：
- 最初の 2 段階（Python 基礎）：AI 補完は **オフ** にして、自分で書く
- 第 4 ステップ以降：AI 補完を使ってもよいが、生成されたコードの各行を **理解する**
- プロジェクトを作るとき：自由に使って効率を上げる
:::
---

## おすすめの VS Code 設定

`Ctrl/Cmd + ,` を押して設定を開き、次の項目を検索して変更します。

| 設定項目 | 推奨値 | 理由 |
|-------|-------|------|
| Auto Save | `afterDelay` | 自動保存できるので、`Ctrl+S` を忘れても安心 |
| Font Size | `14` または `15` | コードの文字が少し大きくなり、見やすい |
| Tab Size | `4` | Python の標準インデント |
| Word Wrap | `on` | 長い行を自動で折り返す |
| Minimap | `off` | 右側のミニマップを消して、画面を広く使う |

または、`settings.json` を直接編集します（`Ctrl/Cmd + Shift + P` → "Open Settings JSON"）：

```json
{
    "files.autoSave": "afterDelay",
    "editor.fontSize": 14,
    "editor.tabSize": 4,
    "editor.wordWrap": "on",
    "editor.minimap.enabled": false,
    "python.terminal.activateEnvironment": true,
    "editor.formatOnSave": true,
    "[python]": {
        "editor.defaultFormatter": "ms-python.black-formatter"
    },
    "python.analysis.typeCheckingMode": "basic"
}
```

:::tip[古い Python フォーマット設定を使わない理由]
古いチュートリアルでは `"python.formatting.provider": "black"` と書かれていることがありますが、現在の VS Code Python ツールではフォーマッタ拡張を使う形が推奨されます。保存時に整形されない場合は、まず **Black Formatter** が入っているか、次に選択中のインタープリタがプロジェクト環境かを確認しましょう。
:::
---

## 実践練習

1. **VS Code** と必須拡張（Python、Pylance、Jupyter、GitLens）をインストールする
2. **プロジェクトを作成** して VS Code で開く

```bash
mkdir vscode-practice && cd vscode-practice && code .
```

3. **`practice.py` を新規作成** して、次のコードを書く

```python
# VS Code のショートカットを練習する
fruits = ["apple", "banana", "cherry", "date", "elderberry"]

for i, fruit in enumerate(fruits):
    print(f"{i + 1}. {fruit}")

# 果物名の平均文字数を計算する
avg_len = sum(len(f) for f in fruits) / len(fruits)
print(f"\n平均の名前の長さ: {avg_len:.1f} 文字")
```

4. **コードを実行** する（右上の ▶ ボタンをクリック）
5. **ショートカットを試す**：
- `Ctrl/Cmd + /` で最後の 2 行をコメントアウトする
- `Alt + ↑/↓` で 1 行を移動する
- `Ctrl/Cmd + D` で単語を複数選択する
- `Ctrl/Cmd + Shift + F` で "fruit" を全体検索する

<details>
<summary>プロジェクト参考とレビュー観点</summary>

1. `code .` は現在のプロジェクトフォルダ全体を開くコマンドです。単一ファイルだけを開くのではありません。
2. 右下やコマンドパレットで選んだ Python インタプリタは、このコース用の環境であるべきです。
3. `practice.py` は 5 個の果物に番号を振って出力し、平均文字数も表示します。
4. 右上の実行ボタンが期待通り動かないときは、VS Code 内の `python --version` と外部ターミナルの環境を比較します。
5. ショートカット練習の目的は、素早く編集・検索・整理できることを確認することです。

</details>
