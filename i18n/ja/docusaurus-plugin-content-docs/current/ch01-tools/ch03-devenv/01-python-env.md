---
title: "1.3.1 Python環境管理"
sidebar_position: 1
description: "Miniconda で Python のバージョンと仮想環境を管理し、パッケージの衝突を根本から防ぐ"
---

# 1.3.1 Python環境管理

![Python 環境と依存関係の図](/img/course/ch01-python-env-stack-ja.png)

## この節の位置づけ

この節では、AI 学習で最もよくある「環境の混乱」と「依存関係の衝突」問題を解決します。なぜ各プロジェクトに独立した環境が必要なのかを理解し、Miniconda を使って Python 環境を作成・切り替え・書き出し・復元できるようになりましょう。

## 学習目標

- なぜ仮想環境が必要なのかを理解する（実際の失敗例を通して）
- Miniconda をインストールして設定する
- 仮想環境の作成・有効化・切り替え・削除を身につける
- conda と pip の違いを理解し、使い分けられるようになる
- 環境設定の書き出しと読み込みを学ぶ
- よくある環境問題を自力で調べて解決できるようになる

---

## なぜ仮想環境が必要なのか？

### 実際によくある失敗例

小明さんは 2 つの AI プロジェクトを進めています。

- **プロジェクト A**（画像分類）：`torch==1.13` が必要。1.13 にしか対応していない古いライブラリを使っているため
- **プロジェクト B**（大規模モデルアプリ）：`torch==2.1` が必要。最新の Flash Attention を使っているため

もし 2 つのプロジェクトの依存関係を同じ Python に入れてしまうと：

```bash
pip install torch==1.13    # プロジェクト A は動く
pip install torch==2.1     # プロジェクト B は動くが、torch は 2.1 に更新される
# そのあとでプロジェクト A を実行すると——エラー！ torch が 2.1 になっているから
```

これが**パッケージのバージョン衝突**です。1 つの Python 環境には、同じ名前のパッケージは 1 つのバージョンしか入れられません。

### 仮想環境はどう解決するのか？

仮想環境 = **独立して分離された Python インストール**です。プロジェクトごとに環境を分ければ、互いに影響しません。

```
プロジェクト A の環境: Python 3.10 + torch 1.13 + ...
プロジェクト B の環境: Python 3.11 + torch 2.1 + ...
```

プロジェクトを切り替えるときは、環境を切り替えるだけでOKです。お互いに干渉しません。

### たとえで理解する

仮想環境は、スマホの**複数ユーザー/作業スペース**のようなものです。各ユーザーが自分専用のアプリを入れられて、互いに干渉しません。たとえば「仕事用ユーザー」に Dingtalk を入れて、「個人用ユーザー」にゲームを入れる、といった具合に完全に分離できます。

### 持っておきたい環境のメンタルモデル

Python プロジェクトでエラーが出たとき、いきなり全部を再インストールしないでください。まず次の 4 つが同じプロジェクトを指しているか確認します。

```mermaid
flowchart LR
    A["プロジェクトフォルダ"] --> B["選択中の Python インタプリタ"]
    B --> C["pip がこの環境にパッケージを入れる"]
    C --> D["VS Code / Jupyter が同じインタプリタを使う"]
    D --> E["コードが依存関係を import できる"]
```

| 確認するもの | コマンドまたは見る場所 | 望ましい状態 |
|---|---|---|
| 現在のフォルダ | `pwd` | プロジェクトフォルダの中にいる |
| 現在の Python | `which python` | 目的の環境を指している |
| pip のインストール先 | `python -m pip --version` | pip が同じ Python 環境に属している |
| VS Code のインタプリタ | `Python: Select Interpreter` | 同じ環境が選ばれている |
| Jupyter Kernel | Notebook の kernel セレクタ | プロジェクト用の環境を使っている |

「インストールしたのに import できない」問題の多くは、このつながりのどこかが別の場所を指していることが原因です。

---

## Miniconda のインストール

### なぜ Miniconda を選ぶのか？

| ツール | 説明 | おすすめ度 |
|------|------|:---:|
| **Miniconda** | 軽量で、最小限の構成だけ入っている。必要なパッケージをあとから追加する | ⭐⭐⭐⭐⭐ |
| Anaconda | フルセットで、250 以上のパッケージが最初から入っている。3GB 以上使う | ⭐⭐⭐ |
| venv + pip | Python 標準機能で軽量。ただし機能は少なめ | ⭐⭐⭐ |

Miniconda は最適な選択です。軽量で、Python のバージョン管理もでき、仮想環境も作れます。AI コミュニティでも広く使われています。

### macOS でのインストール

```bash
# インストールスクリプトをダウンロード（Apple Silicon / M1/M2/M3）
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh

# Intel Mac の場合はこちら
# curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh

# インストールを実行
bash Miniconda3-latest-MacOSX-arm64.sh
```

インストール中は：
- ライセンス表示：`q` でスキップし、`yes` を入力して同意
- インストール先：Enter を押してデフォルトのまま
- 初期化するか：`yes` を入力

インストールが終わったら、**ターミナルを閉じてから開き直してください**。

### Ubuntu/Linux でのインストール

```bash
# ダウンロード
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# インストール
bash Miniconda3-latest-Linux-x86_64.sh

# 画面の案内に従い、最後に yes で初期化
```

ターミナルを閉じてから開き直してください。

### Windows でのインストール

1. インストーラーをダウンロードする：[Miniconda Windows インストーラー](https://docs.conda.io/en/latest/miniconda.html)
2. ダブルクリックして起動し、Next を進める
3. **"Add Miniconda3 to my PATH environment variable" にチェックを入れる**（PowerShell で使いやすくなります）
4. インストール後、PowerShell を再起動する

### インストールの確認

```bash
conda --version
# 例: conda 24.x.x と表示される

python --version
# 例: Python 3.12.x と表示される
```

バージョン番号が表示されれば、インストール成功です。

### 国内ミラーの設定（国内ユーザーには強く推奨）

デフォルトの conda の配布元は海外なので、ダウンロードが遅いことがあります。清華大学のミラーを設定しましょう。

```bash
# 清華ミラーを追加
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch
conda config --set show_channel_urls yes
```

あわせて pip の清華ミラーも設定します（まだ設定していない場合）：

```bash
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 仮想環境の操作

### 環境を作成する

```bash
# ai-basic という名前の環境を作成し、Python 3.11 を使う
conda create -n ai-basic python=3.11

# conda がインストールするパッケージを表示するので、y を入力して確定
```

`-n ai-basic` は環境の名前です。好きな名前を付けられますが、プロジェクト名や用途に合わせるのがおすすめです。

### 環境を有効化する

```bash
conda activate ai-basic
```

有効化すると、ターミナルのプロンプトの前に環境名が表示されます。

```
(ai-basic) zhangsan@MacBook ~ $
```

これは今 `ai-basic` 環境にいるという意味です。この環境にインストールしたパッケージは、この環境だけに属します。

### 環境の中でパッケージをインストールする

```bash
# 現在の環境を確認
conda info --envs
# * が付いているものが、今有効な環境

# pip でパッケージをインストール（ほとんどの場合はこちらがおすすめ）
pip install numpy pandas matplotlib

# conda でパッケージをインストール（特殊なパッケージではこちらがよい場合がある）
conda install scipy

# 今の環境に入っているパッケージを確認
pip list
# または
conda list
```

### 例：プロジェクトごとに別々の環境を作る

```bash
# プロジェクト A：従来の機械学習
conda create -n ml-project python=3.11
conda activate ml-project
pip install numpy pandas scikit-learn matplotlib seaborn jupyter

# プロジェクト B：深層学習
conda create -n dl-project python=3.11
conda activate dl-project
pip install torch torchvision numpy matplotlib tensorboard

# プロジェクト C：大規模モデルアプリ
conda create -n llm-project python=3.11
conda activate llm-project
pip install openai langchain chromadb fastapi
```

3 つのプロジェクト、それぞれ独立した環境です。互いに干渉しません。

### 環境を切り替える

```bash
# ml-project 環境に切り替える
conda activate ml-project

# dl-project 環境に切り替える
conda activate dl-project

# 現在の環境を抜ける（base 環境に戻る）
conda deactivate
```

### すべての環境を表示する

```bash
conda env list
# または
conda info --envs
```

出力例：

```
# conda environments:
#
base                     /Users/zhangsan/miniconda3
ai-basic                 /Users/zhangsan/miniconda3/envs/ai-basic
ml-project            *  /Users/zhangsan/miniconda3/envs/ml-project
dl-project               /Users/zhangsan/miniconda3/envs/dl-project
llm-project              /Users/zhangsan/miniconda3/envs/llm-project
```

`*` は現在有効な環境を表します。

### 環境を削除する

```bash
# もう不要な環境を削除
conda env remove -n ai-basic

# 削除されたことを確認
conda env list
```

---

## conda install と pip install の違い

これは初心者が最もよく聞く質問です。基本の考え方は次の通りです。

| 状況 | 使うもの | 理由 |
|------|-------|------|
| ほとんどの Python パッケージ | `pip install` | pip のパッケージが最も多く、更新も速い |
| CUDA 関連のパッケージ | `conda install` | conda が CUDA の依存関係を自動で処理しやすい |
| システムレベルのライブラリ（例: MKL） | `conda install` | pip ではシステムレベルのライブラリを入れられない |
| どちらを使うか迷うとき | まず `pip install` を試す | pip のほうが汎用的 |

:::warning 重要なルール
同じ環境の中では、**できるだけ conda install と pip install を混ぜて同じパッケージを入れない**でください。たとえば pip で numpy を入れたなら、conda で同じ numpy をもう一度入れないようにしましょう。混在するとバージョンが乱れることがあります。

おすすめの方法：conda 環境の中では、Python パッケージはまず pip で入れる。
:::

---

## 環境の書き出しと読み込み

### シーン：プロジェクト環境を共有する

プロジェクトが完成したあと、同僚や未来の自分がすぐ同じ環境を作れるようにしたいとします。

#### 方法1：pip freeze（もっともよく使う）

```bash
# 現在の環境のパッケージ一覧を requirements.txt に書き出す
pip freeze > requirements.txt
```

`requirements.txt` はこんな内容になります。

```
numpy==1.26.4
pandas==2.2.0
scikit-learn==1.4.0
matplotlib==3.8.2
torch==2.1.2
```

受け取った人は、1 行のコマンドで復元できます。

```bash
# 新しい環境を作成
conda create -n restored-env python=3.11
conda activate restored-env

# すべての依存関係をインストール
pip install -r requirements.txt
```

#### 方法2：conda env export

```bash
# 完全な環境を書き出す（conda と pip で入れたパッケージを含む）
conda env export > environment.yml
```

復元：

```bash
conda env create -f environment.yml
```

#### どれを使うべき？

| ファイル | 適した場面 | メリット | デメリット |
|------|---------|------|------|
| `requirements.txt` | ほとんどのプロジェクト | シンプル、汎用的、クロスプラットフォーム | Python のバージョン情報は含まれない |
| `environment.yml` | conda 特有のパッケージを含むプロジェクト | 完全な情報を含む、Python バージョンも入る | 環境によって差が出る場合がある |

**おすすめ：** 各プロジェクトに `requirements.txt` を置くのが、Python コミュニティの標準的なやり方です。

---

## よくある問題の対処

### 問題1：`conda activate` が効かない

```
CommandNotFoundError: Your shell has not been properly configured to use 'conda activate'.
```

解決方法：

```bash
# conda を初期化する（使っている shell に合わせる）
conda init zsh     # macOS のデフォルト
conda init bash    # Linux のデフォルト

# そのあとターミナルを再起動
```

### 問題2：`command not found: python`

Miniconda を入れたのに、`python` と入力すると見つからないと言われる場合です。

```bash
# conda 環境が有効か確認
conda activate base

# それでもだめなら PATH を確認
which python
echo $PATH
```

### 問題3：パッケージのインストールがタイムアウトする

```
pip install torch
# 長時間止まる、または timeout エラーになる
```

解決方法：国内ミラーが設定されているか確認するか、手動で指定します。

```bash
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 問題4：バージョン衝突

```
ERROR: pip's dependency resolver found conflicts
```

対処の考え方：

```bash
# 方法1: 新しい環境を作って、1つずつ入れる
conda create -n fresh python=3.11
conda activate fresh
pip install パッケージA
pip install パッケージB  # 衝突があれば、どこが問題か表示される

# 方法2: 衝突しているパッケージを互換性のあるバージョンに下げる
pip install "パッケージA>=1.0,<2.0"
```

### 問題5：パッケージを入れたのに import でエラーになる

```python
import torch
# ModuleNotFoundError: No module named 'torch'
```

もっともよくある原因は、**インストールした環境と、コードを実行している環境が違う**ことです。

```bash
# 現在の環境を確認
conda info --envs   # * が付いているものを見る

# どの環境に入っているか確認
conda activate torch を入れたつもりの環境
pip list | grep torch

# Python のパスを確認
which python
# conda 環境のディレクトリを指しているはず
```

---

## 実践練習：最初の学習環境を作る

```bash
# 1. このコース専用の環境を作成
conda create -n ai-course python=3.11
conda activate ai-course

# 2. 第 1 ステーションで必要な基本パッケージをインストール
pip install requests beautifulsoup4 fastapi uvicorn

# 3. 第 2 ステーションで必要なデータ分析パッケージをインストール
pip install numpy pandas matplotlib seaborn jupyter

# 4. インストールを確認
python -c "
import numpy as np
import pandas as pd
print(f'NumPy のバージョン: {np.__version__}')
print(f'Pandas のバージョン: {pd.__version__}')
print('✅ 環境の構築に成功しました！')
"

# 5. 環境設定を書き出す
pip freeze > requirements.txt
cat requirements.txt

# 6. 環境一覧を確認
conda env list
```

最後に `✅ 環境の構築に成功しました！` と表示されれば、Python 環境の準備は完了です。

---

## コマンド早見表

| コマンド | 用途 |
|------|------|
| `conda create -n 名前 python=3.11` | 新しい環境を作成する |
| `conda activate 名前` | 環境を有効化する |
| `conda deactivate` | 現在の環境を抜ける |
| `conda env list` | すべての環境を表示する |
| `conda env remove -n 名前` | 環境を削除する |
| `pip install パッケージ名` | Python パッケージをインストールする |
| `pip list` | インストール済みパッケージを表示する |
| `pip freeze > requirements.txt` | 依存関係リストを書き出す |
| `pip install -r requirements.txt` | ファイルから依存関係をインストールする |
