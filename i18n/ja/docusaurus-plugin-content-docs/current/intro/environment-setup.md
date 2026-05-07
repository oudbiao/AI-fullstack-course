---
sidebar_position: 5
title: "環境準備"
description: "AIフルスタックコースの最初の週に必要な最小ツールを準備し、Python、Git、プロジェクトフォルダを小さな実行確認で検証します。"
keywords: [AI環境準備, Python環境, VS Code, Git, Miniconda, クイックスタート]
---

# 環境準備

![AIコース最小セットアップキット](/img/course/intro-minimal-setup-kit-ja.png)

**目標：**最初の週に必要なものだけを入れ、Python が動き、Git でコードを保存できることを確認します。

環境構築で 20 分以上止まるなら、いったん [Google Colab](https://colab.research.google.com) で進めましょう。環境エラーは普通のエンジニアリング作業です。

## 1. 先に入れるもの

| 入れるもの | 今必要な理由 |
|---|---|
| モダンブラウザ | コース、Colab、GitHub、AIツールを開く |
| VS Code | コード編集とフォルダ確認 |
| Python 3.11 | 序盤の例を動かす |
| Git | プロジェクトのチェックポイントを保存する |
| Miniconda または `venv` | プロジェクトごとに依存関係を分ける |

GPUドライバ、CUDA、Docker、ベクトルDB、大きなAIフレームワークはまだ不要です。使う章で入れます。

## 2. 5分チェック

バージョンを確認します。

```bash
python --version
git --version
```

macOS や Linux で `python` が見つからない場合：

```bash
python3 --version
```

最初のプロジェクトフォルダを作ります。

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -c "print('AI course environment is ready')"
git init
```

Windows PowerShell では次の有効化手順を使います。

```powershell
mkdir ai-learning-lab
cd ai-learning-lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -c "print('AI course environment is ready')"
git init
```

このような出力が見えれば十分です。

```text
AI course environment is ready
Initialized empty Git repository ...
```

## 3. まず知っておく言葉

| 用語 | 意味 |
|---|---|
| ターミナル | コマンドを入力する場所 |
| インタープリタ | Python を実行するプログラム |
| 仮想環境 | 1つのプロジェクト専用の依存関係スペース |
| パッケージ | `pip` や `conda` で入れる再利用コード |
| リポジトリ | Git が追跡するプロジェクトフォルダ |
| API Key | オンライン AI サービスを呼ぶための秘密の鍵 |

## 4. 失敗したら

| 症状 | 最初に試すこと |
|---|---|
| `python` が見つからない | `python3` または `py -3.11` を試し、Python 3.11 を入れ直す |
| `git` が見つからない | Git を入れてターミナルを開き直す |
| Windows で `source` が失敗する | 上の PowerShell コマンドを使う |
| `pip install` が遅い | いったん Colab を使うか、地域ミラーを使う |
| 複雑に感じる | Colab で進め、第1章後に戻る |

合格ラインは、フォルダに入り、Python を実行し、Git を初期化できることです。
