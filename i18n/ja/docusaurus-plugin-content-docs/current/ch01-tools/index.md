---
title: "1 開発者ツールの基礎"
sidebar_position: 0
description: "これからの AI プロジェクトに必要な、最小限のターミナル、Git、エディタ、Python 環境、Notebook ワークフローを作ります。"
keywords: [ターミナル, コマンドライン, Git, VS Code, 開発環境, Python 環境構築]
---

# 1 開発者ツールの基礎

![開発者ツールの基礎メインビジュアル](/img/course/ch01-tools-foundation-ja.png)

この章の目的は 1 つだけです。**コードを作り、実行し、保存し、他の人が再実行できるように説明できること**。

## 1.0.1 まず作業台を見る

![開発者ツール AI 作業台漫画ガイド](/img/course/ch01-ai-workstation-comic-ja.png)

先に図を見てください。この章全体は次の流れです。

```text
ターミナル -> プロジェクトフォルダ -> Python 環境 -> エディタ/Notebook -> Git 履歴
```

今すべてのツールを完璧に覚える必要はありません。安定した作業台を 1 つ作り、後の AI プロジェクトで使い回します。

## 1.0.2 学習順序とタスクリスト

この表を、学習ガイド兼タスクリストとして使います。

| ページ | 手を動かすこと | 残す証拠 |
|---|---|---|
| [1.1.1 ターミナルとコマンドライン](ch01-terminal/01-why-cli.md) | ターミナルを開き、`pwd`、`ls`、`cd` を実行する | 短いコマンドログ |
| [1.1.2 基本的なターミナル操作](ch01-terminal/02-basic-operations.md) | 練習フォルダでファイルを作成・移動・確認・削除する | フォルダのスクリーンショットまたは端末出力 |
| [1.1.3 パッケージマネージャ](ch01-terminal/03-package-managers.md) | 自分の環境でツールを入れる方法を確認する | ツールのバージョンメモ |
| [1.2.1 Git の基礎](ch02-git/01-git-basics.md) と [1.2.2 Git の基本操作](ch02-git/02-core-operations.md) | 最初のローカルプロジェクト記録を保存する | clean な Git commit 1 回 |
| [1.3.1 Python 環境](ch03-devenv/01-python-env.md) | 仮想環境を作り、その中で Python を実行する | Python バージョンと環境コマンド |
| [1.3.2 VS Code](ch03-devenv/02-vscode.md) と [1.3.3 Jupyter](ch03-devenv/03-jupyter.md) | エディタでコードを書き、Notebook で探索する | エディタ/Notebook の動作メモ |
| [1.4.1 ハンズオンワークショップ](ch04-workshop/01-hands-on-tools-workshop.md) | ターミナル、Python、エディタ、Notebook、Git をつなげる | 再現可能な `ai-learning-lab` README |

ワークショップは最後に置きます。統合のための実践なので、先に部品を学び、そのあと組み合わせます。

## 1.0.3 最初の実行ループ

練習フォルダで次を実行します。小さなプロジェクトを作り、実行し、説明を書き、Git に保存します。

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
printf 'print("AI learning lab is ready")\n' > hello_ai.py
printf '# AI Learning Lab\n\nRun with: python hello_ai.py\n' > README.md
python hello_ai.py
git init
git add README.md hello_ai.py
git commit -m "init learning lab"
```

期待される出力：

```text
AI learning lab is ready
```

失敗したら、エラーを消さないでください。コマンド、完全な出力、OS、Python バージョン、現在のディレクトリを残します。それも価値のあるプロジェクト証拠です。

## 1.0.4 よくある失敗

| 症状 | 最初に確認すること | よくある修正 |
|---|---|---|
| command not found | ツールが入っているか、PATH が有効か | ターミナルを開き直す、またはツールを入れ直す |
| Python import が失敗する | `python` と `pip` が同じ環境か | `python -m pip install ...` で入れる |
| ファイルが見つからない | 今いるディレクトリが正しいか | `pwd` と `ls` を実行し、プロジェクトフォルダへ移動する |
| Git commit が失敗する | 初期化、stage、ユーザー設定が済んでいるか | `git status` を見て、必要ならユーザー名とメールを設定する |
| README のコマンドが動かない | 必要な手順が README に全部あるか | 新しいターミナルで試し、README を直す |

## 1.0.5 通過チェック

次の 5 つに答えられたら、第 2 章へ進めます。

- 今のターミナルはどのディレクトリを使っていますか？
- スクリプトを実行している Python はどれですか？
- 前回の Git commit から何が変わりましたか？
- 新しいターミナルからプロジェクトを再実行するコマンドは何ですか？
- 最初のエラーと修正方法をどこに記録しましたか？

目的はツールを完璧にすることではなく、この先の学習に使える安定した作業台を作ることです。
