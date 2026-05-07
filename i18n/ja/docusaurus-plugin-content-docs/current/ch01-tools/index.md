---
title: "1 開発者ツールの基礎"
sidebar_position: 0
description: "これからの AI プロジェクトに必要な、最小限のターミナル、Git、エディタ、Python 環境、Notebook ワークフローを作ります。"
keywords: [ターミナル, コマンドライン, Git, VS Code, 開発環境, Python 環境構築]
---

# 1 開発者ツールの基礎

![開発者ツールの基礎メインビジュアル](/img/course/ch01-tools-foundation-ja.png)

この章の目的は 1 つだけです。**コードを作り、実行し、保存し、他の人が再実行できるように説明できること**。この作業台が不安定だと、後の AI テーマは必要以上に難しく感じます。

## まず作業台を見る

![開発者ツール AI 作業台漫画ガイド](/img/course/ch01-ai-workstation-comic-ja.png)

図を 1 つの流れとして読んでください。

```text
ターミナル -> プロジェクトフォルダ -> Python 環境 -> エディタ/Notebook -> Git 履歴
```

すべてのコマンドを暗記する必要はありません。まず、小さく再現できる流れを安定させます。

## 段階目標

| 項目 | 目標 |
|---|---|
| 対象者 | 初学者、または開発環境が不安定な学習者 |
| 目安時間 | 8-12 時間 |
| 最小成果 | 実行できる `ai-learning-lab` フォルダ、1 つの Python ファイル、1 回の Git commit |
| ポートフォリオ成果 | README、環境メモ、スクリーンショット/ログ、分かりやすい Git 履歴 |

## 推奨学習順序

| 手順 | ページ | やること |
|---|---|---|
| 1.1 | [1.1.1 ターミナルとコマンドライン](ch01-terminal/01-why-cli.md) | ターミナルを開き、フォルダを移動し、ファイルを確認し、コマンドを実行する |
| 1.2 | [1.1.2 基本的なターミナル操作](ch01-terminal/02-basic-operations.md) | 練習フォルダの中で、ファイルを作成・移動・確認・削除する |
| 1.3 | [1.1.3 パッケージマネージャ](ch01-terminal/03-package-managers.md) | ツールのインストール方法と確認方法を理解する |
| 1.4 | [1.2.1 Git の基礎](ch02-git/01-git-basics.md) | Git で最初のプロジェクト記録を残す |
| 1.5 | [1.2.2 Git の基本操作](ch02-git/02-core-operations.md) | `status`、`add`、`commit`、`log`、`diff` を使う |
| 1.6 | [1.3.1 Python 環境](ch03-devenv/01-python-env.md) | 分離された Python 環境を作り、依存関係を正しく入れる |
| 1.7 | [1.3.2 VS Code](ch03-devenv/02-vscode.md) と [1.3.3 Jupyter](ch03-devenv/03-jupyter.md) | エディタでプロジェクトを書き、Notebook で探索する |
| 1.8 | [1.4.1 ハンズオンワークショップ](ch04-workshop/01-hands-on-tools-workshop.md) | すべてのツールを 1 つの再現可能なミニプロジェクトにつなげる |

ワークショップは最後に置きます。統合のための実践だからです。先に部品を学び、そのあと組み合わせます。

## この段階で必ず完了するタスク

| タスク | 成果物 | 完了チェック |
|---|---|---|
| ターミナルを安全に使う | コマンド練習ログ | `pwd`、`ls`、`cd` がどこを操作しているか説明できる |
| Python を実行する | `hello_ai.py` | `python hello_ai.py` が期待どおりに出力する |
| 環境を分離する | `.venv` または Conda 環境メモ | どの Python が有効か分かる |
| エディタを使う | VS Code などで開いたプロジェクト | 編集、実行、ターミナル出力の確認ができる |
| Git で保存する | 少なくとも 1 回のローカル commit | commit 後に `git status` が clean になる |
| ワークショップを完了する | README とログ付きの `ai-learning-lab` | 他の人が README を見て再実行できる |

## 最小実行実験

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

## よくある失敗

| 症状 | 最初に確認すること | よくある修正 |
|---|---|---|
| command not found | ツールが入っているか、PATH が有効か | ターミナルを開き直す、またはツールを入れ直す |
| Python import が失敗する | `python` と `pip` が同じ環境か | `python -m pip install ...` で入れる |
| ファイルが見つからない | 今いるディレクトリが正しいか | `pwd` と `ls` を実行し、プロジェクトフォルダへ移動する |
| Git commit が失敗する | 初期化、stage、ユーザー設定が済んでいるか | `git status` を見て、必要ならユーザー名とメールを設定する |
| README のコマンドが動かない | 必要な手順が README に全部あるか | 新しいターミナルで試し、README を直す |

## 段階の成果物

| 成果物 | 最小版 | ポートフォリオ版 |
|---|---|---|
| 学習リポジトリ | `ai-learning-lab` があり、Python ファイルを 1 つ実行できる | フォルダ、README、スクリーンショット/ログ、commit 履歴が整理されている |
| 環境メモ | Python バージョンとインストールコマンドを記録する | 仮想環境手順と依存関係ファイルを含める |
| コマンドログ | 5-10 個のよく使うコマンドを保存する | 目的、出力、失敗時の対応も書く |
| Git 記録 | 1 回のローカル commit がある | commit message で小さな進歩が分かる |
| README | `hello_ai.py` の実行方法を説明する | 目的、セットアップ、実行コマンド、出力例、次の一歩を書く |

## 段階通過基準

| レベル | 次へ進める条件 |
|---|---|
| 最小通過 | ターミナルを開き、Python を実行し、Git commit を 1 回できる |
| 推奨通過 | 仮想環境を作り、依存関係を入れ、README を分かりやすく書ける |
| ポートフォリオ通過 | ワークショップを完了し、他の人が再実行できる証拠を残せる |

## 段階通過質問

- 今のターミナルはどのディレクトリを使っていますか？
- スクリプトを実行している Python はどれですか？
- 前回の Git commit から何が変わりましたか？
- 新しいターミナルからプロジェクトを再実行するコマンドは何ですか？
- 最初に出会ったエラーと修正方法をどこに記録しましたか？

この章の後は1.2 節へ進みます。目的はツールを完璧にすることではなく、この先の学習に使える安定した作業台を作ることです。
