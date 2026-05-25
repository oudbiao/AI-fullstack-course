---
title: "1 開発者ツールの基礎"
description: "これからの AI プロジェクトに必要な、最小限のターミナル、Git、エディタ、Python 環境、Notebook ワークフローを作ります。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "ターミナル, コマンドライン, Git, VS Code, 開発環境, Python 環境構築"
---
![開発者ツールの基礎メインビジュアル](/img/course/ch01-tools-foundation-ja.webp)

この章の目的は 1 つだけです。**コードを作り、実行し、保存し、他の人が再実行できるように説明できること**。

## まず作業台を見る

![開発者ツール AI 作業台漫画ガイド](/img/course/ch01-ai-workstation-comic-ja.webp)

先に図を見てください。この章全体は次の流れです。

```text
ターミナル -> プロジェクトフォルダ -> Python 環境 -> エディタ/Notebook -> Git 履歴
```

今すべてのツールを完璧に覚える必要はありません。安定した作業台を 1 つ作り、後の AI プロジェクトで使い回します。

## 学習順序とタスクリスト

この順序を、学習ガイド兼タスクリストとして使います。

1. [1.1.1 ターミナルとコマンドライン](/ja/ch01-tools/ch01-terminal/01-why-cli/): `pwd`、`ls`、`cd` を実行し、短いコマンドログを残す。
2. [1.1.2 基本的なターミナル操作](/ja/ch01-tools/ch01-terminal/02-basic-operations/): ファイルを作成・移動・確認・削除し、画面か端末出力を残す。
3. [1.1.3 パッケージマネージャ](/ja/ch01-tools/ch01-terminal/03-package-managers/): 自分の環境でツールを入れる方法を確認し、バージョンメモを残す。
4. [1.2.1 Git の基礎](/ja/ch01-tools/ch02-git/01-git-basics/) と [1.2.2 Git の基本操作](/ja/ch01-tools/ch02-git/02-core-operations/): 最初のローカルプロジェクト記録を保存し、clean な Git commit を 1 回残す。
5. [1.3.1 Python 環境](/ja/ch01-tools/ch03-devenv/01-python-env/)、[1.3.2 VS Code](/ja/ch01-tools/ch03-devenv/02-vscode/)、[1.3.3 Jupyter](/ja/ch01-tools/ch03-devenv/03-jupyter/): 正しい環境で Python を動かし、コードを編集し、Notebook を restart-and-run する。
6. [1.4.1 ハンズオンワークショップ](/ja/ch01-tools/ch04-workshop/01-hands-on-tools-workshop/): ターミナル、Python、エディタ、Notebook、Git を再現可能な `ai-learning-lab` README にまとめる。

ワークショップは最後に置きます。統合のための実践なので、先に部品を学び、そのあと組み合わせます。

## 最初の実行ループ

練習フォルダで次を実行します。小さなプロジェクトを作り、実行し、説明を書き、Git に保存します。

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
. .venv/bin/activate
python -c "import sys; print(sys.executable)"
printf '.venv/\n__pycache__/\n' > .gitignore
printf 'print("AI 学習ラボの準備ができました")\n' > hello_ai.py
printf '# AI 学習ラボ\n\n環境を有効化：. .venv/bin/activate\n実行方法：python hello_ai.py\n' > README.md
python hello_ai.py
git init
git add .gitignore README.md hello_ai.py
git commit -m "init learning lab"
```

期待される出力：

```text
AI 学習ラボの準備ができました
```

失敗したら、エラーを消さないでください。コマンド、完全な出力、OS、Python バージョン、現在のディレクトリを残します。それも価値のあるプロジェクト証拠です。

Windows PowerShell では `. .venv/bin/activate` の代わりに `.venv\Scripts\Activate.ps1` を使います。手元の環境で `python3` を使う場合は、コマンドと README の `python` をすべて `python3` にそろえてください。

### この出力の読み方

- `AI 学習ラボの準備ができました` は、スクリプトがプロジェクトフォルダ内で動いたことを示します。
- `python -c "import sys; print(sys.executable)"` は、実際にどのインタプリタが動いているかを示します。
- Git commit は、プロジェクトをあとで保存・確認・再現できることを示します。
- どこかのコマンドが失敗した場合も、コマンドと完全なエラー出力は証拠であり、ノイズではありません。

## 深度ラダー

| レベル | 証明できること |
|---|---|
| 最低合格 | フォルダを作り、スクリプトを実行し、現在のディレクトリと Python インタプリタを説明できる。 |
| プロジェクト利用可 | 新しいターミナルから README どおりに再実行でき、`.venv/` が無視され、`git status` には意図した変更だけが出る。 |
| 深い確認 | PATH、作業ディレクトリ、shell、インタプリタ選択が、別のマシンで結果を変える理由を説明できる。 |

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
ワークスペース：端末、Git リポジトリ、エディタ、Python 環境、Notebook のすべてを確認済み
成果物: 短いコマンドログ、コミット履歴、スクリプト出力、または notebook セル結果
デバッグメモ: 1つのセットアップ問題と、その診断方法
失敗確認: パスの混乱、環境不一致、Git の状態、または依存関係不足
期待される成果：学習準備が整ったワークステーションの証拠パック
```

## よくある失敗

| 症状 | 最初に確認すること | よくある修正 |
|---|---|---|
| command not found | ツールが入っているか、PATH が有効か | ターミナルを開き直す、またはツールを入れ直す |
| Python import が失敗する | `python` と `pip` が同じ環境か | `python -m pip install ...` で入れる |
| ファイルが見つからない | 今いるディレクトリが正しいか | `pwd` と `ls` を実行し、プロジェクトフォルダへ移動する |
| Git commit が失敗する | 初期化、stage、ユーザー設定が済んでいるか | `git status` を見て、必要ならユーザー名とメールを設定する |
| README のコマンドが動かない | 必要な手順が README に全部あるか | 新しいターミナルで試し、README を直す |

## 通過チェック

次の 5 つに答えられたら、第 2 章へ進めます。

- 今のターミナルはどのディレクトリを使っていますか？
- スクリプトを実行している Python はどれですか？
- 前回の Git commit から何が変わりましたか？
- 新しいターミナルからプロジェクトを再実行するコマンドは何ですか？
- 最初のエラーと修正方法をどこに記録しましたか？

<details>
<summary>確認の考え方と解説</summary>

1. 今いる場所は `pwd` で確認できます。答えにはプロジェクトルートか、その中の作業フォルダかを含めます。
2. 実行中の Python は `which python`、`python --version`、または VS Code の選択中インタプリタで確認します。
3. 前回の commit からの差分は `git status --short` と `git diff` で説明します。
4. 再実行コマンドは例として `python3 src/workstation_check.py` のように、プロジェクトルートから動く形で書きます。
5. エラー記録は README、`reports/`、または学習ノートに残し、「症状 -> 原因 -> 修正」が読める状態にします。

</details>

目的はツールを完璧にすることではなく、この先の学習に使える安定した作業台を作ることです。
