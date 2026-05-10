---
sidebar_position: 1
title: "0.2 環境準備"
description: "最初の週に必要な最小ツールだけを準備します。ブラウザ、Python、Git、1つのプロジェクトフォルダです。"
keywords: [AI環境準備, Python環境, VS Code, Git, Miniconda, クイックスタート]
---

# 0.2 環境準備

![AIコース最小セットアップキット](/img/course/intro-minimal-setup-kit-ja.webp)

最初は少なく入れます。目標は、1つのフォルダに入り、Python を動かし、Git で保存し、他の人が再実行できるだけの証拠を残すことです。

## 今入れるもの

| ツール | 用途 |
|---|---|
| ブラウザ | コース、Colab、GitHub、AIツール |
| VS Code | ファイル編集 |
| Python 3.11 | 例を動かす |
| Git | チェックポイント保存 |

Docker、CUDA、ベクトルDB、大きなフレームワークは後で入れます。早く入れすぎると、初学者のエラー原因が見つけにくくなります。

## Python コマンドを1つ選ぶ

環境によって Python の起動コマンドは違います。最初に動くものを選び、メモの中では同じコマンドを使い続けます。

| システム | 先に試す | 失敗したら |
|---|---|---|
| macOS / Linux | `python3 --version` | `python --version` |
| Windows PowerShell | `py -3.11 --version` | `python --version` |
| Colab | ローカル install 不要 | Notebook runtime を使う |

後のコマンドで `python` と書かれている場合は、自分の環境で動いたコマンドに置き換えてかまいません。

## 5分チェック

```bash
python3 --version
git --version
mkdir ai-learning-lab
cd ai-learning-lab
python3 -m venv .venv
source .venv/bin/activate
python -c "print('AI course environment is ready')"
git init
```

Windows PowerShell の有効化：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

このような出力が見えれば十分です。

```text
AI course environment is ready
Initialized empty Git repository ...
```

## チェックが失敗したら

| 症状 | まず行うこと | 残す証拠 |
|---|---|---|
| `python3` が見つからない | 上のコマンド表を試し、必要なら Python 3.11 を入れる | コマンドと完全なエラー |
| 仮想環境の有効化に失敗 | shell を確認する。zsh/bash は `source`、PowerShell は `Activate.ps1` | shell 名と有効化コマンド |
| `git` が見つからない | Git を入れ、terminal を開き直し、`git --version` を再実行 | version 出力またはエラー |
| permission error | system 保護フォルダではなく、自分のユーザーフォルダに project を置く | `pwd` の現在地 |

それでも失敗したら、いったん Colab で進め、第1章の後に戻ります。合格ラインは、フォルダに入り、Python を実行し、Git を初期化できることです。

## 経験者が確認すること

すでに環境がある場合も、完全には飛ばさないでください。次を説明できるか確認します。

- このコース project を動かす interpreter はどれか。
- 依存関係はどこに install されるか。
- 別の machine で環境を再作成する手順は何か。
- commit すべきファイルと、local に残すべきファイルは何か。

環境もコース成果物の一部です。自分の laptop でしか動かない project は、まだ完成ではありません。
