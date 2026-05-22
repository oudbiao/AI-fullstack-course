---
title: "0.2 環境準備"
description: "最初の週に必要な再現可能なAIエンジニアリング最小環境を準備します。ブラウザ、Python、Git、1つのプロジェクトフォルダです。"
sidebar:
  order: 1
head:
  - tag: meta
    attrs:
      name: keywords
      content: "AIエンジニアリング環境, AI環境準備, Python環境, VS Code, Git, クイックスタート"
---

# 0.2 環境準備

![AIコース最小セットアップキット](/img/course/intro-minimal-setup-kit-ja.webp)

最初は少なく入れます。目標は、1つのフォルダに入り、Python を動かし、Git で保存し、他の人が再実行できるだけの証拠を残すことです。

仕事としての AI プロジェクトでは、環境準備は雑用ではありません。自分の laptop から、レビュー担当者、チームメイト、デプロイ先へ移せることを示す最初の証拠です。

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
git status
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

`git status` は、repository の中にいることを示すはずです。まだ commit する必要はありません。まず、このフォルダが作業を追跡できる状態か確認します。

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

環境もコース成果物の一部です。自分の laptop でしか動かない project は、まだ完成ではありません。より強い project には、短いセットアップ説明、明確な Python version、依存関係の方針、基本 runtime が動くことを示すコマンドがあります。

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
マシン状態：OS、Python/Node のバージョン、エディタ、ターミナル、パッケージマネージャ
検証記録：実行したコマンド、表示されたバージョン、最初のスクリプト出力
デバッグメモ: インストールエラー、パス問題、権限問題、または環境不一致
復旧計画：次に進む前に再試行する正確なコマンドまたはドキュメントページ
期待される成果: 再現可能な project フォルダ、成功したコマンド出力、既知のフォールバック1件
```
