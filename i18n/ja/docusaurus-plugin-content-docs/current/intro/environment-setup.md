---
sidebar_position: 5
title: "環境準備"
description: "AI フルスタックコースの最初に必要な最小限の道具を準備し、Python、Git、プロジェクトフォルダが動くことを確認します。"
keywords: [AI 環境構築, Python 環境, VS Code, Git, Miniconda, クイックスタート]
---

# 環境準備

![AI コース最小セットアップキット](/img/course/intro-minimal-setup-kit-ja.png)

**目標：** 最初の章に必要な道具だけを準備し、自分のパソコンで小さなプロジェクトを動かせることを確認します。

ローカル環境で詰まったら、いったん [Google Colab](https://colab.research.google.com) で学習を続け、後で戻って直して大丈夫です。環境トラブルは普通に起きます。AI が向いていないという意味ではありません。

## まず入れるものはこれだけ

| 道具 | 何か | 今なぜ必要か |
| --- | --- | --- |
| モダンブラウザ | Chrome、Edge、Safari、Firefox など | コース、Colab、GitHub、AI ツールを開く |
| VS Code | コードエディタ | コードを書き、プロジェクトファイルを見る |
| Python 3.11 | プログラミング言語 | コース例を実行する |
| Git | バージョン管理ツール | 学習チェックポイントを保存し、後でプロジェクトを公開する |
| Miniconda | Python 環境マネージャ | プロジェクトごとの依存関係を分ける |

第 1 章の前に、GPU ドライバ、CUDA、Docker、ベクトルデータベース、すべての AI フレームワークを先に入れる必要はありません。必要な章で追加すれば十分です。

## 5 分チェック

基本ツールを入れたら、ターミナルを開いて確認します。

```bash
python --version
git --version
```

macOS や Linux では、`python` が見つからない場合に `python3 --version` を使うことがあります。

次に、小さなプロジェクトを作ります。

```bash
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Windows PowerShell では、仮想環境の有効化コマンドが違います。

```powershell
mkdir ai-learning-lab
cd ai-learning-lab
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

最後に実行確認をします。

```bash
python -c "print('AI course environment is ready')"
git init
```

期待される出力はだいたい次の形です。

```text
AI course environment is ready
Initialized empty Git repository ...
```

これが見えれば、第 1 章に進めます。

## まず意味だけ知っておく言葉

| 用語 | シンプルな意味 |
| --- | --- |
| ターミナル | コマンドを入力して実行する場所 |
| エディタ | コードを書き、ファイルを整理する場所 |
| Python インタプリタ | Python コードを実行するプログラム |
| 仮想環境 | 1 つのプロジェクト用に依存関係を分ける小さな部屋 |
| パッケージ | `pip` や `conda` で入れる再利用可能なコード |
| リポジトリ | Git で履歴を管理するプロジェクトフォルダ |
| API key | オンライン AI サービスを呼び出すための秘密の鍵 |
| GPU | 深層学習を高速化するハードウェア。後で役立つが今は不要 |

## 後で入れればよいもの

| 後で使う道具 | 役立つタイミング |
| --- | --- |
| Jupyter Notebook | 第 3 章のデータ分析 |
| PyTorch | 第 6 章の深層学習 |
| Hugging Face `transformers` | 第 7 章と第 11 章 |
| OpenAI 互換 SDK | 第 8 章の LLM アプリ |
| Docker | 第 8 章のデプロイと再現性 |
| ベクトルデータベース | 第 8 章の RAG |
| GPU またはクラウド GPU | 第 6 章以降のモデル実験 |

章で必要になったときに入れるほうが、最初の負担が軽く、依存関係の衝突も起きにくくなります。

## うまくいかないとき

| 症状 | まず確認すること |
| --- | --- |
| `python` が見つからない | `python3` または `py -3.11` を試し、Python 3.11 が入っているか確認する |
| `git` が見つからない | Git をインストールし、ターミナルを開き直す |
| Windows で `source` が使えない | 上の PowerShell 用コマンドを使う |
| `pip install` がとても遅い | 地域の PyPI ミラーを使うか、いったん Colab で続ける |
| ローカル環境がつらい | 先に Colab で学び、第 1 章の後で戻って整える |

今の合格ラインはシンプルです。ターミナルを開き、プロジェクトフォルダへ入り、Python を実行し、Git を初期化できれば十分です。
