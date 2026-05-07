---
sidebar_position: 5
title: "0.2 環境準備"
description: "最初の週に必要な最小ツールだけを準備します。ブラウザ、Python、Git、1つのプロジェクトフォルダです。"
keywords: [AI環境準備, Python環境, VS Code, Git, Miniconda, クイックスタート]
---

# 0.2 環境準備

![AIコース最小セットアップキット](/img/course/intro-minimal-setup-kit-ja.png)

最初は少なく入れます。目標は、Python を動かし、Git で保存し、プロジェクトフォルダを1つ持つことです。

## 今入れるもの

| ツール | 用途 |
|---|---|
| ブラウザ | コース、Colab、GitHub、AIツール |
| VS Code | ファイル編集 |
| Python 3.11 | 例を動かす |
| Git | チェックポイント保存 |

Docker、CUDA、ベクトルDB、大きなフレームワークは後で入れます。

## 5分チェック

```bash
python --version
git --version
mkdir ai-learning-lab
cd ai-learning-lab
python -m venv .venv
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

失敗したら、いったん Colab で進め、第1章の後に戻ります。合格ラインは、フォルダに入り、Python を実行し、Git を初期化できることです。
