---
title: "A.8 学習が止まったときのレスキュー"
description: "環境確認、最小再現、デバッグ順序、質問の仕方で、学習とコードの詰まりを切り分ける。"
sidebar:
  order: 5
---
![学習の詰まりを切り分けるマップ](/img/course/appendix-troubleshooting-rescue-map-ja.webp)

![最小再現と質問のフローチャート](/img/course/appendix-debug-mre-help-flow-ja.webp)

詰まったら、「自分には無理」ではなく「どこで失敗しているかを探せる」に変えます。

## まず問題を分類する

| 症状 | ありそうな問題 | 最初の動き |
|---|---|---|
| `ModuleNotFoundError` | 環境違い、または依存関係不足 | Python と `pip` の場所を確認 |
| ファイルが見つからない | 作業ディレクトリや相対パスの問題 | `Path.cwd()` を出力 |
| コードは動くが結果が変 | 入力、ラベル、評価指標の問題 | サンプルと中間値を出力 |
| 学習が改善しない | データ、loss、学習率、ラベル形式の問題 | 小さなデータに過学習できるか試す |
| GPU メモリ不足 | batch、入力、モデルが大きすぎる | まず batch size を下げる |
| プロジェクトが大きすぎる | 最小閉ループがない | 1 入力、1 処理、1 出力にする |

## まず実行する確認

```bash
python --version
which python
pip --version
pip list
pwd
ls
```

NVIDIA GPU を使う場合:

```bash
nvidia-smi
```

パス問題はこう確認します。

```python
from pathlib import Path

print(Path.cwd())
print(Path("data").exists())
```

期待される出力：

現在のフォルダは環境によって変わりますが、形は次のようになります。

```text
/your/current/project
False
```

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
症状：正確なエラーメッセージ、コマンド、入力、環境
最小再現：まだ失敗する最小のコードまたはコマンド
仮説：依存関係、パス、データ、API、モデル、またはブラウザ/runtime の問題
次の確認：多くの変更をする前に、1つのコマンドかログを確認する
期待される成果：再現可能なバグメモと、テスト済みの修正または代替策
```

## この順番でデバッグする

1. 最初の 2 件の入力とラベルを出す。
2. shape、長さ、値の範囲を出す。
3. モデルに入る前の中間結果を 1 つ出す。
4. 評価指標を計算する前のモデル出力を 1 つ出す。
5. モデルやパラメータを変えるのは最後にする。

最小確認例:

```python
texts = ["refund request", "invoice copy", "shipping delay"]
labels = ["support", "billing", "support"]

print("samples:", len(texts))
print("first texts:", texts[:2])
print("first labels:", labels[:2])
print("label set:", sorted(set(labels)))
```

期待される出力：

```text
samples: 3
first texts: ['refund request', 'invoice copy']
first labels: ['support', 'billing']
label set: ['billing', 'support']
```

## 助けを求めるときの形

```text
やっていること:
期待した結果:
実際に起きたこと:
エラー全文の最後 20 行:
すでに試したこと:
最小再現コード:
```

## 最小再現の習慣

プロジェクトが複雑すぎるときは、まず動くところまで縮めます。

```python
def predict(x):
    return x * 2

data = [1, 2, 3]
preds = [predict(x) for x in data]
print(preds)
```

期待される出力：

```text
[2, 4, 6]
```

そこへ実際の処理を 1 層ずつ戻します。壊れた層が、調べるべき層です。

## 止まるか、続けるか

| 状況 | よい行動 |
|---|---|
| 30 分ほどランダムに直している | 止まって仮説を書く |
| コピーしているコマンドの意味を説明できない | 止まって環境を確認する |
| 1-2 個の明確な仮説がある | テストを続ける |
| 次に見るべき結果がわかっている | 進める |
