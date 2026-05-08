---
title: "3.4.1 可視化ロードマップ：スタイルより先にグラフを選ぶ"
sidebar_position: 16
description: "短い可視化ロードマップです。傾向、比較、分布、関係、レポートに合うグラフを選びます。"
keywords: [データ可視化, グラフ選択, matplotlib, seaborn, plotly, 可視化ロードマップ]
---

# 3.4.1 可視化ロードマップ：スタイルより先にグラフを選ぶ

可視化は飾りではありません。分析結果を、他の人がすぐ理解できる形に変える作業です。

## まずグラフ選択マップを見る

![データ可視化ロードマップ](/img/course/ch03-visualization-roadmap-ja.webp)

最初はこの判断で十分です。

| 見せたいこと | まず使うグラフ |
|---|---|
| 時間による変化 | 折れ線グラフ |
| カテゴリ比較 | 棒グラフ |
| 分布 | ヒストグラムまたは箱ひげ図 |
| 2つの数値の関係 | 散布図 |
| 相関行列 | ヒートマップ |

グラフの種類が合ってから、タイトル、軸、凡例、色、注釈を整えます。

## グラフを一度作る

`visual_first_loop.py` を作り、`pandas` と `matplotlib` をインストールしてから実行します。

```python
import pandas as pd
import matplotlib.pyplot as plt

sales = pd.DataFrame(
    {
        "month": ["2026-01", "2026-02", "2026-03", "2026-04"],
        "amount": [120, 180, 160, 220],
    }
)

ax = sales.plot(x="month", y="amount", marker="o", legend=False)
ax.set_title("Monthly sales")
ax.set_xlabel("Month")
ax.set_ylabel("Amount")
plt.tight_layout()
plt.savefig("sales_trend.png", dpi=150)

print("saved: sales_trend.png")
```

出力：

```text
saved: sales_trend.png
```

画像を開いて、1つだけ確認します。読者は3秒以内に傾向を理解できますか。

## この順番で学ぶ

| 順番 | 読む | 練習すること |
|---|---|---|
| 1 | [3.4.2 Matplotlib 基礎](./01-matplotlib.md) | Figure、Axes、折れ線/棒/散布図 |
| 2 | [3.4.3 Seaborn 統計可視化](./02-seaborn.md) | 探索用グラフを素早く作る |
| 3 | [3.4.5 可視化ベストプラクティス](./04-best-practices.md) | グラフ選択、ラベル、色、誤解を招く表現 |
| 4 | [3.4.4 Plotly インタラクティブ可視化](./03-plotly.md) | プロジェクトで必要なときだけ使う |

## 合格ライン

1つのデータセットから有用なグラフを4つ作り、それぞれのグラフを選んだ理由を説明できれば合格です。
