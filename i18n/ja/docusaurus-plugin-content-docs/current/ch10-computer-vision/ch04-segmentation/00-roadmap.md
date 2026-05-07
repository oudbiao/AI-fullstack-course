---
title: "10.4.1 Segmentation ロードマップ：Pixel-Level Regions"
sidebar_position: 0
description: "Image segmentation の短い実践ロードマップ：masks、semantic segmentation、instance segmentation、IoU、boundary failures を理解する。"
keywords: [image segmentation guide, semantic segmentation, instance segmentation, mask]
---

# 10.4.1 Segmentation ロードマップ：Pixel-Level Regions

Segmentation は detection より細かいです。box ではなく mask を出力し、どの pixels が class または instance に属するかを示します。

## 10.4.1.1 まず mask workflow を見る

![Image segmentation 章の学習順序図](/img/course/ch10-segmentation-chapter-flow-ja.png)

![Semantic segmentation mask example](/img/course/semantic-segmentation-mask-ja.png)

![Semantic segmentation IoU and boundary map](/img/course/ch10-semantic-segmentation-iou-boundary-map-ja.png)

この章の中心 object は mask です。よくある failure は boundary quality、small objects、occlusion、class confusion です。

## 10.4.1.2 Mask IoU check を動かす

この script は 2 つの小さな binary masks を比較します。

```python
truth = [
    [1, 1, 0],
    [1, 0, 0],
    [0, 0, 0],
]

pred = [
    [1, 0, 0],
    [1, 1, 0],
    [0, 0, 0],
]

intersection = 0
union = 0
for y in range(3):
    for x in range(3):
        intersection += truth[y][x] == 1 and pred[y][x] == 1
        union += truth[y][x] == 1 or pred[y][x] == 1

print("mask_iou:", round(intersection / union, 3))
```

出力：

```text
mask_iou: 0.5
```

Segmentation report では mask、metrics、boundary errors を示します。colored overlay だけで終わらせないでください。

## 10.4.1.3 この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Semantic segmentation | すべての pixel に 1 つの class を予測する |
| 2 | Instance segmentation | 同じ class の別 object を分ける |
| 3 | Segmentation practice | masks、IoU/Dice、boundary errors、failed samples を比較する |

## 10.4.1.4 合格ライン

mask を作成または inspect し、簡単な overlap metric を計算し、boundary または class-confusion failure を説明できれば、この章は合格です。
