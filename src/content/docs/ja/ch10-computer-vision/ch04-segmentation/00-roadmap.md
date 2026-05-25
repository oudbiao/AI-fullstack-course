---
title: "10.4.1 画像分割ロードマップ：ピクセル単位の領域"
description: "Image segmentation の短い実践ロードマップ：masks、semantic segmentation、instance segmentation、IoU、boundary failures を理解する。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "image segmentation guide, semantic segmentation, instance segmentation, mask"
---
Segmentation は detection より細かいです。box ではなく mask を出力し、どの pixels が class または instance に属するかを示します。

## まず mask workflow を見る

![Image segmentation 章の学習順序図](/img/course/ch10-segmentation-chapter-flow-ja.webp)

![Semantic segmentation の mask 例](/img/course/semantic-segmentation-mask-ja.webp)

![Semantic segmentation の IoU と boundary 対応図](/img/course/ch10-semantic-segmentation-iou-boundary-map-ja.webp)

この章の中心 object は mask です。よくある failure は boundary quality、small objects、occlusion、class confusion です。

## Mask IoU check を動かす

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

## この順番で学ぶ

| 手順 | 読む内容 | 実践アウトプット |
|---|---|---|
| 1 | Semantic segmentation | すべての pixel に 1 つの class を予測する |
| 2 | Instance segmentation | 同じ class の別 object を分ける |
| 3 | Segmentation practice | masks、IoU/Dice、boundary errors、failed samples を比較する |

## 合格ライン

mask を作成または inspect し、簡単な overlap metric を計算し、boundary または class-confusion failure を説明できれば、この章は合格です。

<details>
<summary>確認の考え方と解説</summary>

1. 合格レベルの答えでは、task を class label、bounding box、mask、OCR text、embedding、video event など正しい視覚出力に対応づけます。
2. 証拠には、rendered visual artifact と、metric または定性的な error note を含めます。
3. class confusion、missed object、bad mask、lighting shift、domain shift、annotation quality など、失敗モードを1つ説明できればよいです。

</details>


## 残す証拠

このページを終えたら、この evidence card を残します。

```text
入力画像：元画像とターゲットマスクまたはクラスマップ
予測：予測マスク、重ね合わせ可視化、境界の例
指標：IoU、Dice、クラス別スコア、境界の失敗メモ
失敗確認：アノテーション品質、境界が薄い、領域が小さい、またはクラス混同
期待される成果：マスク重ね合わせとセグメンテーションメトリクス要約
```
