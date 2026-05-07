---
title: "10.5.1 Advanced Vision ロードマップ：OCR、Face、Video、3D"
sidebar_position: 0
description: "Advanced vision directions の短い実践ロードマップ：input、output、risk、project goals に基づいて OCR、face、video、3D を選ぶ。"
keywords: [Advanced Vision Guide, OCR, Video Analysis, Face Recognition, 3D Vision]
---

# 10.5.1 Advanced Vision ロードマップ：OCR、Face、Video、3D

Advanced vision は model names のリストではありません。同じ vision foundation の上にある application directions で、inputs、outputs、constraints、risks がより複雑になります。

## 10.5.1.1 まず direction map を見る

![Advanced vision direction selection map](/img/course/ch10-advanced-vision-route-map-ja.png)

![OCR layout reading order map](/img/course/ch10-ocr-layout-reading-order-map-ja.png)

![Video frame tracking temporal window map](/img/course/ch10-video-frame-tracking-temporal-window-map-ja.png)

OCR は documents、face recognition は identity-sensitive scenarios、video は time and motion、3D vision は spatial structure に向いています。

## 10.5.1.2 Direction choice check を動かす

4 方向を浅く試すより、1 つを選びます。

```python
requirement = {
    "input": "screenshot",
    "needs_text": True,
    "needs_identity": False,
    "needs_time": False,
    "needs_depth": False,
}

if requirement["needs_text"]:
    direction = "OCR"
elif requirement["needs_identity"]:
    direction = "Face"
elif requirement["needs_time"]:
    direction = "Video"
elif requirement["needs_depth"]:
    direction = "3D"
else:
    direction = "Classification or detection"

print("direction:", direction)
print("first_output:", "text with layout")
```

出力：

```text
direction: OCR
first_output: text with layout
```

face、surveillance、medical、identity projects では、results を見せる前に privacy と usage boundaries を書きます。

## 10.5.1.3 この順番で学ぶ

| 手順 | 方向 | 実践アウトプット |
|---|---|---|
| 1 | OCR | text、layout、fields、confidence、failure samples を抽出する |
| 2 | Face | faces を検出し、threshold、privacy、bias risks を説明する |
| 3 | Video | frames をまたいで events を追跡し、temporal failures を記録する |
| 4 | 3D vision | depth、point cloud、geometry、sensor assumptions を説明する |

## 10.5.1.4 合格ライン

1 方向を選び、input/output を定義し、minimum project を動かし、failure cases と usage boundaries を文書化できれば、この章は合格です。
