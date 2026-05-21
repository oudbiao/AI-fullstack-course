---
title: "10.1.3 OpenCV 基本操作"
sidebar_position: 2
description: "OpenCV を使って画像の読み書き、拡大縮小、切り抜き、色変換、描画を学び、CV 実践の第一歩を踏み出しましょう。"
keywords: [OpenCV, cv2, 画像読み込み, 画像リサイズ, 描画, 色変換]
---

# 10.1.3 OpenCV 基本操作

## 学習目標

この節を終えると、あなたは次のことができるようになります。

- OpenCV を使って画像を作成、読み込み、保存する
- 拡大縮小、切り抜き、反転などの基本変換を行う
- OpenCV でよくある色順の問題を理解する
- OpenCV を使って画像上に四角形、円、文字を描画する

---

## 一、なぜ CV 入門はほぼ OpenCV から始まるのか？

OpenCV はコンピュータビジョンの「スイスアーミーナイフ」のようなものだからです。

- 画像の読み込み、書き込みができる
- 拡大縮小、回転、切り抜きができる
- フィルタ処理、エッジ検出ができる
- 顔検出、動画処理ができる

しかも、初学者が「実務っぽさ」を身につけるのにとても向いています。

:::info 依存関係のインストール
下のコードはそのまま実行できます。

```bash
pip install opencv-python numpy
```
:::

---

## 二、外部ファイルに頼らず、まず画像を自分で作る

コードをそのまま実行できるように、まずは空白画像を自分で作ってみます。

```python
import cv2
import numpy as np

# 240 高、320 幅、3 チャンネルの黒いキャンバスを作成
img = np.zeros((240, 320, 3), dtype=np.uint8)

print("shape:", img.shape)
print("dtype:", img.dtype)

cv2.imwrite("opencv_blank.png", img)
print("opencv_blank.png を保存しました")
```

実行結果の例：

```text
shape: (240, 320, 3)
dtype: uint8
opencv_blank.png を保存しました
```

ここでの `shape = (240, 320, 3)` は、次の意味です。

- 高さ 240
- 幅 320
- 3 つの色チャンネル

---

## 三、OpenCV の色順は RGB ではなく BGR

これは非常によくある落とし穴です。

OpenCV のデフォルトは、

> **BGR**

です。
私たちがよく知っている RGB ではありません。

```python
import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)

# この色は BGR であって、RGB ではありません
img[:, :] = (255, 0, 0)

cv2.imwrite("opencv_blue.png", img)
print("青い画像 opencv_blue.png を保存しました")
```

実行結果の例：

```text
青い画像 opencv_blue.png を保存しました
```

もし `(255, 0, 0)` を赤だと思っていると、「色が違う」画像になってしまいます。

### RGB に変換する

```python
import cv2
import numpy as np

img_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
img_bgr[:, :] = (255, 0, 0)

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

print("BGR ピクセル:", img_bgr[0, 0].tolist())
print("RGB ピクセル:", img_rgb[0, 0].tolist())
```

実行結果の例：

```text
BGR ピクセル: [255, 0, 0]
RGB ピクセル: [0, 0, 255]
```

---

## 四、よく使う基本操作：拡大縮小、切り抜き、反転

```python
import cv2
import numpy as np

img = np.zeros((200, 300, 3), dtype=np.uint8)
img[:, :] = (40, 180, 240)

# 拡大縮小
small = cv2.resize(img, (150, 100))

# 切り抜き：先に行、次に列、つまり [y1:y2, x1:x2]
crop = img[50:150, 80:220]

# 反転
flip_horizontal = cv2.flip(img, 1)

print("元画像:", img.shape)
print("縮小後:", small.shape)
print("切り抜き後:", crop.shape)
print("水平反転後:", flip_horizontal.shape)

cv2.imwrite("opencv_small.png", small)
cv2.imwrite("opencv_crop.png", crop)
cv2.imwrite("opencv_flip.png", flip_horizontal)
```

実行結果の例：

```text
元画像: (200, 300, 3)
縮小後: (100, 150, 3)
切り抜き後: (100, 140, 3)
水平反転後: (200, 300, 3)
```

### なぜ切り抜きは `[y1:y2, x1:x2]` と書くのか？

画像は本質的には 2 次元配列なので、配列のアクセス順は次のようになります。

1. 先に行（高さ方向、`y`）
2. 次に列（幅方向、`x`）

![OpenCV BGR、座標と切り抜き順序の図](/img/course/ch10-opencv-bgr-coordinate-crop-map-ja.webp)

:::tip 読み方のヒント
OpenCV 入門でよくある 2 つのつまずきは、色のデフォルトが RGB ではなく BGR であること、そして配列の切り抜きが `y` を先に、`x` を後に書くことです。この図を見るときは、画像を「行と列の配列」として捉え、平面の座標用紙として見ないようにしましょう。
:::

---

## 五、画像に描画する

多くの画像処理タスクでは、結果を画像上に注釈として描く必要があります。たとえば、

- 検出枠を描く
- クラス名を表示する
- 中心点を示す

```python
import cv2
import numpy as np

canvas = np.ones((300, 400, 3), dtype=np.uint8) * 255

# 四角形を描く
cv2.rectangle(canvas, (50, 50), (180, 180), (0, 255, 0), 2)

# 円を描く
cv2.circle(canvas, (280, 120), 40, (255, 0, 0), -1)

# 直線を描く
cv2.line(canvas, (30, 250), (350, 250), (0, 0, 255), 3)

# 文字を書く
cv2.putText(
    canvas,
    "CV Demo",
    (120, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 0, 0),
    2
)

cv2.imwrite("opencv_draw_demo.png", canvas)
print("opencv_draw_demo.png を保存しました")
```

実行結果の例：

```text
opencv_draw_demo.png を保存しました
```

---

## 六、グレースケール画像への変換

多くの古典的な画像処理では、まずカラー画像をグレースケール画像に変換します。理由は次の通りです。

- 計算が速くなる
- 色の影響を取り除ける
- 明るさの情報だけを残せる

```python
import cv2
import numpy as np

img = np.zeros((100, 100, 3), dtype=np.uint8)
img[:, :50] = (0, 0, 255)      # 赤
img[:, 50:] = (0, 255, 0)      # 緑

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

print("元画像の shape:", img.shape)
print("グレースケール画像の shape:", gray.shape)
print("グレースケール画像の最初の 5 ピクセル:", gray[0, :5].tolist())

cv2.imwrite("opencv_gray.png", gray)
```

実行結果の例：

```text
元画像の shape: (100, 100, 3)
グレースケール画像の shape: (100, 100)
グレースケール画像の最初の 5 ピクセル: [76, 76, 76, 76, 76]
```

---

## 七、小さなプロジェクト：「情報カード画像」を作る

この例では、これまでの知識をまとめて使います。画像の作成、描画、文字表示、保存を行います。

```python
import cv2
import numpy as np

card = np.ones((220, 420, 3), dtype=np.uint8) * 245

cv2.rectangle(card, (20, 20), (400, 200), (60, 120, 200), 2)
cv2.circle(card, (80, 85), 35, (60, 120, 200), -1)

cv2.putText(card, "AI Fullstack", (140, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2)
cv2.putText(card, "Chapter 10: CV Basics", (140, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
cv2.putText(card, "OpenCV starter demo", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (20, 20, 20), 2)

cv2.imwrite("opencv_info_card.png", card)
print("opencv_info_card.png を保存しました")
```

実行結果の例：

```text
opencv_info_card.png を保存しました
```

![OpenCV 基本操作の保存結果図](/img/course/ch10-opencv-saved-outputs-result-map-ja.webp)

---

## 八、初学者によくある誤解

### `cv2.imshow()` の結果ウィンドウが開かない

多くのリモート環境、Notebook、サーバー環境では、`imshow()` は使いにくいです。
教材やスクリプトでは、まず `cv2.imwrite()` で結果を保存する方法をおすすめします。

### BGR を RGB だと思い込む

これは OpenCV 初学者に最もよくあるバグの 1 つです。

### 切り抜きで `x` と `y` の順番を逆にする

画像配列のインデックスは `[y, x]` であって、`[x, y]` ではありません。

---

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
入力画像：実行で使うソース画像または生成画像
配列形状: 幅、高さ、channels、dtype、座標規約
処理済み出力：グレースケール、切り抜き、エッジ、しきい値処理、または保存済み中間画像
失敗確認：チャネル順、リサイズの歪み、座標ミス、または過剰処理
期待される成果：前後の画像と、出力された shape またはピクセル値
```

## まとめ

この節のポイントは、OpenCV の API をすべて暗記することではなく、「もう画像を操作できる」という感覚を身につけることです。

- 画像を作成できる
- 画像を変換できる
- 画像に注釈を付けられる
- 結果を保存できる

ここまでできれば、次の節のフィルタ処理、エッジ検出、形態学的処理もかなり楽になります。

---

## 練習

1. キャンバスの色を別の色に変えて、もう一度カード画像を作ってみましょう。
2. 同じ画像に複数の四角形や円を描いて、座標系に慣れましょう。
3. 画像をいくつかの解像度にリサイズしてから、結果を保存してみましょう。

<details>
<summary>解法と解説</summary>

1. OpenCV の描画関数を使う場合、色のタプルは多くの場合 RGB ではなく BGR です。正しいカード画像は保存でき、開いたときに意図した色になっているはずです。
2. 矩形や円を描くときは、座標が画像の範囲内にあるか確認します。描画順も重要で、後から描いた図形が前の図形を覆うことがあります。
3. リサイズは画素数を変えます。幅と高さの比率を変えると画像が歪むので、比較するときは歪ませた版とアスペクト比を保った版を分けて残すとよいです。

</details>
