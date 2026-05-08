---
title: "10.1.4 画像処理技術"
sidebar_position: 3
description: "フィルタリング、エッジ検出、形態学的処理まで、実行できる OpenCV の例で古典的な画像処理の仕組みを理解します。"
keywords: [フィルタリング, エッジ検出, 形態学, OpenCV, Canny, blur]
---

# 10.1.4 画像処理技術

![画像処理パイプライン図](/img/course/cv-image-processing-pipeline-ja.png)

## 学習目標

この節を終えると、あなたは次のことができるようになります。

- 画像フィルタリングが何をしているかを理解する
- OpenCV を使って平滑化、エッジ検出、二値化を行う
- 膨張、収縮などの形態学的処理の直感を理解する
- 古典的な画像処理タスクの基本コードを読み解く

---

## 一、画像処理は何を処理しているの？

古典的な画像処理は、次のように考えられます。

> **一連のルールで、ピクセルを再調整すること。**

深層学習と違って、「データからルールを学ぶ」のではなく、あらかじめルールを自分で書きます。

代表的なタスクは次のとおりです。

- ノイズ除去
- ぼかし
- エッジ抽出
- 二値化
- 輪郭強調

:::info 依存関係のインストール
以下のコードはそのまま実行できます。

```bash
pip install opencv-python numpy
```
:::

---

## 二、まずはテスト画像を作る

サンプルを外部画像に依存させないため、まずは自分で簡単な画像を生成します。

```python
import cv2
import numpy as np

img = np.zeros((240, 320), dtype=np.uint8)

# 白い長方形と灰色の円を描く
cv2.rectangle(img, (30, 40), (140, 180), 255, -1)
cv2.circle(img, (230, 120), 45, 180, -1)

cv2.imwrite("processing_original.png", img)
print("processing_original.png を保存しました")
```

実行結果の例：

```text
processing_original.png を保存しました
```

ここではグレースケール画像を直接使います。あとでエッジや閾値処理をする際に扱いやすいからです。

---

## 三、フィルタリング：画像を「少しなめらかにする」

フィルタリングの直感は次のようなものです。

> 周囲のピクセルの値も考慮して、画像をより滑らかにする。

### 平均フィルタ

```python
import cv2
import numpy as np

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
blurred = cv2.blur(img, (7, 7))

cv2.imwrite("processing_blur.png", blurred)
print("processing_blur.png を保存しました")
```

実行結果の例：

```text
processing_blur.png を保存しました
```

平均フィルタはエッジをやわらかくしますが、細部も失いやすくなります。

### ガウシアンフィルタ

```python
import cv2

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
gaussian = cv2.GaussianBlur(img, (7, 7), 0)

cv2.imwrite("processing_gaussian.png", gaussian)
print("processing_gaussian.png を保存しました")
```

実行結果の例：

```text
processing_gaussian.png を保存しました
```

ガウシアンフィルタは、単純な平均フィルタよりもよく使われます。より自然な見た目になりやすいからです。

---

## 四、エッジ検出：変化が最も大きい場所を見つける

エッジは次のように考えられます。

> 明るさが急に変わる位置

たとえば、黒い背景にある白い長方形の境界は、典型的なエッジです。

### Canny エッジ検出

```python
import cv2

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(img, threshold1=50, threshold2=150)

cv2.imwrite("processing_edges.png", edges)
print("processing_edges.png を保存しました")
```

実行結果の例：

```text
processing_edges.png を保存しました
```

### 2つの閾値はどう考える？

ざっくり次のように覚えるとよいです。

- 低い閾値より小さい: ほぼエッジではない
- 高い閾値より大きい: エッジの可能性が高い
- その中間: 周辺とのつながりも見て判断する

---

## 五、閾値処理：グレースケール画像を白黒画像にする

閾値処理は、線を1本引くイメージです。

- その値より大きいものは白
- その値より小さいものは黒

```python
import cv2

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)
_, binary = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)

cv2.imwrite("processing_binary.png", binary)
print("processing_binary.png を保存しました")
```

実行結果の例：

```text
processing_binary.png を保存しました
```

この操作は、次のような場面でよく使われます。

- 文書スキャン
- 前景 / 背景の分離
- 輪郭抽出の前処理

---

## 六、形態学的処理：形を加工する

形態学的処理は、特に二値画像の処理に向いています。

「白い領域を少し揉む、広げる、縮める」と考えるとわかりやすいです。

### 収縮（Erosion）

白い領域が小さくなります。

```python
import cv2
import numpy as np

img = cv2.imread("processing_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
eroded = cv2.erode(img, kernel, iterations=1)

cv2.imwrite("processing_eroded.png", eroded)
print("processing_eroded.png を保存しました")
```

実行結果の例：

```text
processing_eroded.png を保存しました
```

### 膨張（Dilation）

白い領域が大きくなります。

```python
import cv2
import numpy as np

img = cv2.imread("processing_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)
dilated = cv2.dilate(img, kernel, iterations=1)

cv2.imwrite("processing_dilated.png", dilated)
print("processing_dilated.png を保存しました")
```

実行結果の例：

```text
processing_dilated.png を保存しました
```

### オープニングとクロージング

- オープニング = 先に収縮してから膨張。小さなノイズを取り除くのに向いている
- クロージング = 先に膨張してから収縮。小さな穴を埋めるのに向いている

```python
import cv2
import numpy as np

img = cv2.imread("processing_binary.png", cv2.IMREAD_GRAYSCALE)
kernel = np.ones((5, 5), np.uint8)

opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

cv2.imwrite("processing_opened.png", opened)
cv2.imwrite("processing_closed.png", closed)
print("processing_opened.png と processing_closed.png を保存しました")
```

実行結果の例：

```text
processing_opened.png と processing_closed.png を保存しました
```

![古典的な画像処理操作の選択図](/img/course/ch10-image-processing-operation-decision-map-ja.png)

:::tip 読み方のヒント
古典的な画像処理は、ただの API 一覧ではありません。目的がはっきりしたピクセルルールの集まりです。ノイズ除去にはフィルタリング、変化を見つけるにはエッジ、前景と背景を分けるには閾値処理、小さなノイズや穴の処理には形態学的処理を使います。
:::

---

## 七、これらの処理をつなげてみる

実際のタスクでは、これらの処理は連続して使われることがよくあります。

たとえば、ある対象の輪郭を抽出したいなら、次のような流れになります。

1. グレースケールに変換
2. フィルタリングでノイズ除去
3. 二値化
4. 形態学的処理で整える
5. さらにエッジ検出や輪郭解析を行う

以下に、ひとつの小さな処理フローの例を示します。

```python
import cv2
import numpy as np

img = cv2.imread("processing_original.png", cv2.IMREAD_GRAYSCALE)

# ノイズ除去
smoothed = cv2.GaussianBlur(img, (5, 5), 0)

# 二値化
_, binary = cv2.threshold(smoothed, 100, 255, cv2.THRESH_BINARY)

# クロージングで小さなすき間を埋める
kernel = np.ones((5, 5), np.uint8)
cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# エッジ抽出
edges = cv2.Canny(cleaned, 50, 150)

cv2.imwrite("processing_pipeline_smoothed.png", smoothed)
cv2.imwrite("processing_pipeline_binary.png", binary)
cv2.imwrite("processing_pipeline_cleaned.png", cleaned)
cv2.imwrite("processing_pipeline_edges.png", edges)
print("処理フロー全体の結果を保存しました")
```

実行結果の例：

```text
処理フロー全体の結果を保存しました
```

---

## 八、なぜ今でも古典的な手法を学ぶの？

今でもとても役立つからです。

- 深層学習の前処理として使える
- 小さなプロジェクトで素早く効果を出せる
- 工業用途でルールベースの補完ができる
- 「画像がどう処理されるか」の直感を身につけられる

初心者はすぐに CNN を学びたくなりがちですが、グレースケール、エッジ、閾値の感覚がないと、あとで画像モデルを理解するのが難しくなります。

---

## 九、初心者がよくやる間違い

### フィルタリングは「画像をきれいに見せるため」だけだと思う

それだけではありません。  
多くの場合、後続のアルゴリズムを安定させるために使います。

### 閾値は固定で変わらないと思う

実際の画像では照明の変化が大きいので、閾値は場面に合わせて調整することがよくあります。

### API だけ学んで、目的を理解しない

常に自分にこう問いかけましょう。

- このステップはノイズ除去？
- それとも境界の強調？
- それとも形の整理？

---

## まとめ

この節で押さえるべき核心は次のとおりです。

> **古典的な画像処理とは、ルールを使ってピクセルを並べ替え、選び直すことです。**

これは深層学習そのものではありませんが、画像認識タスクを理解するうえでとても重要な土台です。

---

## 練習

1. `threshold()` の閾値を `60`、`120`、`180` に変えて、二値画像の変化を観察しましょう。
2. 収縮と膨張のカーネルサイズを `(3, 3)` から `(7, 7)` に変えて、形の変化を観察しましょう。
3. 元の画像に小さな白い点を1つ追加して、オープニングでそれを消せるか試してみましょう。
