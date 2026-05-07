---
title: "5.2.6 SVM：最大間隔と核法"
sidebar_position: 7
description: "初心者にもわかる形で、サポートベクターマシン、最大間隔、サポートベクター、核法、C、gamma、特徴量スケーリング、そして古典的機械学習での歴史的意義を学びます。"
keywords: [SVM, サポートベクターマシン, 最大間隔, サポートベクター, カーネルトリック, RBFカーネル, C, gamma, 監督学習]
---

# 5.2.6 SVM：最大間隔と核法

![SVM 最大間隔の直感図](/img/course/ch05-svm-margin-map-ja.png)

![SVM の間隔と核法コミック](/img/course/ch05-svm-margin-kernel-comic-ja.png)

:::tip この節の位置づけ
SVM は今では必ずしもすべてのプロジェクトで最初に選ぶモデルではありませんが、古典的機械学習の中ではとても重要な存在です。

新人がまず覚えておくべき、一番大事な一文はこれです：

> **分類は正しく分けるだけではなく、境界を両側のサンプルからできるだけ遠くに置くことも大切。**
:::

## 学習目標

- SVM がなぜ最大間隔の境界を重視するのか理解する
- サポートベクターとは何か、なぜ重要かを知る
- 数式に閉じ込められず、核法の直感を理解する
- `StandardScaler`、`SVC`、`C`、`gamma` を安全に使えるようになる
- いつ SVM を試す価値があり、いつ木のアンサンブルの方が実用的かを判断する

## 用語の整理

| 用語 | ここでの意味 | 実務での役割 |
|------|------|------|
| `SVM` | Support Vector Machine。最大間隔の境界を探すモデル | 小〜中規模データや境界の安定性を考えたい場面で有用です |
| `margin` | 境界から両側の最近傍サンプルまでの距離 | 間隔が大きいほど、境界は安定しやすいです |
| `support vector` | 境界に最も近い訓練サンプル | 境界をどこに置けるかを決めます |
| `kernel` | 変換後の特徴空間での類似度を計算する関数 | 手作業で全特徴を作らずに非線形境界を作れます |
| `RBF` | Radial Basis Function。代表的な非線形カーネル | 関係が直線ではなく曲がっているときのよい初期候補です |
| `C` | 分類ミスへの罰の強さ | 大きいほど訓練データに厳しく、小さいほど広い間隔を許します |
| `gamma` | RBF カーネルで 1 サンプルの影響が届く範囲 | 大きいほど境界は局所的で細かく曲がりやすいです |
| `StandardScaler` | 特徴量のスケールをそろえる前処理 | SVM は距離に基づくため、ほぼ必須です |
| `SVC` | sklearn の Support Vector Classifier | 分類用 SVM の例でよく使うクラスです |

---

## 一、なぜ SVM は生まれたのか？

ここまでで、ロジスティック回帰を学びました。ロジスティック回帰は、サンプルを2つのクラスに分ける境界線を学習します。

でも、ここで問題が出てきます。

> もし訓練サンプルを分けられる直線がたくさんあるなら、どれが一番よいのか？

SVM の答えはとても面白いです：

> **両側の最近傍サンプルから最も遠い線を選ぶ。**

これが最大間隔の考え方です。

3つのモデルは、次のように比べると理解しやすいです。

| モデル | 中心となる問い |
|---|---|
| ロジスティック回帰 | 「このサンプルが正例である確率はいくつか？」 |
| 決定木 | 「どの順番のルールでデータを分けられるか？」 |
| SVM | 「どの境界が最も安全で、最も広い間隔を残せるか？」 |

---

## 二、生活の例えで最大間隔を理解する

2つのクラスの列のあいだに、安全ラインを引く場面を想像してみてください。

- ただ分けられれば、それでも一応はOK
- でも、線がある1人の近くにぴったり引かれていると危険
- 少し動いただけで、境界を越えてしまうかもしれない

より安定した引き方はこうです：

> **安全ラインを、両側の間でできるだけ広い場所に置く。**

SVM は、まさにそれに似たことをしています。

| 概念 | たとえ |
|---|---|
| 決定境界 | 2つのクラスの間の安全ライン |
| 間隔 margin | 安全ラインから両側の最近傍サンプルまでの距離 |
| サポートベクター | 安全ラインに最も近い、いちばん重要なサンプル |

大事なのは、SVM が「訓練サンプルを正しく分けたか」だけでなく、「境界にどれだけ余裕があるか」も見ていることです。

---

## 三、サポートベクターとは何か？

SVM という名前の「サポートベクター」は、境界線に最も近いサンプルを指します。

これらが重要なのは、次の理由です。

- 境界から遠い点は、ふつう境界線を変えない
- 境界に最も近い点が、境界をどこに置けるかを決める

サポートベクターは「境界を支える点」と考えるとわかりやすいです。境界はすべてのサンプルの平均で決まるのではなく、最も重要で、最も危ういサンプルによって支えられます。

つまり SVM は「全部の点を同じように見る」モデルではありません。境界近くの重要な点が最終的な線を支えています。

---

## 四、核法：直線で分けられないなら、空間を変えて見る

SVM が歴史的に特に重要なのは、核法です。

データによっては、元の平面では分けられないことがあります。たとえば同心円です。

```text
元の空間：どう線を引いても分けにくい
高次元の見方：見方を変えると、平面で分けられるかもしれない
```

核法の直感はこうです：

> **本当にデータを高次元空間へ移して計算するとは限らない。核関数を使って、高次元空間での「似ている度合い」を効率よく計算する。**

これによって、SVM は非線形の境界にも対応できます。

最初は次のように覚えると十分です。

- `linear` カーネル：直線または超平面で分ける
- `rbf` カーネル：局所的な類似度を使って曲がった境界を作る
- `poly` カーネル：多項式的な曲がりを扱う

カーネルを丸暗記する前に、「この問題は直線境界では単純すぎるか？」を考えましょう。

---

## 五、最小の実行例

```python
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

X, y = make_moons(n_samples=300, noise=0.25, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)

model.fit(X_train, y_train)
svc = model.named_steps["svc"]

print(f"accuracy: {model.score(X_test, y_test):.3f}")
print(f"support vectors by class: {svc.n_support_.tolist()}")
print(f"total support vectors: {int(svc.n_support_.sum())}")
```

期待される出力：

```text
accuracy: 0.907
support vectors by class: [40, 39]
total support vectors: 79
```

ここで特に注目したい点は2つです。

- `StandardScaler()` はとても重要です。SVM は特徴量のスケールにかなり敏感だからです
- `kernel="rbf"` は、よく使われる非線形核を指定しています

---

## 六、なぜ特徴量スケーリングが重要なのか？

![SVM の特徴量スケーリングコミック](/img/course/ch05-svm-feature-scaling-ja.png)

SVM は距離と類似度に強く依存します。ある特徴量の単位が小さく、別の特徴量の単位が非常に大きい場合、大きなスケールの特徴が境界を支配してしまうことがあります。

この図は、実務上の注意として読んでください。ある特徴量が `0` から `1000`、別の特徴量が `0` から `10` の範囲だと、モデルは単に数値が大きいだけで前者を重要だと見なしてしまうことがあります。`StandardScaler` は各行の意味を変えるのではなく、距離ベースのモデルが特徴量を公平に比べられるように座標系を整えます。

```python
X_scaled = X.copy()
X_scaled[:, 1] *= 100  # 2つ目の特徴量を人工的に大きくする

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_scaled, y, test_size=0.25, random_state=42
)

raw_model = SVC(kernel="rbf", C=1.0, gamma="scale")
raw_model.fit(X_train2, y_train2)

scaled_model = make_pipeline(
    StandardScaler(),
    SVC(kernel="rbf", C=1.0, gamma="scale")
)
scaled_model.fit(X_train2, y_train2)

print(f"without scaling: {raw_model.score(X_test2, y_test2):.1%}")
print(f"with scaling:    {scaled_model.score(X_test2, y_test2):.1%}")
```

期待される出力：

```text
without scaling: 81.3%
with scaling:    90.7%
```

これは SVM のとても実践的な教訓です。SVM にとって前処理は飾りではありません。モデルにとっての「近い」「遠い」の意味を変えます。

---

## 七、線形カーネル vs RBF カーネル

```python
for kernel in ["linear", "rbf"]:
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel=kernel, C=1.0, gamma="scale")
    )
    clf.fit(X_train, y_train)
    svc = clf.named_steps["svc"]
    print(
        f"kernel={kernel:6s}: "
        f"train={clf.score(X_train, y_train):.1%}, "
        f"test={clf.score(X_test, y_test):.1%}, "
        f"support_vectors={int(svc.n_support_.sum())}"
    )
```

期待される出力：

```text
kernel=linear: train=84.9%, test=90.7%, support_vectors=80
kernel=rbf   : train=90.7%, test=90.7%, support_vectors=79
```

この小さなデータではテストスコアが近いですが、意味は違います。

- 線形 SVM は境界を直線に保とうとします
- RBF SVM は非線形構造に合わせて境界を曲げられます

実務では、1回の訓練/テスト分割だけで決めず、交差検証で比べましょう。

---

## 八、`C` と `gamma` をどう理解するか

初学者が特に迷いやすいのは、`C` と `gamma` です。まずは次のように覚えると理解しやすいです。

![SVM の C と gamma の境界調整コミック](/img/course/ch05-svm-c-gamma-boundary-ja.png)

| パラメータ | 最初の直感 | 小さすぎると | 大きすぎると |
|---|---|---|---|
| `C` | 分類ミスをどれくらい厳しく罰するか | 境界は広くなるが、未学習になりやすい | 各訓練点を分けようとしすぎて、過学習しやすい |
| `gamma` | RBF 核で、1つのサンプルの影響がどこまで届くか | 境界はなめらかで広くなる | 境界がサンプルの周りで細かく曲がりやすい |

```python
from sklearn.model_selection import cross_val_score

settings = [
    (0.1, "scale"),
    (1.0, "scale"),
    (100.0, "scale"),
    (1.0, 0.1),
    (1.0, 10.0),
]

for C, gamma in settings:
    clf = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=C, gamma=gamma)
    )
    cv_scores = cross_val_score(clf, X_train, y_train, cv=5, scoring="accuracy")
    clf.fit(X_train, y_train)
    print(
        f"C={C:<5}, gamma={str(gamma):<5}: "
        f"cv={cv_scores.mean():.1%} ± {cv_scores.std():.1%}, "
        f"test={clf.score(X_test, y_test):.1%}"
    )
```

期待される出力：

```text
C=0.1  , gamma=scale: cv=87.1% ± 4.5%, test=90.7%
C=1.0  , gamma=scale: cv=89.3% ± 3.8%, test=90.7%
C=100.0, gamma=scale: cv=90.7% ± 2.6%, test=92.0%
C=1.0  , gamma=0.1  : cv=84.4% ± 5.3%, test=92.0%
C=1.0  , gamma=10.0 : cv=90.7% ± 2.2%, test=94.7%
```

この小さなデータだけで結論を急がないでください。大事なのは、`C` と `gamma` を交差検証で調整し、最後に保留したテストセットで確認する習慣です。

---

## 九、SVM、ロジスティック回帰、決定木はどう選ぶ？

| モデル | 何をしているイメージか | 新人はどう理解するとよいか |
|---|---|---|
| ロジスティック回帰 | 確率的な線形境界を学習する | いちばん基本的な分類の baseline |
| SVM | 最大間隔の境界を学習する | 分類境界は安定させて、サンプルに近づけすぎない |
| 決定木 | ルールで少しずつデータを分ける | 人が読めるルールの木に近い |
| ランダムフォレスト / Boosting | たくさんの木を組み合わせる | 表形式データで強い baseline |

SVM の強みは、境界の考え方がとても美しいことです。小〜中規模データでは、よい結果が出ることもよくあります。一方で、大規模データでは学習が遅くなることがあり、パラメータや核関数の選択にも経験が必要です。

実務の最初の順番としては、次が安定です。

1. まずロジスティック回帰で単純な baseline を作る
2. データが小〜中規模で、間隔やカーネルが効きそうなら SVM を試す
3. 表形式データで強い実用 baseline が欲しいなら、ランダムフォレストや Boosting を試す

---

## 十、SVM を歴史の流れに戻して見る

1995 年、Corinna Cortes と Vladimir Vapnik による論文「Support-Vector Networks」は、最大間隔分類器を古典的機械学習の重要な節目にしました。

歴史的に重要なのは、いつまでも最強だからではありません。次の2点をとても明確に示したからです。

- 汎化は、訓練集で正しく分けられるかだけでは判断できない
- 決定境界をサンプルから少し遠ざけると、モデルはふつうより安定する

だからこそ、今でも多くの表形式データでは XGBoost、LightGBM、ランダムフォレストを先に試すことが多いとしても、SVM を学ぶ価値は十分にあります。

---

## まとめ

| ポイント | 覚えること |
|------|------|
| 最大間隔 | ただ分けられる境界ではなく、もっとも安全な境界を選ぶ |
| サポートベクター | 最も近いサンプルが境界を決める |
| カーネルトリック | より豊かな空間で見ているかのように類似度を計算する |
| スケーリング | SVM は距離に基づくため、特徴量の尺度が重要 |
| `C` と `gamma` | 訓練スコアだけでなく、交差検証で調整する |

## この節でいちばん持ち帰ってほしいこと

この最初の段階では、SVM の最適化公式を全部導出できなくても大丈夫です。それより大事なのは、まず次の3つの直感を持つことです。

1. SVM は、訓練集で正しく分けるだけでなく、最大間隔を目指す
2. サポートベクターは、境界を決める重要なサンプル
3. 核法によって、線形モデルでも非線形を扱えるようになる

「なぜ SVM では特徴量のスケーリングがよく必要なのか」を説明できるようになれば、もう単なるアルゴリズム名ではなく、実務で使うモデルとして理解できています。

## 手を動かしてみよう

### 練習 1：`C` を調整する

`make_moons` を使い、`gamma="scale"` のまま、`C=[0.01, 0.1, 1, 10, 100]` を試して交差検証の正解率を比べましょう。

### 練習 2：`gamma` を調整する

`C=1` のまま、`gamma=[0.01, 0.1, 1, 10]` を試し、それぞれの決定境界を描いてください。

### 練習 3：スケーリング実験

1つの特徴量を 100 倍または 1000 倍し、`StandardScaler` あり/なしで SVM の性能を比較してください。
