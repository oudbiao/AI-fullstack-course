---
title: "8.2 プロジェクト：画像分類システム"
sidebar_position: 1
description: "本当に見せられる画像分類プロジェクトを題材に、テーマ選定、データ、baseline、学習、評価、デモ方法まで、ひとつながりの納品サイクルを通して学びます。"
keywords: [image classification project, CNN, confusion matrix, error analysis, computer vision]
---

# プロジェクト：画像分類システム

:::tip この節の位置づけ
画像分類は、最初のビジョン系プロジェクトとしてとても向いています。最も簡単だからではなく、工程全体の流れをいちばんわかりやすく説明しやすいからです。

- クラスをどう決めるか
- データをどう整理するか
- baseline をどう作るか
- 指標をどう見るか
- エラーをどう分析するか

この節の目標は、「動くモデル」を作ることではなく、**きちんと説明できるプロジェクト**を作ることです。
:::

## 学習目標

- 作品集に載せやすい画像分類テーマを定義できるようになる
- データ、baseline、評価、エラー分析をひとつの流れにまとめられるようになる
- 最小実行例でプロジェクト構成を表現できるようになる
- 画像分類プロジェクトで本当に見せるべきものを理解する

---

## 一、いきなりモデルを選ばず、まずはテーマを正しく決める

### 1.1 練習に向いているテーマには、たいてい3つの特徴がある

1. クラスの境界がはっきりしている  
   たとえば `猫 / 犬 / 鳥`、`リンゴの葉の病気分類`、`ごみ分類`
2. データを用意できる  
   最初から、サンプルをまったく集められないテーマは選ばない
3. エラーを説明できる  
   間違えたときに理由を説明できる。単に「モデルがダメ」で終わらない

### 1.2 とても安定したプロジェクトテーマ

たとえば：

> **「ペット写真分類器」を作って、画像を `cat / dog / rabbit` の3クラスに分ける。**

良い点は次のとおりです。

- クラスが直感的
- データを比較的集めやすい
- confusion matrix やエラーサンプル分析にとても向いている

### 1.3 最初からやらないほうがいいテーマ

たとえば：

- 数百クラスの細かい分類
- クラス境界がとてもあいまい
- データがかなり不均衡なのに、まだ対処方法を準備していない

---

## 二、プロジェクトの最小閉ループはどんな形か？

最小でも完全な画像分類プロジェクトには、少なくとも次が必要です。

1. テーマとラベルの定義
2. データセットの整理と分割
3. baseline
4. 学習と検証
5. 評価とエラー分析
6. デモ方法

この6つがきちんと説明できれば、モデルが複雑でなくても、プロジェクトとして十分説得力があります。

---

## 三、まずは最小のプロジェクト計画オブジェクトを見てみよう

```python
from dataclasses import dataclass, field


@dataclass
class CVProjectPlan:
    name: str
    classes: list
    dataset_split: dict
    baseline: str
    metrics: list
    risks: list = field(default_factory=list)


plan = CVProjectPlan(
    name="pet_image_classifier",
    classes=["cat", "dog", "rabbit"],
    dataset_split={"train": 900, "val": 180, "test": 180},
    baseline="small_cnn",
    metrics=["accuracy", "confusion_matrix", "error_cases"],
    risks=["クラス不均衡", "背景情報の漏れ", "ラベルノイズ"],
)

print(plan)
```

### 3.1 なぜこのオブジェクトが大事なのか？

プロジェクトの最初に足りなくなりやすいのは、コードではなく「境界」です。  
この最小オブジェクトは、先に次のことをはっきりさせるよう促してくれます。

- 何を作るのか
- どんなクラスがあるのか
- どんな baseline を使うのか
- どの指標で成功・失敗を判断するのか

---

## 四、まずは「疑似特徴量」の baseline で、プロジェクト評価を理解する

余計な依存を増やさずに、画像分類プロジェクトの検証フローを確認するために、小さな toy baseline を使います。

ここでは、各画像に次の3つのかなり粗い統計特徴があると仮定します。

- `fur`
- `ear_shape`
- `size`

もちろん実際のプロジェクトではこうはしませんが、次の流れを理解するのにとても役立ちます。

- 訓練データ
- クラスの代表値
- 予測
- confusion matrix

この一連の流れです。

```python
train_data = [
    ("cat", [0.9, 0.8, 0.4]),
    ("cat", [0.8, 0.7, 0.5]),
    ("dog", [0.7, 0.5, 0.8]),
    ("dog", [0.6, 0.4, 0.9]),
    ("rabbit", [0.5, 0.9, 0.3]),
    ("rabbit", [0.4, 0.8, 0.2]),
]

test_data = [
    ("cat", [0.85, 0.75, 0.45]),
    ("dog", [0.65, 0.45, 0.85]),
    ("rabbit", [0.45, 0.85, 0.25]),
    ("dog", [0.82, 0.72, 0.42]),  # わざと cat っぽい誤りサンプルを入れる
]


def class_prototypes(data):
    grouped = {}
    for label, features in data:
        grouped.setdefault(label, []).append(features)

    prototypes = {}
    for label, rows in grouped.items():
        dim = len(rows[0])
        prototypes[label] = [
            sum(row[i] for row in rows) / len(rows)
            for i in range(dim)
        ]
    return prototypes


def l1_distance(a, b):
    return sum(abs(x - y) for x, y in zip(a, b))


def predict(features, prototypes):
    distances = {label: l1_distance(features, proto) for label, proto in prototypes.items()}
    return min(distances, key=distances.get), distances


prototypes = class_prototypes(train_data)
print("prototypes:", prototypes)

results = []
for gold, features in test_data:
    pred, distances = predict(features, prototypes)
    results.append({"gold": gold, "pred": pred, "distances": distances})
    print(results[-1])
```

### 4.1 なぜこの例にもプロジェクト価値があるのか？

プロジェクトで本当に大切なのは、ライブラリ名ではなく評価の考え方だからです。  
この toy baseline だけでも、次の流れが見えます。

- train -> prototype
- test -> predict
- gold vs pred

これは、あとで本物の CNN プロジェクトになっても、必ず通る道です。

---

## 五、最小の confusion matrix とエラー分析

```python
labels = ["cat", "dog", "rabbit"]


def confusion_matrix(rows, labels):
    matrix = {g: {p: 0 for p in labels} for g in labels}
    for row in rows:
        matrix[row["gold"]][row["pred"]] += 1
    return matrix


cm = confusion_matrix(results, labels)
print("confusion matrix:")
for gold in labels:
    print(gold, cm[gold])

error_cases = [row for row in results if row["gold"] != row["pred"]]
print("\nerror cases:", error_cases)
```

### 5.1 なぜ confusion matrix は画像分類で特に大事なのか？

総合 accuracy だけでは、次のことしかわかりません。

- どれだけ正解したか

でも confusion matrix なら、次がわかります。

- どの2クラスが混ざりやすいか

これは、次にデータやモデルを改善するときにとても重要な情報です。

### 5.2 エラーサンプルが総得点より価値が高いのはなぜか？

実際に中身を見て、次のことを確認できるからです。

- 背景にモデルが引っ張られていないか
- あるクラスだけ写真の角度がばらばらではないか
- ラベル付けが間違っていないか

これこそが、画像プロジェクトで最も洞察が得られる部分です。

---

## 六、本当のプロジェクトで補うべき3つの層

### 6.1 データ層

少なくとも次を説明できる必要があります。

- 1クラスあたり何枚くらいあるか
- train / val / test をどう分けたか
- クラス不均衡があるか

### 6.2 モデル層

まずは2種類の baseline を作るのがおすすめです。

1. 小さな CNN
2. 転移学習モデル

こうすると、次のことがはっきり言えます。

- より複雑なモデルで何が改善されたのか

### 6.3 見せ方の層

画像分類プロジェクトを作品集に載せるとき、特に見せる価値が高いのは次の内容です。

- ラベル定義
- confusion matrix
- 典型的な正解サンプル
- 典型的な誤りサンプル

「学習が終わった」スクリーンショットを1枚貼るだけでは足りません。

---

## 七、このプロジェクトで特にハマりやすい落とし穴

### 7.1 総 accuracy だけを見る

本当に問題が多いクラス分布を見逃しやすくなります。

### 7.2 クラス定義が雑すぎる

クラス境界自体があいまいだと、モデルも評価も一緒にあいまいになります。

### 7.3 データリーク

似た画像が train と test の両方に入ると、  
結果が実力以上に良く見えてしまいます。

---

## まとめ

この節でいちばん大事なのは、プロジェクトの見方を身につけることです。

> **画像分類プロジェクトで本当に説得力があるのは、モデル名ではなく、クラス境界、データ整理、baseline、confusion matrix、エラー分析をひとつの閉ループとして説明できることです。**

この閉ループさえしっかり作れれば、小さなプロジェクトでも、作品級の課題にかなり近づきます。

---



## バージョン別の進め方

| バージョン | 目標 | 交付の重点 |
|---|---|---|
| 基礎版 | 最小の閉ループを通す | 入力できる、処理できる、出力できる。さらにサンプルを1組残す |
| 標準版 | 見せられるプロジェクトにする | 設定、ログ、エラー処理、README、スクリーンショットを追加する |
| チャレンジ版 | 作品集レベルに近づける | 評価、比較実験、失敗サンプル分析、次の改善方針を追加する |

まずは基礎版を完成させましょう。最初から大きく作りすぎないことが大切です。  
バージョンを1段階上げるたびに、「何が増えたか」「どう検証したか」「まだ何が課題か」を README に書きましょう。

## 練習

1. toy データの `dog` サンプルを2件追加して、confusion matrix がどう変わるか見てみましょう。
2. もし `cat` と `rabbit` がいつも混同されるなら、まず何を確認しますか？ データ、ラベル、それともモデルでしょうか。理由も考えてみましょう。
3. 画像分類プロジェクトが、なぜ confusion matrix を使った見せ方に向いているのか考えてみましょう。
4. このプロジェクトを作品集ページにするなら、最初にどの4項目を載せますか？
