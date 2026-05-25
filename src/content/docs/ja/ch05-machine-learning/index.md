---
title: "5 機械学習入門から実践まで"
description: "実用的なモデリングループを学びます：タスク定義、データ分割、baseline、評価、エラー確認、特徴量改善、レポート作成。"
sidebar:
  order: 0
head:
  - tag: meta
    attrs:
      name: keywords
      content: "機械学習, Scikit-learn, 教師あり学習, 教師なし学習, 回帰, 分類, クラスタリング"
---
![機械学習のメインビジュアル](/img/course/ch05-machine-learning-ja.webp)

第 5 章の目的は 1 つです。データの問題を、**学習でき、評価でき、改善できる機械学習プロジェクト**に変えることです。

## メインルートでの位置

ここまでに、データが数値になること、loss と勾配がモデルの改善をどう説明するかを学びました。この章では、その考えを実務へ移します。予測問題を定義し、baseline を作り、指標を選び、エラーを見て、証拠があるときだけ改善します。

これは数学の直感からモデルエンジニアリングへ進む橋です。第 6 章では同じ証拠の習慣を保ったまま、モデルが tensor と逆伝播で学習するニューラルネットワークになります。

## まずモデリングループを見る

![機械学習モデリングの主線閉ループ](/img/course/ch05-modeling-loop-backbone-ja.webp)

先に図を見てください。信頼できる機械学習の多くは、この流れです。

```text
タスクを定義する -> データを分ける -> baseline を学習する -> 評価する -> エラーを見る -> 改善する
```

モデル名を追う前に baseline を作ります。baseline があると、後の変更が本当に改善したか判断できます。

## 学習順序とタスクリスト

このチェックリストを、本章の学習ガイド兼タスクリストとして使います。モデルを増やす前に、baseline と評価の習慣を先に作ります。

1. **[5.1 機械学習の基礎](/ja/ch05-machine-learning/ch01-ml-basics/00-roadmap/)**
   手を動かすこと：分類、回帰、クラスタリング、異常検知、特徴量、ラベル、train/test 分割、sklearn の流れを見分ける。
   残す証拠：問題定義メモ。

2. **[5.1.5 機械学習の歴史](/ja/ch05-machine-learning/ch01-ml-basics/04-history-breakthroughs/)**
   手を動かすこと：任意の背景として、古典アルゴリズムがなぜ現れたかを軽く読む。
   残す証拠：「このアルゴリズムがある理由」のメモ。

3. **[5.2 教師あり学習](/ja/ch05-machine-learning/ch02-supervised/00-roadmap/)**
   手を動かすこと：多数のモデル比較の前に、回帰と分類の例を動かす。
   残す証拠：baseline スコアと改善後スコア。

4. **[5.3 教師なし学習](/ja/ch05-machine-learning/ch03-unsupervised/00-roadmap/)**
   手を動かすこと：ラベルがないときにクラスタリング、次元削減、異常検知を試す。
   残す証拠：グラフまたはクラスタ解釈。

5. **[5.4 評価](/ja/ch05-machine-learning/ch04-evaluation/00-roadmap/)**
   手を動かすこと：指標を選び、交差検証を使い、バイアス/バリアンスを診断し、慎重に調整する。
   残す証拠：指標選択メモとエラーサンプル。

6. **[5.5 特徴量エンジニアリング](/ja/ch05-machine-learning/ch05-feature-engineering/00-roadmap/)**
   手を動かすこと：欠損値、カテゴリ、スケーリング、特徴量作成、特徴量選択、Pipeline を扱う。
   残す証拠：特徴量処理ログとリーク確認。

7. **[5.6 プロジェクト](/ja/ch05-machine-learning/ch06-projects/00-roadmap/) と [5.6.6 ワークショップ](/ja/ch05-machine-learning/ch06-projects/05-hands-on-ml-workshop/)**
   手を動かすこと：住宅価格、離反、分群、Kaggle の前に、再現可能な証拠パックを作る。
   残す証拠：README、モデル比較、エラー分析、次の計画。

## 必修ルート、拡張、深掘り

| 層 | いま学ぶこと | どう使うか |
|---|---|---|
| 必修コア | タスク種類、train/test 分割、baseline、指標、エラーサンプル、リーク確認、Pipeline | 後で LLM Prompt、RAG 検索、Agent 振る舞いの評価習慣になります |
| 任意の拡張 | 追加の古典アルゴリズム、機械学習史、Kaggle 型の反復 | より広いアルゴリズム比較や競技型ワークフローが必要なときに戻ります |
| 深掘り課題 | データと指標を固定し、特徴量またはモデル選択を 1 つだけ変え、前後のエラーを説明する | 根拠のないモデル変更を防ぎます |

本章でよく使う用語：

| 用語 | 意味 |
|---|---|
| `feature` | モデルが使える入力列 |
| `label` / `target` | モデルが予測する答え |
| `baseline` | まず超えるべき最も単純なモデルやルール |
| `metric` | F1、AUC、MAE、RMSE など、モデルを測るものさし |
| `leakage` | テストデータや答えの情報が学習へ漏れること |
| `Pipeline` | 前処理とモデルをまとめ、リークを減らす仕組み |

## 最初の実行ループ

sklearn がなければ先に入れます。

```bash
python -m pip install scikit-learn
```

次の自己完結した baseline を実行します。内蔵データセットを使い、データを分割し、dummy baseline と実モデルを学習して比較します。

```python
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
print("Baseline")
print(classification_report(y_test, baseline.predict(X_test), zero_division=0))

model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)
print("Logistic regression")
print(classification_report(y_test, model.predict(X_test), zero_division=0))
```

期待される形：

```text
Baseline
...
Logistic regression
...
```

最終スコアだけを比べないでください。どのクラスが簡単で、どのクラスが難しく、実運用でどのエラーが重要かを考えます。

### この出力の読み方

- baseline は、ほとんど有用なパターンを学ばない素朴なモデルがどこまでできるかを示します。
- Logistic regression は baseline を超えるべきですが、全体スコアよりクラスごとの precision と recall が重要です。
- あるクラスの recall が低いなら、モデルを変える前に、その見逃し例を確認します。
- 次の実験と比べるときは、データ分割、指標、失敗サンプルを固定します。

## 深度ラダー

| レベル | 証明できること |
|---|---|
| 最低合格 | タスク種類を言え、データを分け、baseline を学習し、スコアを読める。 |
| プロジェクト利用可 | その指標が目的に合う理由を説明し、1 つのエラーサンプルを示せる。 |
| 深い確認 | リークを確認し、2 つの特徴量案を比べ、実際の製品やデータ更新で何が変わるか言える。 |

## 失敗サンプル練習

この章を出る前に、間違った予測または弱いクラスタ解釈を1つ保存します。次の形式で書きます。

```text
case_id:
input_summary:
true_or_expected:
model_output:
why_it_matters:
next_controlled_change:
```

この小さな failure note は、別の model name を覚えるより役立ちます。後の deep learning curves、prompt evaluation、RAG retrieval errors、Agent traces を読む共通習慣になります。

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
モデリングループ：data、features、model、metric、エラー समीक्षा、次の実験
成果物: code、score、chart、pipeline、または project README
失敗確認: リーク、指標不一致、不安定な分割、過学習、または不明確なビジネス目標
次の行動：多数のパラメータ変更ではなく、1つの制御された実験を行う
期待される成果: 深層学習に備えた再現可能なML証拠
```

## よくある失敗

| 症状 | 最初に確認すること | よくある修正 |
|---|---|---|
| スコアが妙に高い | リークまたは train/test 分割ミス | 学習前に特徴量と分割方法を確認する |
| 学習スコアは高いがテストスコアが低い | 過学習 | モデルを単純にする、正則化する、データを増やす |
| すべてのモデルが弱い | ラベルが悪い、特徴量が弱い、指標が合っていない | エラーサンプルとラベル定義を見る |
| accuracy はよいが実務リスクが高い | クラス不均衡または false negative のコストが高い | recall、precision、F1、AUC、しきい値確認を使う |
| 結果を再現できない | random seed、データバージョン、依存関係が変わった | seed を固定し、バージョンを記録する |

## 通過チェック

次の 5 つに答えられたら、第 6 章へ進めます。

- このタスクは分類、回帰、クラスタリング、異常検知のどれですか？
- baseline は何で、実モデルはどのスコアを超える必要がありますか？
- 目的に合う指標は何で、accuracy はいつ誤解を招きますか？
- データリークをどう確認しましたか？
- モデルは何が得意で、何が苦手で、次にどこを改善しますか？

<details>
<summary>確認の考え方と解説</summary>

1. まず target を見ます。カテゴリを予測するなら分類、連続値を予測するなら回帰、ラベルがないならクラスタリングや異常検知を疑います。
2. baseline は、最も単純で再現できるモデルまたはルールです。同じ分割と同じ指標でそれを超えたときだけ、より複雑なモデルに意味があります。
3. 指標はミスのコストから選びます。クラス不均衡がある場合や、片方のミスが高コストな場合、accuracy は誤解を招きます。
4. リーク確認では、各特徴量に target、未来情報、テストデータ、人手レビュー結果が混じっていないかを確認します。
5. よい次の一手は、弱点を 1 つ、証拠サンプルを 1 つ、変更点を 1 つに絞って説明できます。

</details>

印刷用のチェックリストが必要なときは、[5.0 学習ガイドとタスクリスト](/ja/ch05-machine-learning/study-guide/) を使ってください。次の章では sklearn モデルからニューラルネットワークと深層学習へ進みます。
