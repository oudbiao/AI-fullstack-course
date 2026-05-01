---
title: "1.3 埋め込みと意味表現"
sidebar_position: 2
description: "one-hot から dense vector、さらに文表現と文脈依存表現へ。モデルがなぜ「意味が近い」をベクトル空間の距離として扱えるのかを理解します。"
keywords: [embedding, semantic representation, cosine similarity, sentence embedding, contextual embedding]
---

# 埋め込みと意味表現

![Embedding の意味空間図](/img/course/embedding-semantic-space.png)

:::tip この節の位置づけ
Tokenizer が解決するのは：

- テキストをどう分割するか

Embedding が解決するのは：

- 分割された token を、どうやって意味のあるベクトルに変えるか

Embedding を初めて学ぶと、こう考えがちです：

- それぞれの単語に数字の並びを割り当てる

でも、それだけでは不十分です。  
本当に大事なのは次です。

> **これらの数字は適当に埋められるのではなく、だんだんと「意味空間」を作り、近い単語や近い文同士が空間上で近づいていくことです。**
:::

## 学習目標

- one-hot と dense embedding の根本的な違いを理解する
- 似た意味がなぜベクトルの距離に表れるのかを理解する
- 単語ベクトル、文ベクトル、文脈依存表現の流れを理解する
- 動く例を通して、embedding がどのように類似度計算を支えるかを理解する

---

## 一、なぜ one-hot で単語を直接表せないのか？

### 1.1 one-hot はきれいだけれど、意味関係は表せない

たとえば、語彙に次の4語があるとします。

- `refund`
- `return`
- `password`
- `banana`

one-hot 表現はこんな感じです。

- `refund` -> `[1, 0, 0, 0]`
- `return` -> `[0, 1, 0, 0]`
- `password` -> `[0, 0, 1, 0]`

問題は：

- `refund` と `return` は意味がとても近い
- `refund` と `banana` は意味がとても遠い

でも one-hot の空間では、どれも同じように「遠い」のです。

つまり：

> **one-hot は「同一かどうか」は区別できても、似ているかどうかは表せません。**

### 1.2 Dense embedding の本当の価値

embedding がやりたいことは：

- 意味が近い単語のベクトルも近くすること

たとえば：

- `refund` と `return`
- `reset` と `recover`

これらはベクトル空間の中で近くに置けます。

ここが embedding の重要な点です。

- 単なるエンコードではない
- 意味を表現するもの

### 1.3 たとえ話：単語を地図に置く

embedding は地図の座標のように考えられます。

- one-hot は身分証番号のようなもので、区別はできる
- embedding は地図の位置のようなもので、区別だけでなく、誰が誰に近いかも分かる

この意味の地図ができると、  
モデルは次のような関係を見つけやすくなります。

- 似た文脈で出る単語はどれか
- 似た意味を表す文はどれか

![One-hot から dense embedding への意味空間図](/img/course/ch07-embedding-onehot-dense-map.png)

:::tip 図の見方
この図のポイントは比較です。one-hot は身分証番号のように「同じ単語かどうか」しか区別できません。一方、dense embedding は地図の座標のように「どちらが近いか」を表せます。ここから、テキストは初めて計算できる意味空間に入ります。
:::

---

## 二、なぜ単語ベクトルに意味が宿るのか？

### 2.1 文脈の中で学習されるから

embedding は人が決めるものではありません。  
通常は学習の中で少しずつ獲得されます。

もし2つの単語が似た文脈にたくさん出てくるなら、  
モデルはそれらを近いベクトルとして学習する傾向があります。

これは有名な分布仮説です。

> **単語の意味は、その単語が現れる文脈によって大きく決まる。**

### 2.2 意味が近いことと、完全に同義であることは違う

ベクトルが近いというのは、次のことを意味しがちです。

- 使われ方が似ている
- 文脈の分布が近い

でも、次とは限りません。

- 完全に置き換え可能

たとえば：

- `doctor` と `hospital`

は、よく一緒に出るので近くなることがあります。  
つまり embedding における「近さ」は、意味の近さというより、分布上の近さです。

### 2.3 単語から文へ、表現はさらに積み上げられる

複数の token ベクトルを組み合わせると、  
次のような表現を作れます。

- フレーズベクトル
- 文ベクトル
- 段落ベクトル

だから embedding は単語の類似度だけでなく、  
次の用途でも広く使われます。

- 検索
- クラスタリング
- 分類
- RAG

---

## 三、まずは意味比較のイメージが分かる例を実行してみよう

次のコードでは、3つのことをします。

1. いくつかの単語に小さな embedding を与える
2. 単語同士の cosine similarity を計算する
3. 「token ベクトルの平均」で文ベクトルを作り、文の類似度を比較する

```python
from math import sqrt

embeddings = {
    "refund": [0.90, 0.80, 0.10],
    "return": [0.88, 0.78, 0.12],
    "password": [0.10, 0.20, 0.95],
    "reset": [0.12, 0.18, 0.92],
    "order": [0.75, 0.70, 0.15],
    "banana": [0.05, 0.95, 0.10],
}


def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(x * x for x in b))
    return dot / (norm_a * norm_b)


def sentence_embedding(tokens, embedding_table):
    valid = [embedding_table[token] for token in tokens if token in embedding_table]
    dim = len(valid[0])
    return [sum(vec[i] for vec in valid) / len(valid) for i in range(dim)]


print("refund vs return  :", round(cosine(embeddings["refund"], embeddings["return"]), 3))
print("refund vs password:", round(cosine(embeddings["refund"], embeddings["password"]), 3))

query_a = ["reset", "password"]
query_b = ["password", "reset"]
query_c = ["refund", "order"]

vec_a = sentence_embedding(query_a, embeddings)
vec_b = sentence_embedding(query_b, embeddings)
vec_c = sentence_embedding(query_c, embeddings)

print("query_a vs query_b:", round(cosine(vec_a, vec_b), 3))
print("query_a vs query_c:", round(cosine(vec_a, vec_c), 3))
```

### 3.1 このコードは何を示しているのか？

ここでは2つのレベルを示しています。

第1のレベル：

- `refund` と `return` のような近い意味の単語は、ベクトル空間でも近くなる

第2のレベル：

- token ベクトルをまとめると、文同士でも類似度を比べられる

これが、embedding が意味検索や recall を支えられる理由です。

### 3.2 なぜ `query_a` と `query_b` はとても近くなるのか？

単語の順序が違うだけで、  
平均するとほぼ同じ表現になるからです。

同時に、これは単純平均の弱点も示しています。

- 順序をほとんど見ない

そのため、初期の静的な文ベクトルは便利でしたが、  
表現力には限界がありました。

### 3.3 それでもこのコードに価値がある理由

embedding の本質的な直感をつかめるからです。

> **「意味が近い」は「ベクトルが近い」に変えられる。**

この考え方は、その後に出てくる文ベクトルモデル、双塔検索モデル、LLM embedding API でも、基本的には同じです。

---

## 四、単語ベクトルから文脈依存表現へ

### 4.1 初期の embedding：1つの単語に1つの固定ベクトル

たとえば昔の単語ベクトルでは：

- `bank`

は次のどちらでも同じベクトルでした。

- river bank
- bank account

これだと、多義性の問題が起きます。

### 4.2 文脈依存表現：同じ単語でも文脈で変わる

Transformer の時代になると、  
単語の表現は完全に固定ではなく、文脈に応じて変化するようになりました。

つまり：

- 金融の文脈にある `bank` のベクトル
- 川岸の文脈にある `bank` のベクトル

は異なり得ます。

これが文脈依存表現の大きな進歩の1つです。

![文脈依存表現で多義語を解消する図](/img/course/ch07-contextual-embedding-sense-map.png)

:::tip 図の見方
この図では `bank` だけに注目してください。`bank account` では金融の概念に近くなり、`river bank` では地理の概念に近くなります。Transformer の文脈依存表現により、同じ token が常に同じ座標を持つわけではなくなります。
:::

### 4.3 簡単な文脈のシミュレーション

次の例は本物の Transformer ではありません。  
でも、「同じ単語でも状況でベクトルが変わる」という感覚をつかむ助けになります。

```python
base_bank = [0.50, 0.50, 0.50]
finance_context = [0.30, -0.10, 0.20]
river_context = [-0.20, 0.25, -0.10]

bank_in_finance = [a + b for a, b in zip(base_bank, finance_context)]
bank_in_river = [a + b for a, b in zip(base_bank, river_context)]

print("金融の bank:", [round(x, 2) for x in bank_in_finance])
print("川岸の bank:", [round(x, 2) for x in bank_in_river])
```

ここで大事なのは、実際のモデルを再現することではなく、  
次を覚えることです。

- 静的 embedding：1つの単語に1つのベクトル
- 文脈依存表現：同じ単語でも文脈によって変わる

---

## 五、Embedding は実際のプロジェクトで何に使うのか？

### 5.1 検索と RAG

質問と文書の両方をベクトルに変換すると、  
次のことができます。

- 類似度による検索

これは多くの RAG システムの基礎です。

### 5.2 意味クラスタリングと重複排除

2つのテキストベクトルが近ければ、  
意味も近いことが多いです。

これは次の用途に使えます。

- テキストクラスタリング
- FAQ の統合
- 近似重複の検出

### 5.3 下流タスクの入力特徴量

多くの分類、マッチング、ランキングのタスクでは、まずテキストを embedding に変換し、  
その上で head を学習したり、類似度スコアを計算したりします。

---

## 六、Embedding で最も誤解されやすい点

### 6.1 誤解1：ベクトルが近ければ必ず同義語

そうとは限りません。  
むしろ次を意味することが多いです。

- 分布が近い
- 使われ方が関係している

### 6.2 誤解2：文ベクトルは単純なほど良い

単語ベクトルの平均は分かりやすいですが、  
次のような情報を失いやすいです。

- 順序
- 否定
- 長距離依存

### 6.3 誤解3：embedding があれば言語を理解したことになる

embedding はあくまで表現の層です。  
本当の理解には、さらに次が必要です。

- 文脈のモデリング
- タスクの目的
- 学習データ

---

## まとめ

この節で一番大事なのは、公式をいくつか覚えることではなく、次の見方を持つことです。

> **Embedding の本質的な価値は、離散的な token を、比較できて、組み合わせられて、意味関係を反映できるベクトル空間に変えること。**

この主線をつかめれば、  
次に学ぶ次の内容も自然に理解しやすくなります。

- 文ベクトル
- 検索モデル
- RAG
- 文脈依存表現

---

## 練習

1. 例の単語ベクトルを自分で少し変えて、どの単語が近くなったり遠くなったりするか観察してみましょう。
2. なぜ単語ベクトルの平均は最初の直感をつかむのに役立つけれど、すべての意味現象を表すのには向かないのでしょうか？
3. 自分の言葉で説明してみましょう：静的 embedding と文脈依存表現の最大の違いは何ですか？
4. 考えてみましょう：FAQ 検索を作るとき、embedding はまず何を助けてくれるでしょうか？
