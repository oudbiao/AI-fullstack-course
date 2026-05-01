---
title: "1.3 文書処理とベクトル化"
sidebar_position: 2
description: "クリーニング、分割、重なり、メタデータから簡単なベクトル化まで、RAG の前処理パイプラインがなぜ効果の上限を決めるのかを理解します。"
keywords: [chunking, 文書分割, ベクトル化, metadata, RAG preprocessing]
---

# 文書処理とベクトル化

![文書解析とベクトル化のフローチャート](/img/course/document-processing-vectorization-ja.png)

## 学習目標

この節を終えると、あなたは次のことができるようになります。

- なぜ RAG の効果が前処理に大きく左右されるのかを理解する
- 文書クリーニング、分割、重なり、メタデータの感覚をつかむ
- 簡単に動く分割と検索の例を自分で書く
- 「ベクトル化」が何をしているのかを理解する

---

## 一、なぜ RAG は「文書をそのまま入れる」だけではだめなのか？

実際の文書は、たいてい長くて、雑然としていて、情報が混ざっています。

たとえば PDF には次のようなものが含まれます。

- ヘッダーとフッター
- 目次
- 空行
- 見出しの階層
- 表
- 重複したテキスト

これをそのままモデルに入れると、よくある問題は次のとおりです。

- コンテキストが長すぎて入りきらない
- 重要な情報が長文に埋もれて、検索しにくい
- ノイズが多く、検索品質が落ちる

つまり、文書処理は実は次の作業をしています。

> **資料を、モデルが見つけやすく、使いやすい知識ブロックに整えること。**

---

## 二、文書処理のよくある 4 ステップ

### 1. クリーニング

不要なノイズを取り除きます。たとえば：

- 余分な空白
- ページ番号
- 重複した見出し

### 2. 分割（Chunking）

長文を、検索しやすい小さな断片に分けます。

### 3. メタデータ付与

各ブロックに次のような情報を付けます。

- 元ファイル
- 見出し
- ページ番号
- タグ

### 4. ベクトル化

テキストブロックを、類似度検索に使えるベクトルへ変換します。

---

## 三、なぜ分割がそんなに重要なのか？

分割のサイズは、「勉強ノートを 1 枚のカードにどれくらい書くか」によく似ています。

- 大きすぎる: 一度に情報が多すぎて、検索が不正確になる
- 小さすぎる: 文脈が足りず、回答が途切れやすい

唯一の正解はありませんが、必ずタスクに合わせて調整する必要があります。

たとえるなら：

> 開いた試験のためのノートを作るとき、教科書 1 冊をそのまま超巨大ポスター 1 枚に貼りつけたり、逆に 1 文字ずつ紙切れに切り分けたりはしませんよね。

![Chunk サイズと overlap のトレードオフ図](/img/course/ch08-chunk-size-overlap-tradeoff-map-ja.png)

:::tip 読み方のヒント
この図は中央の「証拠の完全性」から見るのがポイントです。chunk が大きすぎると検索が鈍くなり、小さすぎると証拠が分断されます。overlap の役割は、境界部分の情報に少し余裕を持たせることです。
:::

---

## 四、最小限で動く分割の例

```python
import re

text = """
返金ポリシー：
コース購入後 7 日以内で、学習進捗が 20% 未満なら、返金を申請できます。
7 日を過ぎると、無条件返金はできません。

証明書の説明：
すべての必修項目を完了し、修了テストに合格すると、修了証を取得できます。

学習順序：
まず Python、データ分析、機械学習を学び、その後に深層学習と大規模モデルの段階へ進むのがおすすめです。
""".strip()

def split_into_sentences(text):
    parts = re.split(r"[。！？\\n]+", text)
    return [p.strip() for p in parts if p.strip()]

sentences = split_into_sentences(text)
print("文のリスト:")
for s in sentences:
    print("-", s)
```

もし文がすでにかなり短いなら、そのまま 1 文を 1 chunk として使えます。  
ただし実際には、複数の文をまとめて 1 つのブロックにすることのほうが多いです。

---

## 五、重なり付きの分割

なぜ多くの RAG システムで chunk overlap を使うのでしょうか？

情報がちょうど chunk の境界にまたがることがあるからです。  
少し重なりを持たせると、「文脈が切れる」確率を下げられます。

```python
def chunk_sentences(sentences, chunk_size=2, overlap=1):
    chunks = []
    start = 0
    while start < len(sentences):
        end = start + chunk_size
        chunk = "。".join(sentences[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
        if chunk_size - overlap <= 0:
            raise ValueError("chunk_size は overlap より大きくなければなりません")
    return chunks

chunks = chunk_sentences(sentences, chunk_size=2, overlap=1)

print("分割結果:")
for i, chunk in enumerate(chunks):
    print(f"[chunk {i}] {chunk}")
```

---

## 六、なぜメタデータが重要なのか？

初心者はテキスト内容ばかりに注目して、メタデータを見落としがちです。  
でもメタデータは、検索結果と表示体験にそのまま影響します。

1 つの chunk には、よく次のようなメタデータがあります。

- `source`: どのファイルから来たか
- `section`: どの節に属するか
- `page`: 何ページ目か
- `tags`: どんなテーマか

たとえば：

```python
chunks_with_meta = [
    {
        "text": "コース購入後 7 日以内で、学習進捗が 20% 未満なら、返金を申請できます",
        "source": "course_policy.pdf",
        "section": "返金ポリシー",
        "page": 3
    },
    {
        "text": "すべての必修項目を完了し、修了テストに合格すると、修了証を取得できます",
        "source": "course_policy.pdf",
        "section": "証明書の説明",
        "page": 5
    }
]

for item in chunks_with_meta:
    print(item)
```

メタデータの価値は次のとおりです。

- フィルタリングしやすい
- 出典を示しやすい
- 後から UI に表示しやすい

---

## 七、目標が「知識ベース駆動の教材生成アシスタント」なら、分割はもう一段考える

このタイプのプロジェクトは、普通の FAQ 質問応答と大きく違います。

- 単に「関連箇所を見つけたい」だけではない
- さらに資料を「知識点 / 例題 / 練習問題」に再構成したい

なので、最初に作るときは、分割を単なる長さだけで考えないでください。  
「内容の種類」でも考える必要があります。

よくある安定したデフォルトの考え方は、次のとおりです。

| 内容タイプ | どう分けるのが向いているか |
|---|---|
| 概念の定義 | 定義と式を丸ごと残し、途中で切らない |
| 例題の解説 | 問題文と解法をできるだけ同じブロックに入れる |
| 練習問題 | 1 問 1 ブロックにして、後で個別に取り出しやすくする |
| 章のまとめ | 見出しと要点リストを残す |

この表が大事なのは、初心者に次のことを気づかせてくれるからです。

> **分割は固定のテキスト操作ではなく、その先の生成目的のためにある。**

![教材知識ブロックのメタデータ schema 図](/img/course/ch08-courseware-chunk-metadata-schema-map-ja.png)

:::tip 読み方のヒント
教材生成で一番困るのは、「文字は見つかったのに、どこに入れればいいかわからない」ことです。図を見るときは、`topic`、`content_type`、`source_origin`、`page_or_slide` の 4 つに注目してください。これらが、後で知識点、例題、練習問題として安定して組み立てられるかを左右します。
:::

## 八、教材生成プロジェクトらしい知識ブロックの例

```python
courseware_chunks = [
    {
        "topic": "割引の文章題",
        "content_type": "concept",
        "section": "知識点の確認",
        "page": 1,
        "text": "割引 = 定価 × 割引率",
    },
    {
        "topic": "割引の文章題",
        "content_type": "example",
        "section": "例題の解説",
        "page": 2,
        "text": "商品の定価が 100 円で、2 割引の後の値段はいくらですか？",
    },
    {
        "topic": "割引の文章題",
        "content_type": "exercise",
        "section": "授業内練習",
        "page": 3,
        "text": "服の定価が 80 円で、3 割引の後はいくらですか？",
    },
]

for item in courseware_chunks:
    print(item["content_type"], "->", item["text"])
```

この例で初心者が特に注目すべき点は次のとおりです。

- 同じテーマでも、知識ブロックは概念・例題・練習に分けるのがよい
- そうすると、後で Word 教材を生成するときに、どの内容をどの欄に入れるべきかシステムが判断しやすい

---

## 九、ベクトル化は実際には何をしているのか？

ベクトル化の核心は、テキストブロックを「意味空間」に写すことです。

そうすると、クエリも文書ブロックもベクトルになり、あとは類似度を比べられます。

コードがそのまま動くように、まずは超簡単な単語袋ベクトルでこの流れを真似してみましょう。

```python
import math
import re
from collections import Counter

chunks = [
    "コース購入後 7 日以内で、学習進捗が 20% 未満なら、返金を申請できます",
    "すべての必修項目を完了し、修了テストに合格すると、修了証を取得できます",
    "まず Python、データ分析、機械学習を学び、その後に深層学習と大規模モデルの段階へ進むのがおすすめです"
]

def tokenize(text):
    return re.findall(r"[\\w\\u4e00-\\u9fff]+", text.lower())

vocab = sorted(set(token for chunk in chunks for token in tokenize(chunk)))
vocab_index = {word: idx for idx, word in enumerate(vocab)}

def vectorize(text):
    vec = [0] * len(vocab)
    counts = Counter(tokenize(text))
    for word, count in counts.items():
        if word in vocab_index:
            vec[vocab_index[word]] = count
    return vec

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

query = "返金を申請するにはどうすればいいですか"
query_vec = vectorize(query)

scores = []
for chunk in chunks:
    score = cosine_similarity(query_vec, vectorize(chunk))
    scores.append((score, chunk))

scores.sort(reverse=True)
for score, chunk in scores:
    print(round(score, 4), "->", chunk)
```

これが、最小限の「検索」の原理です。

---

## 十、実際のプロジェクトでは、もっと複雑になる

本当の RAG システムでは、ベクトル化に単純な単語頻度ではなく、専用の embedding モデルを使うのが一般的です。

でも考え方は同じです。

1. クエリをベクトルにする
2. 文書ブロックをベクトルにする
3. ベクトル空間で最も近いブロックを探す

なので、「ベクトルデータベース」という言葉におびえる必要はありません。  
本質はあくまで類似度検索で、違いは規模が大きく、効率が高いことです。

---

## 十一、文書処理で最も失敗しやすいポイント

### 1. chunk が大きすぎる

再現率が下がり、コンテキストを無駄にします。

### 2. chunk が小さすぎる

情報が不完全になり、モデルが見る断片がバラバラになります。

### 3. クリーニングしすぎる

見出し、階層、表の構造など、価値のある情報まで消してしまいます。

### 4. メタデータがない

後から「答えはどこから来たのか」を説明しにくくなります。

### 5. 長さだけで分割し、タスクに合わせて分けていない

教材生成プロジェクトでは、これにより次の問題が起こります。

- 例題と解法が分断される
- 概念と練習問題が混ざる
- 後で決まった形式の文書に安定して組み立てにくくなる

---

## 文書処理の確認表

文書処理が終わったら、「chunk が何個できたか」だけでなく、その chunk が後続の質問応答を本当に支えられるかを確認してください。

| 確認項目 | 合格の状態 | よくある問題 |
|---|---|---|
| テキストのクリーニング | ヘッダー、フッター、重複空白、意味のないノイズを除去できている | クリーニングしすぎて、見出しや表の構造まで消えている |
| chunk の完全性 | 1 つの chunk で、1 つの事実または 1 つの手順を完結して表せる | 重要な条件が隣の chunk に切れている |
| chunk の粒度 | 正確に再現でき、かつ細かすぎない | 大きすぎて検索が不正確、小さすぎて証拠が不完全 |
| メタデータ | source、section、page、topic、content_type が残っている | 答えの出典を示せない、テーマで絞り込めない |
| サンプル確認 | ランダムに 10 個ほど chunk を目視で確認する | 数だけ見て、品質を見ない |

いちばん実用的なのは、まず「chunk 抜き取り確認表」を作ることです。分割ルールを変えるたびに、chunk をランダムに数件抽出し、検索・引用・表示に向いているかを確認しましょう。

## chunk 品質の抽出確認スクリプト

次のスクリプトは外部ライブラリに依存せず、確認の習慣をつけるためのものです。実際のプロジェクトでは、確認結果を CSV や Markdown に書き出してもよいでしょう。

```python
chunks_with_meta = [
    {
        "id": "policy_001_01",
        "text": "コース購入後 7 日以内で、学習進捗が 20% 未満なら、返金を申請できます",
        "source": "course_policy.pdf",
        "section": "返金ポリシー",
        "page": 3,
        "content_type": "policy",
    },
    {
        "id": "policy_001_02",
        "text": "すべての必修項目を完了し、修了テストに合格すると、修了証を取得できます",
        "source": "course_policy.pdf",
        "section": "証明書の説明",
        "page": 5,
        "content_type": "rule",
    },
]

required_fields = {"id", "text", "source", "section", "page", "content_type"}

for chunk in chunks_with_meta:
    missing = required_fields - set(chunk)
    too_short = len(chunk["text"]) < 10
    too_long = len(chunk["text"]) > 300
    print({
        "id": chunk.get("id"),
        "missing_fields": sorted(missing),
        "too_short": too_short,
        "too_long": too_long,
        "preview": chunk["text"][:40],
    })
```

このスクリプトは意味の品質までは判断してくれませんが、まず次のような基本問題を見つけるのに役立ちます。

- フィールド不足
- chunk が短すぎる
- chunk が長すぎる
- 出典を追跡できない

## 分割戦略の比較記録

新しい分割戦略を試すたびに、結果を同じ形式で記録するのがおすすめです。

| 戦略 | パラメータ | 長所 | 表面化した問題 | 残すか |
|---|---|---|---|---|
| 文ごとに分ける | 1 文 1 chunk | シンプルで、検索精度が高い | 証拠が不完全になりやすい | 短い FAQ のみ向く |
| スライディングウィンドウ | 2～4 文、overlap 1 | 文脈が切れにくい | chunk 数が増える | baseline に向く |
| 見出し階層で分ける | H2/H3 配下を 1 ブロック | 構造を残せる | 長い章は大きくなりがち | 教材や文書に向く |
| 内容タイプで分ける | 概念 / 例題 / 練習を分ける | 教材生成に向く | 解析やラベル付けが必要 | 構造化プロジェクトに向く |

どこから始めればいいか迷ったら、まず「見出し階層 + スライディングウィンドウ」を baseline にして、評価用データセットに合わせて調整するのがおすすめです。

## まとめ

この節でいちばん大事なのは次の理解です。

> **RAG の前処理は脇役ではなく、効果の上限を決める重要な要素である。**

検索がうまくいかなければ、生成が安定することはほとんどありません。  
だからこそ、文書クリーニング、分割、メタデータ、ベクトル化は、どれも丁寧に設計する必要があります。

---

## 練習

1. `chunk_size` と `overlap` を調整して、分割結果がどう変わるか観察してみましょう。
2. `chunks` に、返金とはまったく関係のない文を 1 つ追加して、検索スコアの並びをもう一度見てみましょう。
3. 考えてみましょう。1 つの規約が 2 つの段落にまたがる場合、どのように chunk を設計すれば、重要な情報を切り断ちにくいでしょうか？
4. あなたの目標が教材生成なら、概念、例題、練習問題を同じ分割方法でまとめてしまうのがなぜ向かないのか、考えてみましょう。
