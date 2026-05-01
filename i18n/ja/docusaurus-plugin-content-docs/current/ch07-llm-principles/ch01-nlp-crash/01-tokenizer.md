---
title: "1.2 分詞とTokenizer"
sidebar_position: 1
description: "「なぜモデルは文字をそのまま読めないのか」から始めて、文字単位・単語単位・サブワード単位の分かれ方、そして padding、truncation、special tokens が実務でなぜ重要なのかを理解します。"
keywords: [tokenizer, tokenization, subword, BPE, wordpiece, padding, truncation]
---

# 分詞とTokenizer

![Tokenizer サブワード分割フロー図](/img/course/tokenizer-subword-flow.png)

:::tip この節の位置づけ
多くの人は、初めて大規模言語モデルを学ぶとき、モデルの構造ばかりに注目しがちです。  
でも、実際にテキストをモデルへ入れる前には、どうしても通らなければならない関門があります。

> **文字列を、モデルが処理できる単位にどう切り分けるのか？**

これが tokenizer です。

ここを理解しないまま進むと、あとで出てくる次のような言葉が、

- `input_ids`
- `attention_mask`
- context length
- token コスト

全部、バラバラの専門用語に見えてしまいます。

この節の目的は、tokenizer を「黒箱の道具」から、その本質的な役割へ戻して理解することです。
:::

## 学習目標

- なぜモデルは生の文字列をそのまま食べられないのかを理解する
- 文字単位・単語単位・サブワード単位の分かれ方の違いを区別する
- special tokens、padding、truncation が実務で果たす役割を理解する
- 実行可能な例を通して、tokenizer がテキストを `input_ids` に変える流れを理解する

---

## 一、なぜモデルは文字をそのまま読めないのか？

### 1.1 モデルが最終的に扱うのは、文字そのものではなく数字です

ニューラルネットワークが本質的に扱えるのは、数値の tensor です。  
一方で、人間がモデルに入力するのは通常、次のようなものです。

- 中国語の文
- 英語の段落
- コード
- 記号や絵文字が混ざった文字列

モデルは「返金」「password」「hello」といった単語の見た目を、そのままでは認識できません。  
まず次の2段階が必要です。

1. テキストを token に分ける
2. token を整数 id に対応付ける

つまり tokenizer がやっているのは、ただの「単語分割」ではなく、

> **人間の言語を、モデルが扱える離散シーケンスへ変換するための最初のインターフェース**

なのです。

### 1.2 たとえるなら、文章を機械が番号を付けられるブロックに翻訳する作業です

tokenizer は、倉庫の管理人のようなものだと考えると分かりやすいです。

元の文章は、まだ整理されていない大量の荷物のようなものです。  
tokenizer はまず、次のことを決めます。

- 1つ1つのブロックをどの大きさにするか
- それぞれのブロックにどの番号を振るか

その結果、モデルが見るのは「文章」ではなく、

- `[101, 2057, 2024, 2172, 102]`

のような数列になります。

ブロックを細かく切りすぎると長くなりすぎます。  
逆に粗く切りすぎると、知らない単語が増えてしまいます。

---

## 二、もっともよく使われる3つの分け方

### 2.1 文字単位 / 字単位：安定しているが、列が長くなる

いちばん単純な考え方は、

- 1文字、または1字を1 token とする

という方法です。

良い点は：

- OOV 問題がほとんど起きない
- 知らない単語でも分解して表現できる

悪い点は：

- シーケンスが長くなる
- 意味の粒度が細かすぎる
- モデルが単語の意味をまとめるために、より多くの層を使う必要がある

たとえば日本語では：

- “返金ルール” -> `返 / 金 / ル / ー / ル`

### 2.2 単語単位：意味は直感的だが、OOV が深刻になりやすい

別の考え方は、

- 完全な単語を1 token とする

という方法です。

良い点は：

- 粒度が自然
- 単語の意味が分かりやすい

悪い点は：

- 新語、表記ゆれ、固有名詞が非常に多い
- 語彙がとても大きくなりやすい

たとえば英語では：

- `refund` はよく出てきます
- でも `refundability` や `refund-processing` は、簡単に未知語になりやすいです

### 2.3 サブワード単位：現実でいちばんよく使われる折衷案

現代の大規模言語モデルで最もよく使われるのは、

- subword tokenizer

です。

これは、単語を「頻出する断片」に分ける方法です。

たとえば：

- `transformers` -> `transform` + `ers`
- `tokenization` -> `token` + `ization`

この方法の利点は：

- 語彙を無限に大きくしなくてよい
- 新しい単語も、既存の subword の組み合わせで表現できる
- シーケンス長と OOV 問題のバランスがよい

だからこそ、BPE、WordPiece、SentencePiece のような方法がとても重要になります。

![Tokenizer 粒度のトレードオフ図](/img/course/ch07-tokenizer-granularity-tradeoff-map.png)

:::tip 図の見方
この図は左から右へ見るのがおすすめです。文字単位は安定していますが列が最も長く、単語単位は意味が直感的ですが OOV のリスクが高く、サブワード単位は語彙サイズ・列の長さ・未知語への対応の間でバランスを取ります。Tokenizer は「どう切ると見た目がきれいか」ではなく、コストと表現力のバランスを取る仕組みです。
:::

---

## 三、まずは本当に意味が分かる tokenizer の例を動かしてみよう

以下のコードは、実際の本格的な工業用 tokenizer を完全再現するものではありません。  
ただし、次の3つをとても分かりやすく示してくれます。

1. 単語単位の分割
2. サブワード単位の分割
3. token を id に変えること、padding、truncation

```python
import re

vocab = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "refund": 4,
    "policy": 5,
    "reset": 6,
    "password": 7,
    "transform": 8,
    "##er": 9,
    "##s": 10,
    "token": 11,
    "##ization": 12,
    "please": 13,
    "help": 14,
}


def word_tokenize(text):
    return re.findall(r"[A-Za-z]+", text.lower())


def subword_tokenize(word, vocab):
    if word in vocab:
        return [word]

    tokens = []
    start = 0
    while start < len(word):
        matched = None
        for end in range(len(word), start, -1):
            piece = word[start:end] if start == 0 else "##" + word[start:end]
            if piece in vocab:
                matched = piece
                tokens.append(piece)
                start = end
                break
        if matched is None:
            return ["[UNK]"]
    return tokens


def encode(text, vocab, max_length=8):
    words = word_tokenize(text)
    tokens = ["[CLS]"]
    for word in words:
        tokens.extend(subword_tokenize(word, vocab))
    tokens.append("[SEP]")

    token_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens]
    token_ids = token_ids[:max_length]
    attention_mask = [1] * len(token_ids)

    if len(token_ids) < max_length:
        pad_count = max_length - len(token_ids)
        token_ids += [vocab["[PAD]"]] * pad_count
        attention_mask += [0] * pad_count

    return tokens, token_ids, attention_mask


examples = [
    "Please help reset password",
    "Transformers policy",
    "Tokenization refund",
]

for text in examples:
    tokens, token_ids, attention_mask = encode(text, vocab, max_length=10)
    print("-" * 60)
    print("text          :", text)
    print("tokens        :", tokens)
    print("input_ids     :", token_ids)
    print("attention_mask:", attention_mask)
```

### 3.1 このコードで特に見るべき行は？

次の3か所が重要です。

1. `word_tokenize`  
   元の文字列が、まずどのように単語へ切られるかを示しています
2. `subword_tokenize`  
   単語が語彙にないとき、どのように貪欲法で subword に分解するかを示しています
3. `encode`  
   special tokens、padding、truncation がどう追加されるかを示しています

### 3.2 なぜ `Transformers` は複数の subword に分かれるのですか？

語彙に完全な `transformers` が入っていないからです。  
でも次のような部分はあります。

- `transform`
- `##er`
- `##s`

そのため、まだ表現できます。

これが subword tokenizer の重要な強みです。

- 新しい単語が、全部そのまま語彙に入っていなくてもよい

### 3.3 `attention_mask` は何のためにあるのですか？

batch に入る文の長さは、普通はそろっていません。  
そのため、短い文の後ろに `[PAD]` を足して、同じ長さにそろえます。

ただし、モデルはその pad の位置を本当の内容だと思ってはいけません。  
そこで `attention_mask` を使って、次のことを伝えます。

- `1` は本物の token
- `0` は padding

![Tokenizer から input_ids と attention_mask への対応図](/img/course/ch07-tokenizer-inputids-mask-length-map.png)

:::tip 図の見方
この図は4段階に分けて見ると分かりやすいです。まず元の文を tokens に分け、次に `input_ids` に変換し、短い文は `[PAD]` でそろえ、最後に `attention_mask` でどこが本物の内容かをモデルに伝えます。batch のエラーや結果の不自然さは、この流れを理解できていないことが原因のことが多いです。
:::

---

## 四、なぜ tokenizer はコストと性能に直接効くのか？

### 4.1 同じ文でも、細かく切るほど token 数は増える

token が増えるということは、次のような意味です。

- コンテキストを使い切りやすくなる
- 推論コストが上がる
- API の料金も高くなる

つまり tokenizer は、理論だけの話ではありません。  
実務コストにも直結します。

### 4.2 語彙が小さすぎても、大きすぎてもよくありません

語彙が小さすぎると：

- たくさんの単語が細かく切られる

語彙が大きすぎると：

- embedding 表が大きくなる
- 稀な単語が増える
- 学習データの使い方が必ずしも良くならない

実際の tokenizer 設計は、こうした要素の間でバランスを取る作業です。

### 4.3 言語が違うと、難しさも変わります

たとえば：

- 英語は空白があるので、分かち書きしやすい
- 日本語や中国語は空白がないので、切り方の粒度がより重要になる
- コードには、キャメルケース、アンダースコア、記号が混ざる

そのため tokenizer は、学習コーパスの言語的特徴に合わせて調整されることが多いです。

---

## 五、special tokens がいつも出てくるのはなぜですか？

### 5.1 `[CLS]`、`[SEP]`、`[PAD]` は飾りではありません

これらの special token には、普通ははっきりした役割があります。

- `[CLS]`：文全体の表現の開始位置
- `[SEP]`：複数の断片を区切る
- `[PAD]`：batch の長さをそろえる

モデルによって具体的な記号は違うことがありますが、考え方はよく似ています。

### 5.2 Chat モデルの system / user / assistant も、考え方は同じです

Chat モデルの時代になると、次のような特別なマークもよく見かけます。

- `<|system|>`
- `<|user|>`
- `<|assistant|>`

これらも本質的には、モデルに次のことを知らせるための special token です。

- どの発話が誰のものか
- 会話の構造をどう区切るか

つまり chat template も、tokenizer エコシステムの一部なのです。

---

## 六、つまずきやすいポイント

### 6.1 誤解1：tokenizer はただの前処理の細部である

違います。  
次のことに直接影響します。

- token 数
- 語彙サイズ
- OOV の扱い
- 下流のテンプレート形式

### 6.2 誤解2：とにかく切れればよい

本当に大事なのは次の点です。

- 安定して切れるか
- コーパスに合っているか
- 長さと意味の粒度のバランスが取れているか

### 6.3 誤解3：中国語は必ず「単語」で切るのが最善

そうとは限りません。  
多くの現代モデルは次のような方法を使っています。

- 文字単位
- サブワード単位
- SentencePiece のような統一的な分かち方

大事なのは、学習目標とデータ分布です。

---

## まとめ

この節でいちばん大切なのは、BPE や WordPiece という名前を覚えることではありません。  
次の1本の軸をつかむことです。

> **Tokenizer の本質は、元のテキストをモデルが扱える離散単位に分け、語彙サイズ・未知語問題・シーケンス長の間で工学的なトレードオフを取ることです。**

この軸ができると、あとで出てくる

- input ids
- attention mask
- context length
- prompt テンプレート

を、バラバラの概念としてではなく、つながった仕組みとして理解できるようになります。

---

## 練習

1. 例の語彙から `transform` か `##ization` を削除して、どの単語が `[UNK]` に落ちるか見てみましょう。
2. なぜ subword tokenizer は、単語単位と文字単位の折衷案だと言えるのでしょうか。
3. `max_length` を短くして、truncation が出力にどんな影響を与えるか観察してみましょう。
4. もしあなたのコーパスにコードがとても多いなら、tokenizer 設計はまず何でつまずきそうでしょうか。
