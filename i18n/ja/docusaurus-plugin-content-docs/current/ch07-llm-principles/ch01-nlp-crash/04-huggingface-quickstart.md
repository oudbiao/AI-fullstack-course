---
title: "1.5 HuggingFace クイックスタート"
sidebar_position: 4
description: "tokenizer、config、model、batch から forward の出力までを通して、HuggingFace で最もよく使うワークフローを理解し、ネットワークからのダウンロードに依存しない実行可能な入門例を示します。"
keywords: [HuggingFace, transformers, tokenizer, model, config, forward, batch]
---

# HuggingFace クイックスタート

:::tip この節の位置づけ
多くの初心者は、HuggingFace に初めて触れるとき、次のような名前に混乱しがちです。

- `AutoTokenizer`
- `AutoModel`
- `pipeline`
- `config`
- `forward`

一見すると API がたくさんあるように見えます。  
でも、中心となる流れを取り出すと、実はとても安定しています。

> **テキスト -> tokenizer -> input ids / mask -> model.forward -> hidden states / logits**

この節の目標は、この流れをはっきり理解することです。
:::

## 学習目標

- HuggingFace で最もよく使う入力から出力までの流れを理解する
- tokenizer、config、model、batch がそれぞれ何を担当するかを区別できるようになる
- オンラインダウンロードに依存しない、最小の `transformers` サンプルを読めるようになる
- 今後、公式サンプルやリポジトリのコードを読むときの、最初のなじみを作る

---

## 一、HuggingFace は実際に何を助けてくれるの？

### 1.1 これは「1つのモデル」ではなく、1つのエコシステムです

多くの人は HuggingFace を次のように誤解しがちです。

- とても強力なモデルプラットフォーム

より正確に言うと、モデル開発を中心にしたツールのエコシステム全体です。代表的なものには次があります。

- `transformers`
- `datasets`
- `tokenizers`
- `peft`

この中で、最もよく触れるのは次の3つです。

- tokenizer
- model
- config

### 1.2 最も一般的なワークフローは数ステップしかありません

分類でも、生成でも、特徴抽出でも、  
最も重要な呼び出しの流れは、たいてい次のようになります。

1. tokenizer でテキストを `input_ids` に変換する
2. `attention_mask` を準備する
3. batch を model に渡す
4. 出力から hidden states、logits、または生成結果を取り出す

![HuggingFace 標準ワークフローのオブジェクト関係図](/img/course/ch07-huggingface-workflow-object-map.png)

:::tip 図の見方
この図を見るときは、HuggingFace を標準化された実験台だと考えてください。`tokenizer` はテキストをテンソルに変換し、`config` はモデル構造を表し、`model.forward` が計算を実行します。その後、出力は hidden states、logits、または生成結果になります。API 名はたくさんありますが、主線は実はとても安定しています。
:::

この流れが頭の中でつながると、  
多くの例はもう複雑に見えなくなります。

### 1.3 たとえで言うと：標準化された実験台を組み立てる感じです

HuggingFace は、実験台の標準部品のように考えられます。

- tokenizer はサンプル前処理器のようなもの
- config はモデルの設計図のようなもの
- model は実際に計算を行う機械のようなもの
- batch は一度に入れるサンプルの束のようなもの

その価値は次の点にあります。

- インターフェースの統一
- 重複作業の削減
- モデルやタスクをより速く試せること

---

## 二、まずはよく出るオブジェクトを整理しよう

### 2.1 Tokenizer：テキストをモデル入力に変える

通常、次の処理を担当します。

- 分かち書き
- token -> id 変換
- padding
- truncation

出力でよく見るフィールドは次の2つです。

- `input_ids`
- `attention_mask`

### 2.2 Config：モデル構造の設計図

config は主に次の内容を表します。

- hidden size
- 層数
- ヘッド数
- 語彙サイズ

これは「モデルがどんな形をしているか」の説明書だと考えられます。

### 2.3 Model：実際に forward を実行する部分

model は config に基づいてニューラルネットワークを構築し、  
テンソル入力を受け取って、次のようなものを出力します。

- `last_hidden_state`
- `pooler_output`
- `logits`

タスクによって出力は少し変わりますが、  
基本的な考え方は同じです。

### 2.4 Batch：なぜ毎回 padding が必要なのか

一括で扱うテキストは長さがバラバラだからです。  
モデルは通常、入力テンソルの形をそろえる必要があるため、次のことを行います。

- 短い文を埋める
- どの位置が本物の token かを mask で知らせる

---

## 三、ダウンロード不要でそのまま実行できる `transformers` の例を見てみよう

このコードには、特に重要な特徴がいくつかあります。

- ネットワークからのモデルダウンロードに依存しない
- ローカルの `BertConfig` を使って小さなモデルをランダム初期化する
- 自分でとても小さな語彙表を用意する
- 2つの文を batch にしてモデルへ渡す

つまり、HuggingFace の基本の流れを一通り動かして確認できます。

:::info 実行メモ
```bash
pip install torch transformers
```
:::

```python
import torch
from transformers import BertConfig, BertModel

vocab = {
    "[PAD]": 0,
    "[CLS]": 1,
    "[SEP]": 2,
    "[UNK]": 3,
    "reset": 4,
    "password": 5,
    "refund": 6,
    "order": 7,
    "please": 8,
    "help": 9,
}


def tokenize(text):
    return text.lower().split()


def encode(text, max_length=6):
    tokens = ["[CLS]"] + tokenize(text) + ["[SEP]"]
    input_ids = [vocab.get(token, vocab["[UNK]"]) for token in tokens][:max_length]
    attention_mask = [1] * len(input_ids)

    if len(input_ids) < max_length:
        pad_count = max_length - len(input_ids)
        input_ids += [vocab["[PAD]"]] * pad_count
        attention_mask += [0] * pad_count

    return input_ids, attention_mask


texts = [
    "please help reset password",
    "refund order",
]

encoded = [encode(text) for text in texts]
input_ids = torch.tensor([item[0] for item in encoded], dtype=torch.long)
attention_mask = torch.tensor([item[1] for item in encoded], dtype=torch.long)

config = BertConfig(
    vocab_size=len(vocab),
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64,
)

model = BertModel(config)
outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("input_ids shape        :", tuple(input_ids.shape))
print("attention_mask shape   :", tuple(attention_mask.shape))
print("last_hidden_state shape:", tuple(outputs.last_hidden_state.shape))
print("pooler_output shape    :", tuple(outputs.pooler_output.shape))
```

### 3.1 このコードはどの順番で読むのがよい？

おすすめの順番は次の通りです。

1. まず `encode` を見て、テキストがどうやって `input_ids` になるかを理解する
2. 次に `BertConfig` を見て、モデル構造がどう定義されているかを知る
3. 最後に `model(...)` の出力 shape を確認する

こうすると、次の3つをすぐにつなげられます。

- テキストの形式
- モデル構造
- forward の出力

### 3.2 なぜここでは `from_pretrained` を使わないのか？

`from_pretrained` は、しばしば重みをダウンロードするためにネット接続が必要です。  
この例では、オフラインでもそのまま動くように、あえて次を使っています。

- `BertConfig(...)`
- `BertModel(config)`

つまり、

- モデルはランダム初期化されています

そのため、実際のタスク予測には使えません。  
でも、HuggingFace の基本的な呼び出しの流れを理解するにはとても適しています。

### 3.3 この例でいちばん見落としやすい点は？

いちばん見落としやすいのは次の点です。

- batch は最初から自然に存在するものではない
- それぞれのテキストを先にエンコードしてから、テンソルにまとめている

この部分までしっかり見えていれば、  
あとで `DataCollator` や `Trainer` のようなラッパーを読むのがかなり楽になります。

---

## 四、実際のプロジェクトでよく見る `from_pretrained` はどんな形？

ネットワーク環境があるなら、  
より一般的な書き方は次のようになります。

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

batch = tokenizer(
    ["please help reset password", "refund order"],
    padding=True,
    truncation=True,
    return_tensors="pt",
)

outputs = model(**batch)
print(outputs.last_hidden_state.shape)
```

このコードは、前のオフライン版と本質的には同じことをしています。  
違いはただ一つです。

- tokenizer とモデル重みを Hub から直接取得している

なので、先ほどのオフライン例は次のように考えるとよいです。

- ブラックボックスを分解して見る

そして、この `from_pretrained` の例は次のように考えられます。

- 公式のラッパーを使って素早く始める

---

## 五、なぜ HuggingFace は入門と実験にとても向いているのか？

### 5.1 インターフェースが統一されているから

内部構造が違っていても、多くのモデルは HuggingFace では似たインターフェースに従います。

- tokenizer がテキスト入力を担当する
- model が forward を担当する

そのため、モデルを切り替えるときの負担がかなり小さくなります。

### 5.2 エコシステムが豊富だから

このあと、次のようなものにも触れることになります。

- `AutoModelForSequenceClassification`
- `AutoModelForCausalLM`
- `Trainer`
- `DataCollator`

これらはすべて、この基本の流れの上に成り立っています。

### 5.3 「まず試して、あとで深く理解する」に向いているから

多くの場合、最初から全部をゼロから実装するのではなく、  
まず標準的なインターフェースを動かし、  
そのあと少しずつ次を理解していきます。

- tokenizer
- attention mask
- logits
- generation config

これが、HuggingFace が学習の入口として価値が高い理由でもあります。

---

## 六、よくある落とし穴

### 6.1 誤解その1：`from_pretrained` が動けば、本当にモデルを理解したことになる

動かせるようになるのは始まりにすぎません。  
本当に理解するには、さらに次を知る必要があります。

- 入力テンソルはどんな形か
- 出力フィールドは何を意味するか
- tokenizer とモデルが合っているか

### 6.2 誤解その2：`attention_mask` を無視する

padding があるのに mask を付けないと、  
モデルが埋めた部分を本当の内容として扱ってしまうことがあります。

### 6.3 誤解その3：ランダム初期化モデルと事前学習済みモデルを同じだと思う

この節のオフライン例は、あくまでインターフェース理解のためのものです。  
実際にタスク能力を持つのは、通常次のようなモデルです。

- 事前学習済み重みを読み込んだモデル

---

## まとめ

この節で一番大事なのは、HuggingFace のクラス名をたくさん覚えることではありません。  
最も基本的な流れを、きちんと一つにつなげることです。

> **テキストはまず tokenizer を通って `input_ids` と `attention_mask` になり、次に config が構造を定義し、model が forward を実行し、最後に hidden states かタスク結果が出力される。**

この流れさえ整理できれば、  
今後、公式サンプル、サードパーティのリポジトリ、学習スクリプトを見ても、表面的な API に怖がらなくなります。

---

## 練習

1. サンプルの `max_length` を小さくして、padding と truncation の変化を観察してみましょう。
2. なぜこの節では `BertConfig + BertModel` を使い、直接 `from_pretrained` を使わなかったのでしょうか？
3. 自分の言葉で、tokenizer、config、model がそれぞれ何を担当するか説明してみましょう。
4. batch に `input_ids` はあるのに `attention_mask` がない場合、まず何の問題を疑いますか？
