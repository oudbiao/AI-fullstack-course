---
title: "11.6.6 Transformers ライブラリ実践"
sidebar_position: 20
description: "tokenizer、config、model から最小の pipeline まで、HuggingFace Transformers の核心インターフェースをオフラインでどう使うかをしっかり身につけます。"
keywords: [transformers, HuggingFace, tokenizer, AutoModel, pipeline, config]
---

# 11.6.6 Transformers ライブラリ実践

![Transformers ライブラリの呼び出しチェーン図](/img/course/ch11-transformers-library-call-chain-map-ja.webp)

:::tip 画像の見方
初めて Transformers ライブラリを使うと、API 名に少し混乱しやすいです。まずは Tokenizer、Config、Model、Task Head、Pipeline の呼び出しチェーンに沿って見て、それぞれの役割を理解してから具体的なクラス名を調べると、ずっと分かりやすくなります。
:::

:::tip この節の位置づけ
事前学習済みモデルは、概念だけにとどまると「説明はできるけど使えない」状態になりがちです。
この節で解決したいのは、もっと実践的な次の疑問です。

> **`transformers` というライブラリに対して、私はどこから手をつければいいのか？**

できるだけ外部ネットワークからのダウンロードに頼らず、最重要のインターフェースを通していきます。
:::

## 学習目標

- `transformers` ライブラリでよく使う主要オブジェクトの役割を理解する
- tokenizer、config、model、pipeline の違いを区別できるようになる
- オフラインで tokenizer + model の最小サンプルを動かす
- 実際のプロジェクトで「動く」から「保守しやすい」へ進む考え方を理解する

---

## 一、まずはライブラリの主役たちを整理しよう

### `Tokenizer`

テキストを、モデルが扱える数値列に変換する役割です。

### `Config`

モデルの構造パラメータを記述します。たとえば：

- hidden size
- 層数
- ヘッド数

### `Model`

実際に前向き計算を行う本体です。

### `Pipeline`

より高レベルのラッパーで、次の処理をひとまとめにしてくれます。

- 分かち書き
- 前向き計算
- 後処理

より簡単に呼び出せるインターフェースにしてくれます。

一言で覚えるなら：

> tokenizer は入口を担当し、model は計算を担当し、pipeline は全体をつなげる役割です。

---

## 二、なぜ多くの人は最初に `transformers` を使うと混乱するのか？

理由は、このライブラリに二つの見方があるからです。

### 概念の世界

たとえば、あなたは次のようなことを知っています。

- BERT は encoder-only
- GPT は decoder-only

### ツールの世界

一方で、次のようなものにも出会います。

- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `pipeline`
- `from_pretrained`

初学者がよくつまずくのは、

> 名前はたくさんあるのに、それぞれが何を解決するのか分からない。

という点です。

なので、この節の核心はインターフェースを暗記することではなく、呼び出し順序の地図を作ることです。

---

## 三、まずはオフラインで最小の tokenizer を作る

:::info 実行環境
```bash
pip install torch transformers
```
:::

### なぜ既製モデルをそのままダウンロードしないのか？

教材では、外部ネットワークがなくても動くようにしておくことが大切だからです。
そこでここでは、手作業でとても小さな `vocab.txt` を用意して、tokenizer が何をしているのかを本当に理解できるようにします。

### 実行可能なサンプル

```python
from pathlib import Path
from tempfile import TemporaryDirectory
from transformers import BertTokenizer

vocab_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "我", "愛", "自", "然", "語", "言", "処", "理", "北", "京"
]

with TemporaryDirectory() as tmpdir:
    vocab_path = Path(tmpdir) / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

    tokenizer = BertTokenizer.from_pretrained(tmpdir)

    encoded = tokenizer("我愛自然言語処理", return_tensors="pt")

    print(encoded)
    print("tokens:", tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
```

想定出力：

```text
{'input_ids': tensor([[ 2,  5,  6,  7,  8, 10,  9, 11, 12,  3]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
tokens: ['[CLS]', '我', '愛', '自', '然', '言', '語', '処', '理', '[SEP]']
```

一時ディレクトリを使うので、練習後にプロジェクト直下へ `vocab.txt` が残りません。`input_ids` は token 番号で、`attention_mask` は有効な位置を示します。

### このコードで学ぶこと

このコードが教えているのは、次のことです。

- tokenizer は魔法のブラックボックスではない
- 本質は「語彙 + 分割ルール + エンコードルール」
- 出力で特に重要なのは `input_ids` と `attention_mask`

---

## 四、次に最小の BERT モデルをオフラインで作る

### なぜランダム初期化のモデルを使うのか？

今の目的は精度を追うことではなく、

> `transformers` ライブラリの model オブジェクトが、入力をどう受け取り、出力をどう返すのかをきちんと理解すること

だからです。

### 実行可能なサンプル

```python
import torch
from transformers import BertConfig, BertModel

config = BertConfig(
    vocab_size=15,
    hidden_size=32,
    num_hidden_layers=2,
    num_attention_heads=4,
    intermediate_size=64
)

model = BertModel(config)

input_ids = torch.tensor([[2, 5, 6, 7, 8, 9, 10, 11, 12, 3]])
attention_mask = torch.ones_like(input_ids)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

print("last_hidden_state shape:", outputs.last_hidden_state.shape)
print("pooler_output shape    :", outputs.pooler_output.shape)
```

想定出力：

```text
last_hidden_state shape: torch.Size([1, 10, 32])
pooler_output shape    : torch.Size([1, 32])
```

このランダムモデルは、精度ではなくインターフェースを学ぶためのものです。最初の shape は、1 件のサンプル、10 個の token 位置、各位置 32 次元の隠れ表現を意味します。

### 本当に理解すべきポイント

- `input_ids` は token の番号
- `attention_mask` はどの位置が有効かをモデルに伝える
- `last_hidden_state` は各位置の文脈化された表現
- `pooler_output` は文全体の表現の一種に近いもの

このあたりを理解しておくと、後で分類 head、マッチング head、生成 head をつなぐときにぐっと楽になります。

---

## 五、tokenizer と model をつないでみる

### 実行可能なサンプル

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import torch
from transformers import BertTokenizer, BertConfig, BertModel

vocab_tokens = [
    "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
    "我", "愛", "自", "然", "語", "言", "処", "理"
]

with TemporaryDirectory() as tmpdir:
    vocab_path = Path(tmpdir) / "vocab.txt"
    vocab_path.write_text("\n".join(vocab_tokens), encoding="utf-8")

    tokenizer = BertTokenizer.from_pretrained(tmpdir)

    config = BertConfig(
        vocab_size=len(vocab_tokens),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64
    )
    model = BertModel(config)

    batch = tokenizer(["我愛自然言語処理", "我愛言語"], padding=True, return_tensors="pt")
    outputs = model(**batch)

    print("input_ids shape        :", batch["input_ids"].shape)
    print("attention_mask shape   :", batch["attention_mask"].shape)
    print("last_hidden_state shape:", outputs.last_hidden_state.shape)
```

想定出力：

```text
input_ids shape        : torch.Size([2, 10])
attention_mask shape   : torch.Size([2, 10])
last_hidden_state shape: torch.Size([2, 10, 32])
```

tokenizer は短い文を長い文と同じ長さまで padding します。そのため、batch はきれいな長方形の tensor になります。

![Tokenizer から BERT の tensor 形状図](/img/course/ch11-transformers-tokenizer-batch-shape-result-map-ja.webp)

:::tip tensor shape の読み方
最初の 2 次元は batch サイズと padding 後の系列長です。`last_hidden_state` は、各 token 位置に 32 次元の文脈ベクトルを足した形になります。
:::

### これが基本の本当の呼び出しチェーン

実際のプロジェクトで最もよくある基本フローは、だいたい次の通りです。

1. テキスト -> tokenizer
2. tokenizer -> tensor
3. tensor -> model
4. model の出力 -> 後処理または task head

ここまでで、このチェーンはすでに自分で動かせるようになりました。

---

## 六、`Auto*` 系インターフェースは何のためにあるのか？

### なぜライブラリには `AutoModel` がたくさんあるのか？

`transformers` は、あなたがたくさんのモデル型を手書きで判定しなくて済むように設計されています。

たとえば：

- `AutoTokenizer`
- `AutoModel`
- `AutoModelForSequenceClassification`
- `AutoModelForCausalLM`

これらの設計目標は、

> モデル名や config を与えれば、適切なクラスを自動で選ぶ

ということです。

### オフラインの `AutoModel.from_config` の例

```python
from transformers import AutoModel, BertConfig

config = BertConfig(
    vocab_size=20,
    hidden_size=16,
    num_hidden_layers=1,
    num_attention_heads=4,
    intermediate_size=32
)

model = AutoModel.from_config(config)
print(type(model))
```

想定出力：

```text
<class 'transformers.models.bert.modeling_bert.BertModel'>
```

`AutoModel` は BERT の config を読み取り、それに合う BERT モデルクラスを自動で作ります。実際の事前学習済み checkpoint を名前で読み込むときも、考え方は同じです。

これが示しているのは次の点です。

- `AutoModel` は必ずしもネットワークからのダウンロードが必要ではない
- 本質は「config に基づいて正しいモデルを自動生成する」こと

---

## 七、`pipeline` は学ぶ価値があるのか？

### 価値はある。ただし、使いどころを知ることが大事

`pipeline` の良い点：

- すぐに使える
- デモを素早く作れる
- 定型コードを減らせる

特に向いているのは：

- 学習
- すばやい検証
- 小さな実験

### ただし、実務では pipeline だけに頼れない

実際のプロジェクトでは、次のような制御も必要になることが多いです。

- batch
- device
- 出力形式
- ログ
- エラー処理

だから、成熟したやり方としては通常、

- まず `pipeline` を使えるようになる
- 同時に、下層の tokenizer + model の呼び出しチェーンも理解する

という流れになります。

---

## 八、Transformers ライブラリでよく使う task head

よく使うものをざっくり覚えておきましょう。

| インターフェース | 向いている用途 |
|---|---|
| `AutoModel` | 基本表現だけ取り出す |
| `AutoModelForSequenceClassification` | テキスト分類 |
| `AutoModelForTokenClassification` | シーケンスラベリング |
| `AutoModelForQuestionAnswering` | 抽出型質問応答 |
| `AutoModelForCausalLM` | 生成タスク |

裏側の考え方はとてもシンプルです。

> 同じ backbone モデル + 異なる task head。

---

## 九、初学者がよくハマる落とし穴

### `pipeline` しか使えず、下層の呼び出しが分からない

これだと、実務の場面で止まりやすくなります。

### tokenizer の出力フィールドを理解していない

最低でも次は読めるようにしておきましょう。

- `input_ids`
- `attention_mask`

### 「モデルの概念」と「ライブラリのインターフェース」を混同する

次の違いを分けて考えられるようにしましょう。

- BERT / GPT はモデルの系統
- `AutoModel` / `pipeline` はライブラリのインターフェース

---

## 残す証拠

このページを終えたら、この evidence card を残します。

```text
model_choice: BERT, GPT, T5, Transformers pipeline, or other pretrained baseline
tokenizer_output: ids, masks, decoded text, or batch shape
task_result: classification, generation, extraction, or text-to-text output
failure_check: wrong model family, token limit, domain mismatch, cost, or latency
Expected_output: model call result plus a short choice rationale
```

## まとめ

この節で最も大切なのは、API を暗記できるかどうかではありません。大事なのは、次のことをはっきりさせることです。

> **Transformers ライブラリの核心的な呼び出しチェーンは、tokenizer がテキストを tensor にエンコードし、model が tensor に対して前向き計算を行い、最後に task head または後処理で結果を得る、という流れである。**

この流れが分かれば、その後に分類、抽出、生成、あるいは fine-tuning を行うときも、考え方がかなり安定します。

---

## 練習

1. この節の mini vocab を変更して、自分の単語をいくつか追加し、tokenizer の出力がどう変わるか見てみましょう。
2. `BertConfig` の `hidden_size` を 64 に変えて、出力 shape がどう変わるか確認してみましょう。
3. 自分の言葉で説明してみましょう：なぜ `transformers` ライブラリを学ぶとき、`pipeline` だけでは不十分なのでしょうか？
4. 考えてみましょう：テキスト分類をしたい場合、まず探すべきなのは `AutoModel` でしょうか、それとも `AutoModelForSequenceClassification` でしょうか？

<details>
<summary>参考解答と解説</summary>

1. mini vocab を変えると token IDs が変わります。vocabulary に語がない場合は unknown token が出ることもあります。
2. `hidden_size` を 64 にすると hidden representation の次元が変わります。sequence length や batch size が変わるわけではありません。
3. `pipeline` は便利ですが、実プロジェクトを debug するには tokenizer、config、model class、tensor、label、evaluation も理解する必要があります。
4. text classification では、分類 head が必要なら `AutoModelForSequenceClassification` から始めます。head を自作するなら `AutoModel` を使います。

</details>
