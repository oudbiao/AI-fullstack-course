---
title: "2.3.4 プロジェクト：AI API クイック体験"
sidebar_position: 4
description: "AI API を呼び出して人工知能の力を体験する"
---

# 2.3.4 プロジェクト：AI API クイック体験

![AI API リクエスト・レスポンスの流れ図](/img/course/ch02-ai-api-request-response-ja.png)

## プロジェクトの位置づけ

このプロジェクトでは、Python 基礎の段階で、先に大規模モデルの能力を体験します。API Key を使って既存の AI サービスを呼び出し、「モデルを学習させること」と「モデルサービスを呼び出すこと」の違いを理解し、シンプルな AI 会話プログラムを作ります。

## プロジェクトの目標

- AI API とは何か、そして AI モデルとの関係を理解する
- OpenAI などの主流 AI API の呼び出し方を学ぶ
- 対話生成、テキスト分析などの AI 能力を体験する
- シンプルな AI チャットボットを作る

---

## AI API とは？

前のプロジェクトでは、自分で API を書く方法を学びました。AI API とは、**すでに学習済みの AI モデルを API としてパッケージ化し、呼び出せるようにしたもの**です。

```
従来の方法：自分でモデルを学習させる（大量のデータ、GPU、時間が必要）
API の方法：他人のモデルをそのまま呼び出す（必要なのは API Key と数行のコードだけ）
```

麦を自分で育てないとパンを食べられないわけではないのと同じです。AI API を使えば、**世界最先端の AI 能力をすぐに使えます**。

### よく使われる AI API サービス

| サービス | 提供元 | 主な機能 |
|------|--------|---------|
| OpenAI API | OpenAI | 対話、テキスト生成、コード生成 |
| Claude API | Anthropic | 対話、文書分析、推論 |
| 通義千問 API | アリババクラウド | 対話、テキスト理解 |
| 文心一言 API | Baidu | 対話、知識Q&A |
| 智譜 API | 智譜AI | 対話、コード生成 |

---

## ステップ1：API Key を取得する

AI API を使うには、まず登録して **API Key** を取得する必要があります。これは、あなたにその API を呼び出す権限があることを証明する「鍵」のようなものです。

:::info API サービスを1つ選ぶ
以下のチュートリアルでは OpenAI API を例にします。中国国内からのアクセスが難しい場合は、次を選べます：
- **智譜 AI**（bigmodel.cn）—— 国内で無料枠が比較的大きい
- **通義千問**（dashscope.aliyun.com）—— Alibaba Cloud 系

これらの国産 API は OpenAI ととてもよく似た呼び出し方で、アドレスと Key を少し変えるだけで使えます。
:::

### OpenAI API Key の取得

1. [platform.openai.com](https://platform.openai.com) にアクセスする
2. アカウントを登録/ログインする
3. API Keys ページに入る
4. "Create new secret key" をクリックする
5. Key をコピーして**安全に保存**する（表示されるのは1回だけ）

### OpenAI SDK をインストールする

```bash
python -m pip install --upgrade openai
```

### API Key を設定する

```bash
# 方法 1：環境変数を設定する（おすすめ）
export OPENAI_API_KEY="your_api_key_here"

# 方法 2：コード内で設定する（Git に提交しないこと）
```

:::caution API Key の安全管理
**絶対に** API Key をコードに直接書いて GitHub に送らないでください！ これは、パスワードを公開するのと同じです。正しい方法は：
1. 環境変数を使う
2. `.env` ファイル + `.gitignore` を使う
:::

---

## ステップ2：AI API を初めて呼び出す

現代の OpenAI テキスト生成チュートリアルでは、まず **Responses API** から始めるのがおすすめです。これは、テキスト生成、ツール呼び出し、マルチモーダル入力、後の Agent 型ワークフローまで扱える新しい統一入口です。古いチュートリアルでは `client.chat.completions.create(...)` をよく見かけますが、この教材では `client.responses.create(...)` を主線にします。

```python
from openai import OpenAI

# クライアントを作成する（OPENAI_API_KEY を環境変数から自動で読み込む）
client = OpenAI()

# シンプルなテキスト生成リクエストを送る
response = client.responses.create(
    model="gpt-4o-mini",
    input="こんにちは！ Python 言語を一文で紹介してください。"
)

# AI の返答を取得する
print(f"AI: {response.output_text}")
```

出力例：

```
AI: Python は、シンプルで優雅、かつ高機能な高級プログラミング言語で、データ分析、人工知能、Web 開発など多くの分野で広く使われています。
```

### リクエスト構造を理解する

```python
response = client.responses.create(
    model="gpt-4o-mini",
    instructions="あなたは Python プログラミングのアシスタントです。初心者にも分かる言葉で説明してください。",
    input=[
        {"role": "user", "content": "リスト内包表記とは何ですか？"},
        {"role": "assistant", "content": "リスト内包表記は、リストを短く作る書き方です。"},
        {"role": "user", "content": "例を出してもらえますか？"},
    ],
)

print(response.output_text)
```

| 部分 | 意味 |
|------|------|
| `model` | どのモデルを呼び出すか。学習段階では小さく低コストのモデルから始め、必要に応じて上げます。 |
| `instructions` | モデルへの上位ルール。役割、口調、出力形式などを指定します。 |
| `input` | ユーザー入力、または過去の user/assistant メッセージ一覧です。 |
| `role: "user"` | ユーザーが書いたメッセージです。 |
| `role: "assistant"` | 以前のモデル返答です。手動で会話履歴を渡すときに使います。 |
| `output_text` | SDK が用意している便利な属性で、モデルのテキスト出力を 1 つの文字列にまとめます。 |

:::info 重要用語
- **SDK（Software Development Kit）**：API を呼び出しやすくするためのライブラリです。`openai` Python パッケージが SDK です。
- **Responses API**：OpenAI の統一モデル応答 API です。テキスト、ツール呼び出し、推論情報などを返せます。
- **Chat Completions API**：以前からあるチャット専用 API です。古い教材や互換プロバイダで今も見かけます。
- **context window（コンテキストウィンドウ）**：1 回のリクエストでモデルが扱える入力と出力 Token の合計上限です。
:::

---

## ステップ3：対話型チャットボットを作る

```python
"""
AI チャットボット
OpenAI Responses API を使って多輪対話を実現する
"""

from openai import OpenAI

def create_chatbot(system_prompt: str = "あなたは親切な AI アシスタントです。"):
    """チャットボットを作成する"""
    client = OpenAI()
    history = []

    print("=" * 50)
    print("  AI チャットボット")
    print("  'quit' を入力すると終了、'clear' を入力すると対話履歴を消去します")
    print("=" * 50)

    while True:
        user_input = input("\nあなた: ").strip()

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("さようなら！")
            break
        if user_input.lower() == "clear":
            history = []
            print("🧹 対話履歴を消去しました")
            continue

        # ユーザーのメッセージを追加する
        history.append({"role": "user", "content": user_input})

        try:
            # API を呼び出す
            response = client.responses.create(
                model="gpt-4o-mini",
                instructions=system_prompt,
                input=history,
                temperature=0.7,        # 創造性を調整する（0=保守的、高いほど変化が増える）
                max_output_tokens=800,  # 最大返信長
                store=False,            # 学習用デモではプロバイダ側にこの会話を保存しない
            )

            # 返答を取得する
            reply = response.output_text
            print(f"\nAI: {reply}")

            # AI の返答も履歴に追加する（多輪対話のため）
            history.append({"role": "assistant", "content": reply})

            # Token 使用量を表示する
            usage = response.usage
            print(f"\n  [Token 使用量: 入力={usage.input_tokens}, "
                  f"出力={usage.output_tokens}, "
                  f"合計={usage.total_tokens}]")

        except Exception as e:
            print(f"\n❌ 呼び出し失敗: {e}")
            history.pop()  # 失敗したユーザーのメッセージを削除する

if __name__ == "__main__":
    create_chatbot("あなたは専門的な Python プログラミングの指導者です。簡潔でわかりやすい言葉で答えてください。")
```

---

## ステップ4：実用的な AI ツール

### ツール1：AI コードレビューアシスタント

```python
def review_code(code: str) -> str:
    """AI にコードをレビューしてもらう"""
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            "あなたは経験豊富な Python コードレビューの専門家です。"
            "ユーザーのコードをレビューし、問題点を指摘して改善提案をしてください。"
            "日本語で返答し、わかりやすい形式で出力してください。"
        ),
        input=f"次のコードをレビューしてください:\n\n```python\n{code}\n```",
        temperature=0.3  # コードレビューは低めの温度にして、より厳密にする
    )

    return response.output_text

# テスト
code = """
def calc(l):
    s = 0
    for i in range(len(l)):
        s = s + l[i]
    return s / len(l)
"""

print(review_code(code))
```

### ツール2：AI テキスト要約ツール

```python
def summarize(text: str, max_sentences: int = 3) -> str:
    """AI にテキスト要約を生成してもらう"""
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=f"次のテキストの核心内容を、{max_sentences} 文以内で要約してください。日本語で返答してください。",
        input=text,
        temperature=0.3
    )

    return response.output_text

# 使用例
long_text = """
Python は広く使われている高級プログラミング言語で、Guido van Rossum によって作られ、1991 年に初めて公開されました。
Python の設計思想は、コードの可読性と簡潔さを重視しており、特徴的な空白インデントでコードブロックを定義します。
手続き型、オブジェクト指向、関数型など、さまざまなプログラミングパラダイムをサポートしています。
Python には大規模な標準ライブラリがあり、豊富なサードパーティライブラリのエコシステムもあります。
人工知能、機械学習、データサイエンス、Web 開発などの分野で、Python は最も人気のあるプログラミング言語の1つです。
"""

print(summarize(long_text))
```

### ツール3：AI 翻訳ツール

```python
def translate(text: str, target_lang: str = "英語") -> str:
    """AI 翻訳"""
    client = OpenAI()

    response = client.responses.create(
        model="gpt-4o-mini",
        instructions=(
            f"あなたはプロの翻訳者です。ユーザー入力のテキストを{target_lang}に翻訳してください。"
            "翻訳結果だけを返し、説明は一切追加しないでください。"
        ),
        input=text,
        temperature=0.3
    )

    return response.output_text

print(translate("人工知能は世界を変えています"))
print(translate("Hello, how are you?", "中国語"))
```

---

## ステップ5：国産 AI API を使う（代替案）

国産 AI API を使う場合、コードの構造はほとんど同じで、API のアドレスと Key を変更するだけです。

### 智譜 AI（GLM モデル）

```bash
python -m pip install --upgrade zhipuai
```

この例は Zhipu 独自の SDK を使うため、メソッド名は Chat Completions に似ています。これはプロバイダ固有のインターフェースとして扱ってください。この教材での OpenAI の現代的なテキスト生成主線は、上で示した Responses API です。

```python
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="your_api_key")

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[
        {"role": "user", "content": "こんにちは！ Python について紹介してください"}
    ]
)

print(response.choices[0].message.content)
```

### 汎用の OpenAI 互換インターフェース

多くのプロバイダは OpenAI 風のインターフェースに対応しています。すでに `client.responses.create(...)` をサポートしているものもあれば、まだ `client.chat.completions.create(...)` だけのものもあります。実際に使う前に、必ずプロバイダのドキュメントを確認してください。

```python
from openai import OpenAI

# 別の API サービスを使う
client = OpenAI(
    api_key="your_api_key",
    base_url="https://api.your-provider.com/v1"  # 該当サービスのアドレスに置き換える
)

# このプロバイダが Responses API をサポートしている場合：
response = client.responses.create(
    model="model_name",
    input="こんにちは"
)
print(response.output_text)
```

---

## AI API の重要なパラメータを理解する

| パラメータ | 意味 | 推奨値 |
|------|------|--------|
| `model` | どのモデルを使うか | 学習用デモでは小さく低コストのモデルから始め、難しいタスクで上げる |
| `instructions` | 上位の振る舞いルール | 役割、口調、出力形式、安全ルール |
| `input` | ユーザー入力または会話履歴 | 文字列またはメッセージリスト |
| `temperature` | 創造性/ランダム性 | 0.0-0.3（事実重視）, 0.7-1.0（創造的） |
| `max_output_tokens` | 最大出力長 | 必要に応じて設定する |
| `store` | プロバイダ側にレスポンスオブジェクトを保存するか | 学習用デモでは `False` |
| `stream` | ストリーミング出力するか | `True` でタイピング風の表示を実現 |

### Token と料金

AI API は **Token** 単位で課金されます。Token は、おおよそ1語または数文字の漢字に相当します。

```python
# Token 使用量を確認する
usage = response.usage
print(f"入力 Token: {usage.input_tokens}")
print(f"出力 Token: {usage.output_tokens}")
print(f"合計 Token: {usage.total_tokens}")
```

:::tip コストを抑える
- 学習用デモでは小さめのモデルを使い、本当に難しいタスクだけ上位モデルに切り替える
- `max_output_tokens` を制御して、不要に長い返答を避ける
- system prompt を最適化して、入力 Token を減らす
- 対話履歴を定期的に整理して、Token の蓄積を防ぐ
:::

---

## 発展課題

### 課題1：ストリーミング出力

タイピング風の表示（AI の返答が1文字ずつ現れる）を実装する：

```python
# ヒント：Responses API で stream=True パラメータを使う
stream = client.responses.create(
    model="gpt-4o-mini",
    input="Python 初心者向けの短い歓迎メッセージを書いてください。",
    stream=True,
)

for event in stream:
    if event.type == "response.output_text.delta":
        print(event.delta, end="", flush=True)
```

### 課題2：FastAPI と組み合わせる

AI チャット機能を API として包装し、他の人が HTTP リクエストであなたの AI ボットを使えるようにする。

### 課題3：役割演技

さまざまな役割の AI アシスタント（Python メンター、英語教師、面接官）を作り、ユーザーに選んでもらう。

### 課題4：ローカル知識ベース

AI にまずローカルファイル（たとえばあなたのノート）を読み込ませ、その内容に基づいて質問に答えさせる。

---

## プロジェクト自己チェックリスト

- [ ] API Key の取得と安全な保存に成功した
- [ ] AI API を正常に呼び出して返答を得られた
- [ ] 多輪対話機能を実装した
- [ ] 少なくとも1つの実用ツール（コードレビュー/要約/翻訳）を作った
- [ ] 例外処理がある（ネットワークエラー、API エラー）
- [ ] API Key をコードにハードコードしていない

---

## フェーズのまとめ

おめでとうございます。これで 2 Python プログラミング基礎の学習をすべて終えました。身につけたスキルを振り返りましょう。

| 章 | 身につけたスキル |
|------|-----------|
| Python 基礎 | 変数、データ型、演算子、制御フロー、データ構造、関数、モジュール |
| Python 発展 | オブジェクト指向、例外処理、ファイル操作、関数型プログラミング、ジェネレーター、型注釈 |
| 実践プロジェクト | コマンドラインツール、Web クローラー、Web API、AI API の呼び出し |

あなたはすでに次の力を身につけています：
- **プログラミング思考**：問題をコードロジックに分解できる
- **エンジニアリング力**：構造がわかりやすく、エラー処理のあるプログラムを書ける
- **実践経験**：4つの本物のプロジェクトを完成させた

:::tip 次のステップ
次は 3 データ分析と可視化 に進みます。NumPy、Pandas、Matplotlib を使ってデータを処理し、可視化していきます。これは AI エンジニアの核心スキルです。なぜなら、AI の第一歩はデータを理解することだからです。2 Python プログラミング基礎で身につけた力を持って、そのまま進んでいきましょう！
:::

## バージョン進化のおすすめ

| バージョン | 目標 | 納品の重点 |
|---|---|---|
| 基本版 | 最小の動作ループを通す | 入力できる、処理できる、出力できる、そして一組のサンプルを残す |
| 標準版 | 提示可能なプロジェクトにする | 設定、ログ、エラー処理、README、スクリーンショットを追加する |
| 挑戦版 | ポートフォリオ品質に近づける | 評価、比較実験、失敗サンプル分析、次のステップへの道筋を追加する |

まずは基本版を完成させることをおすすめします。最初から大きく作ろうとしないでください。バージョンを1つ上げるたびに、「何が新しくできるようになったか、どう検証したか、まだ何が問題か」を README に書いていきましょう。
