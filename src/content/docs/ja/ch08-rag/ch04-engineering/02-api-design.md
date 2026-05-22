---
title: "8.4.3 API 設計とサービス化"
description: "リクエスト構造、レスポンス構造、冪等性、エラー処理からバージョン管理まで、LLM サービス API をより安定して設計する考え方を理解する。"
sidebar:
  order: 18
head:
  - tag: meta
    attrs:
      name: keywords
      content: "API design, service design, idempotency, request schema, response schema, versioning"
---

# 8.4.3 API 設計とサービス化

:::tip[この節の位置づけ]
LLM アプリを作るとき、多くの人はローカルスクリプトまでは書けても、サービス化の段階で急に混乱します。  
本当に大事なのは「インターフェースを書けるか」ではなく、次の点です。

> **このインターフェースを、他の人が長期的に安定して呼び出せるか。**

この節では、その問いに答えます。
:::
## 学習目標

- LLM サービス API が最低限どんな内容を定義すべきか理解する
- わかりやすいリクエスト・レスポンス構造を設計できるようになる
- 冪等性、エラー返却、トレース_id、バージョン管理といったサービス化の重要概念を理解する
- 最小限の API 処理の流れを読めるようになる

## 初学者向けの用語ブリッジ

API 設計は、次の言葉に直感を持てるとかなり読みやすくなります。

| 用語 | 初学者向けの意味 | この節での役割 |
|---|---|---|
| `API` | Application Programming Interface。あるプログラムが別のプログラムを安定して呼ぶための入口 | 他のコードが依存するサービス入口 |
| `endpoint` | `/api/v1/chat` のような、具体的に呼び出せるアドレス | 機能を URL パスとして公開する場所 |
| `schema` | どのフィールドを許可し、何を必須にするかを決めるルール | リクエストとレスポンスの形を予測しやすくする |
| `payload` | リクエストで送るデータ本体 | この節では、ユーザーの質問や関連メタデータを指すことが多い |
| `trace_id` | 1件のリクエストを追跡するための一意な ID | API ログ、検索ログ、モデルログ、エラーをつなげる |
| `idempotency` | 同じリクエストを繰り返しても、制御できない副作用が増えない性質 | タイムアウトやネットワーク失敗後のリトライで重要 |

これらは単なる用語ではありません。実際のシステムでは、フロントエンド、バックエンド、ログ、評価、デプロイが協調するための部品です。

---

## なぜ API 設計は「ただ JSON で包むだけ」ではないのか？

### よくないインターフェースはどんな形？

```python
bad_request = {
    "msg": "返金ポリシーは何ですか"
}

bad_response = {
    "text": "7 日以内なら返金可能です"
}
```

何が問題でしょうか？

- `msg` は何を意味するのか？ ユーザーメッセージ？ システムメッセージ？
- `trace_id` がない
- エラー構造がない
- バージョン情報がない
- コンテキスト用のフィールドがない

### よい API 設計は何をしているのか？

本質的には、次のことに答えています。

- 入力はどんな形か
- 出力はどんな形か
- 失敗したときどう表すか
- 1回呼んでも10万回呼んでも安定するか

つまり、API 設計は「入口を作る」ことではなく、次を定義することです。

> **システムと外部世界の契約。**

---

## まずはリクエスト構造を設計する

### 最小のリクエスト構造には、少なくともこれが必要

- `query`
- `user_id`（任意）
- `session_id`（複数ターンのとき）
- `metadata`（任意）

### もっとわかりやすいリクエストオブジェクト

```python
request = {
    "query": "返金ポリシーは何ですか？",
    "user_id": 1,
    "session_id": "sess_001",
    "metadata": {
        "channel": "web"
    }
}

print(request)
```

想定出力：

```text
{'query': '返金ポリシーは何ですか？', 'user_id': 1, 'session_id': 'sess_001', 'metadata': {'channel': 'web'}}
```

これで、次のことがはっきりします。

- 何についての問い合わせか
- 誰から送られたのか
- どの会話に属するのか
- 追加のコンテキストは何か

これは「文字列を1つ渡すだけ」よりずっと良い設計です。

---

## 次にレスポンス構造を設計する

### なぜレスポンスも規約化する必要があるのか？

実際の呼び出し元は、人だけではありません。たとえば：

- フロントエンド
- 他のサービス
- ログシステム
- 評価システム

これらは、安定した形式で結果を受け取る必要があります。

### より安定したレスポンス構造

```python
response = {
    "trace_id": "trace_001",
    "answer": "コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請できます。",
    "sources": [
        {"id": "doc_001", "section": "返金ポリシー"}
    ],
    "usage": {
        "prompt_tokens": 120,
        "completion_tokens": 35
    }
}

print(response)
```

想定出力：

```text
{'trace_id': 'trace_001', 'answer': 'コース購入後 7 日以内、かつ学習進捗が 20% 未満であれば返金申請できます。', 'sources': [{'id': 'doc_001', 'section': '返金ポリシー'}], 'usage': {'prompt_tokens': 120, 'completion_tokens': 35}}
```

### これらのフィールドに価値がある理由

- `trace_id`：処理の流れを追いやすくなる
- `answer`：実際の業務出力
- `sources`：参照元の確認や検証に使える
- `usage`：コスト分析に使える

---

## エラー応答も必ず設計する

### 多くのシステムは成功時の返却だけを考えがち

でも、実務で多いのはむしろ次のような問題です。

- パラメータが不正
- 上流タイムアウト
- 権限不足
- ナレッジベースが空

### 統一されたエラー構造

```python
error_response = {
    "trace_id": "trace_002",
    "error": {
        "code": "INVALID_ARGUMENT",
        "message": "query は空にできません"
    }
}

print(error_response)
```

想定出力：

```text
{'trace_id': 'trace_002', 'error': {'code': 'INVALID_ARGUMENT', 'message': 'query は空にできません'}}
```

これはとても重要です。呼び出し側が次のことを明確に判断できるからです。

- 何が起きたのか
- エラーの種類は何か
- リトライする価値があるか

![API 契約、エラー構造、バージョン管理の図](/img/course/ch08-api-contract-error-version-map-ja.webp)

:::tip[図の読み方]
API は単なる JSON ではなく、システムの契約です。図を見るときは request schema、response schema、error object、trace_id、version に注目してください。これらが、フロントエンド、評価システム、他サービスから長期的に安定して使われるかどうかを決めます。
:::
---

## 最小で動くサービス化処理関数

### 純粋な Python で API handler を模擬する

```python
def handle_chat(request):
    trace_id = "trace_demo_001"

    if "query" not in request or not request["query"].strip():
        return {
            "trace_id": trace_id,
            "error": {
                "code": "INVALID_ARGUMENT",
                "message": "query は空にできません"
            }
        }

    answer = f"システム応答：{request['query']}"
    return {
        "trace_id": trace_id,
        "answer": answer,
        "sources": [],
        "usage": {"prompt_tokens": 12, "completion_tokens": 8}
    }

print(handle_chat({"query": "返金ポリシーは何ですか？"}))
print(handle_chat({"query": ""}))
```

想定出力：

```text
{'trace_id': 'trace_demo_001', 'answer': 'システム応答：返金ポリシーは何ですか？', 'sources': [], 'usage': {'prompt_tokens': 12, 'completion_tokens': 8}}
{'trace_id': 'trace_demo_001', 'error': {'code': 'INVALID_ARGUMENT', 'message': 'query は空にできません'}}
```

### このコードは何を教えているのか？

教えているのは、次の3点です。

1. まずリクエストを検証する
2. すべてのリクエストに `trace_id` を付ける
3. 成功時と失敗時の返却形式を統一する

これが、サービス化設計の最も重要な基礎です。

---

## なぜ冪等性が重要なのか？

### 冪等性とは？

簡単に言うと、

> 同じリクエストを何回呼んでも、結果が同じ、または制御可能であること。 

これは次のような場面で特に重要です。

- リトライ
- タイムアウト後の再送信
- ネットワークの揺らぎ

### どんなインターフェースで特に意識すべきか？

特に重要なのは、次のようなものです。

- 問い合わせチケットの作成
- 支払いの開始
- 注文変更

一方、純粋な QA インターフェースは、もともと「読み取り」に近いので、冪等性は比較的扱いやすいです。

---

## なぜバージョン管理は後回しにできないのか？

### API は一度他のシステムに組み込まれると、自由に項目を変えにくい

たとえば、今日の返却が

- `answer`

だったのに、明日いきなり

- `response_text`

に変えると、呼び出し側はすぐ壊れます。

### シンプルなバージョン戦略

```python
api_info = {
    "version": "v1",
    "endpoint": "/api/v1/chat"
}

print(api_info)
```

想定出力：

```text
{'version': 'v1', 'endpoint': '/api/v1/chat'}
```

小さなプロジェクトでも、早めにバージョン意識を持つことをおすすめします。

---

## より実際のサービスに近い FastAPI の例

実際のバックエンドに近い書き方を見たいなら、次の最小例が参考になります。

:::note[実行環境]
```bash
pip install fastapi uvicorn
uvicorn app:app --reload
```
:::
```python
from fastapi import FastAPI
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None


app = FastAPI()

@app.post("/api/v1/chat")
def chat(payload: ChatRequest):
    return {
        "trace_id": "trace_demo_002",
        "answer": f"システム応答：{payload.query}",
        "session_id": payload.session_id,
    }
```

このコードはシンプルですが、直接 `dict` を受け取るより実サービスに近い形です。`ChatRequest` はリクエスト schema であり、FastAPI はビジネスロジックに入る前に payload を検証します。本番では通常、認証、統一エラー、ログ、実際の trace_id 生成も追加します。

---

## 目標が「ナレッジベース駆動の SOP 文書アシスタント」なら、API の最小構成はどうなるか？

この種のシステムは、`/chat` だけでは足りないことが多いです。  
少なくとも次のようなインターフェースがあるとよいです。

| インターフェース | 役割 |
|---|---|
| `/sop-drafts/generate` | ポリシー、ケース、チェックリスト根拠から構造化 SOP ドラフトを生成する |
| `/sop-drafts/preview` | エクスポート前に構造化された SOP セクションを確認する |
| `/documents/ingest` | PDF / Word / PPT をアップロードして解析する |
| `/retrieval/search` | 検索結果をデバッグする |

最初に作るときは、より安定した進め方として、だいたい次の順番がよいです。

1. まず `generate` だけを作る
2. まずは構造化結果かエクスポートリンクを返す
3. その後にデバッグ用インターフェースやバッチ処理を追加する

最小のリクエスト構造は、まずこんな形で定義できます。

```python
generate_request = {
    "topic": "返金エスカレーション SOP",
    "audience": "一次サポート",
    "doc_format": "word",
    "case_count": 2,
    "checklist_required": True,
}

print(generate_request)
```

想定出力：

```text
{'topic': '返金エスカレーション SOP', 'audience': '一次サポート', 'doc_format': 'word', 'case_count': 2, 'checklist_required': True}
```

このオブジェクトの価値は、次の点にあります。

- 複数ターン対話で集めた項目を、実際のサービス API のパラメータとして落とし込める

## 実践：SOP ドラフト API の契約を模擬する

本物の FastAPI endpoint を作る前に、まず純粋な Python でリクエスト検証とレスポンス契約を書いてみます。これにより、サービス境界がはっきりします。

```python
REQUIRED_FIELDS = ["topic", "audience", "doc_format", "case_count", "checklist_required"]


def validate_generate_request(payload):
    missing = [field for field in REQUIRED_FIELDS if field not in payload or payload.get(field) is None]
    if missing:
        return False, {
            "code": "INVALID_ARGUMENT",
            "message": f"不足フィールド：{missing}"
        }
    if payload["doc_format"] not in {"word", "ppt"}:
        return False, {
            "code": "INVALID_ARGUMENT",
            "message": "doc_format は word または ppt である必要があります"
        }
    return True, None


def handle_generate(payload):
    trace_id = "trace_sop_001"
    ok, error = validate_generate_request(payload)
    if not ok:
        return {"trace_id": trace_id, "error": error}

    return {
        "trace_id": trace_id,
        "status": "accepted",
        "sop_draft": {
            "title": payload["topic"],
            "audience": payload["audience"],
            "format": payload["doc_format"],
            "sections": ["ポリシー要約", "処理済みケース", "一次サポートチェックリスト"],
        }
    }


generate_request = {
    "topic": "返金エスカレーション SOP",
    "audience": "一次サポート",
    "doc_format": "word",
    "case_count": 2,
    "checklist_required": True,
}

print(handle_generate(generate_request))
print(handle_generate({"topic": "返金エスカレーション SOP", "doc_format": "pdf"}))
```

想定出力：

```text
{'trace_id': 'trace_sop_001', 'status': 'accepted', 'sop_draft': {'title': '返金エスカレーション SOP', 'audience': '一次サポート', 'format': 'word', 'sections': ['ポリシー要約', '処理済みケース', '一次サポートチェックリスト']}}
{'trace_id': 'trace_sop_001', 'error': {'code': 'INVALID_ARGUMENT', 'message': "不足フィールド：['audience', 'case_count', 'checklist_required']"}}
```

![SOP ドラフト API 契約結果図](/img/course/ch08-courseware-api-contract-result-map-ja.webp)

:::tip[図の見方]
2つの経路を同じ検証ゲートに通して読みます。完全な payload は `status=accepted` の SOP ドラフトになり、不完全な payload はビジネスロジックの前で統一された `INVALID_ARGUMENT` エラーに止まります。
:::
この練習が役立つのは、成功と失敗を同時に設計する必要があるからです。成功パスの返却だけでは、サービスが準備できたとは言えません。

## 初学者がよくつまずくポイント

### リクエスト構造が雑すぎる

最初は楽でも、後でとても苦しくなります。

### エラー構造が統一されていない

フロントエンドや他サービスとの接続がどんどん難しくなります。

### `trace_id` がない

問題が起きたときに、処理の流れを追えません。

### 最初からインターフェースを単一の業務ロジックに固定しすぎる

後からの拡張がかなり難しくなります。

---

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
サービス契約: エンドポイント、入力スキーマ、出力スキーマ、エラースキーマ
実行シグナル: レイテンシ、スループット、ログ、ヘルスチェック、またはコンテナ状態
可観測性：request id、trace id、構造化ログ、または metric
失敗確認: タイムアウト、リトライの連鎖、ログ不足、デプロイ不一致
運用アクション：バックオフ、キュー、アラート、段階展開、またはロールバック
```

## まとめ

この節で最も大事なのは、インターフェースを動かすことそのものではなく、次を理解することです。

> **API 設計の本質は、入力、出力、エラー、トレース情報を安定したシステム契約にすること。**

契約がはっきりしていれば、サービスは他人に長く安定して使われるようになります。

---

## 練習

1. `handle_chat()` に `session_id` フィールドのサポートを追加してみましょう。
2. `INVALID_ARGUMENT`、`TIMEOUT`、`NOT_FOUND` のような統一エラーコード列挙を設計してみましょう。
3. 考えてみましょう：もしこれが「チケット作成」インターフェースなら、冪等性をどう考えますか？
4. 自分の言葉で説明してみましょう。なぜ API 設計は本質的にシステム契約を定義することだと言えるのでしょうか？

<details>
<summary>参考実装と解説</summary>

1. `session_id` は request parsing、state lookup、logs、response trace を通って流れるべきです。空値や不正形式も検証します。
2. error enum があると client は安定してエラー処理でき、ユーザー起因のエラーとサービス起因のエラーも分けられます。
3. idempotency key を使うと、client が timeout 後に retry してもチケットが重複作成されません。
4. API contract は入力、出力、エラー、権限、時間的期待、互換性を定義します。

</details>
