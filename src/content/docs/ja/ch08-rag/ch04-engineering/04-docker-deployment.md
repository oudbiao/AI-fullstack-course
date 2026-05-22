---
title: "8.4.5 コンテナ化とデプロイ"
description: "なぜコンテナ化が必要なのか、Dockerfile の基本構成、Compose の起動方法までを通して、LLM アプリがローカルスクリプトからデプロイ可能なサービスへ変わる流れを理解します。"
sidebar:
  order: 20
head:
  - tag: meta
    attrs:
      name: keywords
      content: "Docker, containerization, deployment, Dockerfile, Compose, service deployment"
---

# 8.4.5 コンテナ化とデプロイ

:::tip[この節の位置づけ]
多くのプロジェクトは、ここでつまずきます。

- ローカルでは動く
- 別のマシンだと動かない
- チームメンバーごとに環境が違う
- 本番に上げたら依存関係のバージョンがぐちゃぐちゃになる

コンテナ化の本質的な価値は、アプリを次の段階へ進めることです。

> 「自分のPCでは動く」

から

> 「決めた環境で、安定して再現可能に動く」
:::
## 学習目標

- なぜ LLM アプリが特にコンテナ化に向いているのかを理解する
- 最小限の Dockerfile の重要な構成を読めるようになる
- イメージ、コンテナ、ポート、環境変数といった基本概念を理解する
- 小さな Docker Compose の起動方法を読めるようになる
- コンテナ化はデプロイの終点ではなく、出発点だと理解する

## 初学者向けの用語ブリッジ

Docker は、まず名詞を分けると怖さが減ります。

| 用語 | 初学者向けの意味 | なぜ重要か |
|---|---|---|
| `image` | パッケージ化された実行テンプレート。レシピ + 食材セットのようなもの | 先に image を作り、そこから container を起動する |
| `container` | image から作られた実行中のインスタンス | 実際にリクエストを処理するプロセス |
| `Dockerfile` | image を作るための手順書 | ベース image、依存関係、ファイル、起動コマンドを記録する |
| `port` | サービスがリクエストを受け取る入口 | `-p 8000:8000` はホスト側ポートとコンテナ側ポートをつなぐ |
| `environment variable` | コード外から注入する設定 | API key、モデル名、実行モードをコードに直接書かないため |
| `Compose` | 関連する複数コンテナをまとめて起動する道具 | ベクトル DB、Redis、Postgres などが必要なときに便利 |

中心となる考え方は Docker コマンドの丸暗記ではなく、実行環境を再現可能にすることです。

---

## なぜコンテナ化が必要なのか？

### ローカルスクリプトの最大の落とし穴は何か？

ローカルでプロジェクトが動くときは、たいてい多くの暗黙の条件に支えられています。

- Python のバージョン
- パッケージのバージョン
- システム依存関係
- 環境変数
- 起動コマンド

これらの条件は、担当者が変わる、マシンが変わる、サーバーが変わるだけで、すぐに問題を起こします。

### コンテナ化は何を解決するのか？

コンテナ化の本質は次のとおりです。

> **アプリと、それが依存する実行環境をまとめてパッケージ化すること。**

こうすることで、次の内容をより安定して再現できます。

- 何をインストールしたか
- どのバージョンを使ったか
- どのコマンドで起動するか

これは LLM アプリにとても重要です。なぜなら、LLM アプリは次のようなものに依存することが多いからです。

- Web フレームワーク
- モデルサービス
- ベクトルデータベース
- システムツール

---

## イメージとコンテナとは何か？

### とても実用的なたとえ

- **イメージ（image）**：レシピ + 食材セット
- **コンテナ（container）**：そのレシピで実際に作った一皿

つまり、

- イメージは静的なテンプレート
- コンテナは実行中のインスタンス

### この違いがなぜ重要なのか？

デプロイでは通常、次の順番で進めます。

1. まずイメージを build する
2. それからコンテナを起動する

この順番を理解していないと、後で Docker コマンドを見たときにずっと混乱します。

![Docker イメージ、コンテナと Compose デプロイ図](/img/course/ch08-docker-image-container-compose-map-ja.webp)

:::tip[図の見方]
イメージは再現可能な実行テンプレート、コンテナは実行中のインスタンス、Compose は複数のサービスをまとめて起動する役割です。LLM アプリでは、環境変数、ヘルスチェック、ベクトルデータベース、ログもデプロイ図に含めて考える必要があります。
:::
---

## 最小限の Dockerfile はどんな形なのか？

### まずは完全な例を見る

```dockerfile
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "app.py"]
```

### 各行は何をしているのか？

- `FROM`
  - ベースイメージを選ぶ

- `WORKDIR`
  - 作業ディレクトリを指定する

- `COPY requirements.txt .`
  - 依存ファイルをコピーする

- `RUN pip install ...`
  - 依存関係をインストールする

- `COPY . .`
  - そのあとプロジェクトコードをコピーする

- `EXPOSE 8000`
  - サービスが外部に対して待ち受けるポートを示す

- `CMD`
  - コンテナ起動時にデフォルトで実行するコマンド

これが Dockerfile の最も基本的な骨組みです。

:::tip[バージョンについて]
この節では、本コース更新時点の安定版 Python 系列である `python:3.14-slim` を使います。依存ライブラリがまだ完全に対応していない場合は、検証済みの `python:3.13-slim` や `python:3.11-slim` に固定し、その理由をデプロイメモに残してください。
:::
---

## まずは本当に動く小さなアプリを用意する

### 最小の Python サービス

後の Docker デプロイ例を具体的にするために、まずはとてもシンプルな `app.py` を書きます。

```python
# app.py
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "ok"}).encode())
            return

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"message": "hello from llm app"}).encode())

server = HTTPServer(("0.0.0.0", 8000), Handler)
print("8000 で起動中")
server.serve_forever()
```

まずローカルで実行します。

```bash
python app.py
```

別のターミナルでサービスを確認します。

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

想定出力：

```text
{"message": "hello from llm app"}
{"status": "ok"}
```

### なぜ先にこれを書くのか？

それは、コンテナ化は Dockerfile だけを眺める話ではなく、  
実際に動くアプリを中心に理解する必要があるからです。

---

## それをコンテナ化する

### 対応する requirements.txt

この最小サービスはサードパーティのパッケージを使わないので、空ファイルでも構いません。  
ただし、実際のプロジェクトに近づけるために、ここでは構成だけ残しておきます。

```text
# requirements.txt
```

### 対応する Dockerfile

```dockerfile
FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
```

### 実行コマンド

```bash
docker build -t mini-llm-app .
docker run -p 8000:8000 mini-llm-app
```

そのあと、次の URL にアクセスします。

- `http://localhost:8000/`
- `http://localhost:8000/health`

すると、返り値を確認できます。

コマンドラインでも確認できます。

```bash
curl http://localhost:8000/
curl http://localhost:8000/health
```

想定出力：

```text
{"message": "hello from llm app"}
{"status": "ok"}
```

これで、最小限のコンテナ化の流れが完成です。

---

## なぜ環境変数が重要なのか？

LLM アプリでは、次のような設定がよく登場します。

- API Key
- モデル名
- ベクトルデータベースのアドレス
- 実行モード

これらはコードに直接書き込まず、環境変数で渡すほうが適しています。

### 最小の例

```python
import os

model_name = os.getenv("MODEL_NAME", "demo-model")
port = int(os.getenv("PORT", "8000"))

print("MODEL_NAME =", model_name)
print("PORT =", port)
```

追加の環境変数を渡さない場合の想定出力：

```text
MODEL_NAME = demo-model
PORT = 8000
```

### Docker ではどう渡すのか？

```bash
docker run -p 8000:8000 -e MODEL_NAME=qwen-demo mini-llm-app
```

このステップはとても重要です。実際のデプロイでは、ほぼ必ず設定の注入が必要になるからです。

サービスから設定を返したい場合は、`app.py` で `MODEL_NAME` を読み、ルートエンドポイントから返すこともできます。重要なのは、コードは安定させ、設定はイメージの外から変えるという考え方です。

---

## なぜ Compose がよく使われるのか？

### 実際のプロジェクトは 1 つのサービスだけではないから

LLM アプリは、次のようなサービスと組み合わせることがよくあります。

- Web サービス
- ベクトルデータベース
- Redis
- Postgres

それぞれを手動で `docker run` すると、すぐに管理が複雑になります。

### 最小の Compose 例

```yaml
version: "3.9"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    environment:
      MODEL_NAME: demo-model
```

起動方法：

```bash
docker compose up --build
```

これが、Compose がローカル開発や小規模デプロイでとても便利な理由です。

---

## コンテナ化はデプロイ完了を意味しない

これはとてもよくある誤解です。

### コンテナ化が解決するのは「パッケージ化と実行環境」

でも、本番運用ではさらに次のことを考える必要があります。

- ログ
- ヘルスチェック
- リソース制限
- 自動再起動
- 段階的リリース
- リバースプロキシ

### ヘルスチェックの重要な考え方

前の例のような

- `/health`

というエンドポイントは、とても価値があります。  
なぜなら、デプロイシステムは次のことを知る必要があるからです。

> このコンテナは今生きているか、リクエストを受けられるか。 

---

## 初心者がよくハマる落とし穴

### すべてを巨大なイメージに詰め込む

イメージがどんどん重くなります。

### ヘルスチェックがない

サービスが壊れても気づけません。

### 設定をコードに固定する

環境が変わるとすぐに問題になります。

### コンテナ化したら自動でスケールすると思う

そうではありません。  
コンテナ化は最初の一歩で、その後にオーケストレーション、監視、運用が続きます。

### ローカル Docker のディスク使用量を見ない

build 中に `no space left on device` が出たら、まず Docker の使用量を確認します。

```bash
docker system df
docker builder prune
```

不要だと分かっているものだけを削除してください。チームのマシンや CI 環境では、いきなり image や volume を消すより、まず build cache を掃除する方が安全です。

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

この節で最も大事なのは Docker コマンドを暗記することではなく、次を理解することです。

> **コンテナ化の本質的な価値は、「アプリ + 依存関係 + 起動方法」をまとめて標準化し、デプロイを個人のPCの経験から再現可能な流れへ変えること。**

ここをしっかり固めることで、その先のサービス構成や本番運用の土台ができます。

---

## 練習

1. この節の `app.py` と Dockerfile を使って、ローカルで本当に最小イメージを build してみましょう。
2. サービスに `APP_MODE=dev` のような環境変数を追加してみましょう。
3. 考えてみましょう：なぜ `/health` エンドポイントがデプロイシステムにとって重要なのでしょうか？
4. 自分の言葉で説明してみましょう：なぜコンテナ化はデプロイの終点ではなく、出発点なのでしょうか？

<details>
<summary>参考実装と解説</summary>

1. build 結果は、安定して起動し、期待する port と health endpoint を公開する image であるべきです。
2. `APP_MODE` は環境変数から読み、コード変更なしで config や log に反映します。
3. `/health` は、デプロイシステムが traffic を流すか、container を再起動するか、rollback するかを判断する材料になります。
4. container は runtime をまとめますが、deployment には secrets、scaling、logs、monitoring、storage、networking、security、release process がまだ必要です。

</details>
