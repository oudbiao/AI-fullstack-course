---
title: "1.3 プロジェクト：Web API 開発"
sidebar_position: 3
description: "FastAPI を使って、最初の Web API を構築しよう"
---

# プロジェクト：Web API 開発

![Web API リクエスト・レスポンスのアーキテクチャ図](/img/course/ch02-web-api-request-response-ja.png)

## プロジェクトの位置づけ

このプロジェクトでは、Python をスクリプトからサーバーサイドへ広げていきます。FastAPI を使って機能を他のプログラムから呼び出せる API として包み込み、API がモデル・アプリ・ユーザーをどうつなぐのかを理解し、これからの AI アプリ開発の土台を作ります。

## プロジェクトの目標

- API とは何か、なぜ AI エンジニアが API を書ける必要があるのかを理解する
- FastAPI フレームワークを使って Web API を構築できるようになる
- RESTful API の基本的な設計原則を身につける
- 他のプログラムから呼び出せる AI サービスの入り口を作る

---

## なぜ AI エンジニアは API を書ける必要があるの？

とても良い AI モデルを学習できたとして、その次はどうするのでしょうか？

モデルを学習しただけでは、まだ第一歩にすぎません。ほかの人が**そのモデルを使える**ようにするには、**API サービスとして包む**必要があります。

```
あなたの AI モデル  →  API として包む  →  スマホアプリ / Webサイト / ほかのプログラムが呼び出す

具体例：
- ChatGPT モデル → API でサービス提供 → さまざまなアプリが呼び出す
- 画像認識モデル → API を通して → ユーザーが画像をアップロードして認識結果を受け取る
- レコメンドアルゴリズム → API を通して → ECサイトがおすすめ商品を表示する
```

つまり、**API は AI モデルと現実世界をつなぐ橋**です。

---

## API とは？

**API（Application Programming Interface）** = アプリケーション・プログラミング・インターフェース。

簡単にいうと、API は**プログラム同士の「会話の窓口」**です。

レストランで食事をするときの流れは、API を呼び出すのとよく似ています。

```
あなた（クライアント）  →  ウェイター（API）に「牛肉麺を1杯」と伝える（リクエスト）
ウェイター              →  厨房（サーバー）へ伝える
厨房                    →  麺を作る
ウェイター              →  あなたに麺を運ぶ（レスポンス）
```

あなたは厨房でどのように麺を作っているかを知る必要はありません。知っていればよいのは、**どう注文するか（リクエストを送る）** と **どう受け取るか（レスポンスを受け取る）** です。

### Web API の基本概念

| 概念 | 説明 | たとえ |
|------|------|------|
| **URL（エンドポイント）** | API の住所 | レストランの住所 |
| **HTTP メソッド** | 操作の種類 | 注文する / 返品する / 追加注文する |
| **リクエストボディ** | 送るデータ | 欲しい料理名 |
| **レスポンス** | 返ってくる結果 | 目の前に出てきた料理 |
| **ステータスコード** | 操作が成功したかどうか | 200=成功, 404=その料理はない |

### HTTP メソッド

| メソッド | 用途 | 例 |
|------|------|------|
| `GET` | データを取得する | タスク一覧を取得する |
| `POST` | データを作成する | 新しいタスクを追加する |
| `PUT` | データを更新する（全体） | タスクの情報をまとめて変更する |
| `DELETE` | データを削除する | タスクを1件削除する |

---

## 第1歩：FastAPI をインストールする

```bash
pip install fastapi uvicorn
```

| ライブラリ | 役割 |
|---|------|
| `fastapi` | API を書くための Web フレームワーク |
| `uvicorn` | FastAPI アプリを実行する ASGI サーバー |

---

## 第2歩：Hello World API

`main.py` を作成します：

```python
from fastapi import FastAPI

# アプリのインスタンスを作成
app = FastAPI(title="私の最初の API", version="1.0")

# エンドポイントを定義
@app.get("/")
def root():
    return {"message": "Hello, World!", "status": "running"}

@app.get("/hello/{name}")
def hello(name: str):
    return {"message": f"こんにちは、{name}！", "name": name}
```

サーバーを起動します：

```bash
uvicorn main:app --reload
```

- `main` = ファイル名（`main.py`）
- `app` = FastAPI インスタンス名
- `--reload` = コードを変更したら自動で再起動する（開発時に使う）

ブラウザで次の URL を開いてみましょう：
- `http://127.0.0.1:8000` → Hello World が表示される
- `http://127.0.0.1:8000/hello/小明` → 個別のあいさつが表示される
- `http://127.0.0.1:8000/docs` → **自動生成されたインタラクティブな API ドキュメント！**

:::tip FastAPI の強み：自動ドキュメント
`/docs` にアクセスすると、Swagger UI ベースの美しいインタラクティブ API ドキュメントが表示されます。ブラウザ上でそのまま API を試せるので、フロントエンドを別に作る必要がありません。これが FastAPI がとても人気な理由の1つです。
:::

---

## 第3歩：タスク管理 API を作る

前に作ったコマンドラインのタスク管理ツールを、Web API に変えてみましょう。

```python
"""
タスク管理 API
実行: uvicorn main:app --reload
ドキュメント: http://127.0.0.1:8000/docs
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# アプリを作成
app = FastAPI(
    title="タスク管理 API",
    description="シンプルな RESTful タスク管理インターフェース",
    version="1.0"
)

# ---------- データモデル ----------

class TaskCreate(BaseModel):
    """タスク作成時のリクエストボディ"""
    title: str
    priority: str = "中"

class TaskUpdate(BaseModel):
    """タスク更新時のリクエストボディ"""
    title: str | None = None
    priority: str | None = None
    done: bool | None = None

class Task(BaseModel):
    """タスクの完全なデータ"""
    id: int
    title: str
    priority: str
    done: bool
    created_at: str

# ---------- 模擬データベース（メモリ保存） ----------

tasks_db: list[dict] = []
next_id: int = 1

# ---------- API エンドポイント ----------

@app.get("/")
def root():
    """API のトップページ"""
    return {
        "name": "タスク管理 API",
        "version": "1.0",
        "endpoints": {
            "すべてのタスクを見る": "GET /tasks",
            "タスクを作成する": "POST /tasks",
            "1件のタスクを見る": "GET /tasks/{task_id}",
            "タスクを更新する": "PUT /tasks/{task_id}",
            "タスクを削除する": "DELETE /tasks/{task_id}",
            "API ドキュメント": "GET /docs"
        }
    }


@app.get("/tasks")
def get_tasks(done: bool | None = None):
    """
    すべてのタスクを取得します。

    オプション引数:
    - done: 完了済み(true)または未完了(false)のタスクを絞り込む
    """
    if done is not None:
        filtered = [t for t in tasks_db if t["done"] == done]
        return {"count": len(filtered), "tasks": filtered}
    return {"count": len(tasks_db), "tasks": tasks_db}


@app.post("/tasks", status_code=201)
def create_task(task: TaskCreate):
    """新しいタスクを作成する"""
    global next_id

    new_task = {
        "id": next_id,
        "title": task.title,
        "priority": task.priority,
        "done": False,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    tasks_db.append(new_task)
    next_id += 1

    return {"message": "タスクの作成に成功しました", "task": new_task}


@app.get("/tasks/{task_id}")
def get_task(task_id: int):
    """指定した ID のタスクを取得する"""
    for task in tasks_db:
        if task["id"] == task_id:
            return task

    # タスクが見つからない場合は 404 を返す
    raise HTTPException(status_code=404, detail=f"タスク {task_id} は存在しません")


@app.put("/tasks/{task_id}")
def update_task(task_id: int, task_update: TaskUpdate):
    """タスクを更新する"""
    for task in tasks_db:
        if task["id"] == task_id:
            if task_update.title is not None:
                task["title"] = task_update.title
            if task_update.priority is not None:
                task["priority"] = task_update.priority
            if task_update.done is not None:
                task["done"] = task_update.done
            return {"message": "更新に成功しました", "task": task}

    raise HTTPException(status_code=404, detail=f"タスク {task_id} は存在しません")


@app.delete("/tasks/{task_id}")
def delete_task(task_id: int):
    """タスクを削除する"""
    for i, task in enumerate(tasks_db):
        if task["id"] == task_id:
            removed = tasks_db.pop(i)
            return {"message": "削除に成功しました", "task": removed}

    raise HTTPException(status_code=404, detail=f"タスク {task_id} は存在しません")


@app.get("/stats")
def get_stats():
    """タスクの統計情報を取得する"""
    total = len(tasks_db)
    done = sum(1 for t in tasks_db if t["done"])
    return {
        "total": total,
        "done": done,
        "pending": total - done,
        "completion_rate": f"{done/total:.1%}" if total > 0 else "0%"
    }
```

### 実行とテスト

```bash
# サーバーを起動
uvicorn main:app --reload
```

その後、`http://127.0.0.1:8000/docs` を開くと、ブラウザからすべての API を直接テストできます。

コマンドラインでもテストできます：

```bash
# タスクを作成
curl -X POST http://127.0.0.1:8000/tasks \
  -H "Content-Type: application/json" \
  -d '{"title": "FastAPI を学ぶ", "priority": "高"}'

# すべてのタスクを確認
curl http://127.0.0.1:8000/tasks

# タスクを完了にする
curl -X PUT http://127.0.0.1:8000/tasks/1 \
  -H "Content-Type: application/json" \
  -d '{"done": true}'

# タスクを削除
curl -X DELETE http://127.0.0.1:8000/tasks/1
```

または、Python の `requests` ライブラリを使うこともできます：

```python
import requests

BASE_URL = "http://127.0.0.1:8000"

# タスクを作成
resp = requests.post(f"{BASE_URL}/tasks", json={"title": "Python を学ぶ", "priority": "高"})
print(resp.json())

# すべてのタスクを取得
resp = requests.get(f"{BASE_URL}/tasks")
print(resp.json())
```

---

## Pydantic のデータモデルを理解する

FastAPI は **Pydantic** を使ってリクエストデータを検証します。データモデルを定義しておけば、FastAPI が自動でチェックしてくれます。

```python
from pydantic import BaseModel, Field

class TaskCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=100, description="タスクのタイトル")
    priority: str = Field(default="中", description="優先度: 高/中/低")

# ユーザーが不正なデータを送った場合
# POST /tasks {"title": ""} → 自動で 422 エラーを返す（タイトルが短すぎる）
# POST /tasks {} → 自動で 422 エラーを返す（title がない）
# POST /tasks {"title": "OK"} → 成功、priority はデフォルト値の "中" を使う
```

自分で検証コードを書く必要はありません。Pydantic と FastAPI がやってくれます。

---

## 発展チャレンジ

### チャレンジ1：ファイルへの永続化を追加する

今はデータがメモリ上にあるだけなので、サーバーを再起動すると消えてしまいます。JSON ファイルに保存するように変えてみましょう。

### チャレンジ2：検索機能を追加する

```python
@app.get("/tasks/search")
def search_tasks(keyword: str):
    """キーワードでタスクを検索する"""
    keyword_lower = keyword.lower()
    return [task for task in tasks if keyword_lower in task.title.lower()]
```

### チャレンジ3：ページネーションを追加する

タスクが多くなったら、ページごとに返せるようにしましょう。

```
GET /tasks?page=1&size=10
```

### チャレンジ4：AI モデルをつなぐ

先取り学習として、`/predict` エンドポイントを作り、テキスト入力を受け取って感情分析の結果を返してみましょう。

---

## プロジェクト自己チェックリスト

- [ ] API が正常に起動してアクセスできる
- [ ] CRUD（作成・取得・更新・削除）の操作をすべて実装した
- [ ] Pydantic でデータ検証を行っている
- [ ] 適切なエラー処理（HTTPException）がある
- [ ] 自動生成された API ドキュメント（`/docs`）を正常に使える
- [ ] curl または requests で各エンドポイントをテストできる

:::tip プロジェクトの経験
FastAPI は、AI エンジニアがよく使う Web フレームワークの1つです。多くの AI プロジェクトは、学習したモデルを FastAPI で API として包み、サーバーにデプロイする形で公開されます。FastAPI を身につけると、**AI モデルを製品として形にする力** が手に入ります。さらに、FastAPI の自動ドキュメント機能のおかげで、フロントエンドとバックエンドの協力もとてもスムーズになります。
:::

## バージョン別の進め方のおすすめ

| バージョン | 目標 | 仕上げるポイント |
|---|---|---|
| 基本版 | 最小の流れを動かす | 入力できる、処理できる、出力できる。さらにサンプルを1セット残す |
| 標準版 | 見せられるプロジェクトにする | 設定、ログ、エラー処理、README、スクリーンショットを追加する |
| 発展版 | 作品集レベルに近づける | 評価、比較実験、失敗例の分析、次の改善方針を追加する |

まずは基本版を完成させましょう。最初から全部入りを目指す必要はありません。1つバージョンを上げるたびに、「何が新しくできるようになったか、どうやって確認したか、まだ何が課題か」を README に書き足していきましょう。
