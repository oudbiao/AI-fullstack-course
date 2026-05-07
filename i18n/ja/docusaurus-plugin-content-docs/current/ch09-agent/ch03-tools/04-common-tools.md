---
title: "9.3.5 よく使うツールの統合"
sidebar_position: 14
description: "検索、計算機、データベース、ファイルシステム、ブラウザまで、Agent で最もよく使うツールの種類と、それらをシステムにどう接続するかを理解します。"
keywords: [tool integration, search, calculator, database, filesystem, browser, Agent]
---

# 9.3.5 よく使うツールの統合

:::tip この節の位置づけ
ツール層を説明するとき、抽象的な schema だけにとどまると、どうしても実感がわきにくいです。  
この節では少し視点を近づけて、次の点を直接見ていきます。

> **Agent システムで最もよく使うツールは何か、それぞれどうつなぐのか？**

名前は違っても、接続のしかたには共通点が多いことが分かります。
:::

## 学習目標

- Agent でよく使う代表的なツールの種類を理解する
- 各種類のツールが何の問題に向いているかを理解する
- 統一されたツール登録とディスパッチの例を読めるようになる
- ツール統合でよく起きる失敗点と、実装上の注意点を理解する

---

## なぜツールを種類ごとに見るのか？

### 「ツール」という言葉の範囲が広すぎるから

検索もツール、計算機もツール、データベース検索もツール、ファイルの読み書きもツールです。  
これらを全部まとめて「1つの関数」と考えると、すぐに混乱します。

より実用的なのは、まず次のように分類することです。

1. 検索系
2. 計算系
3. データアクセス系
4. ファイル / 環境操作系
5. 外部サービス呼び出し系

### なぜ分類が役立つのか？

種類ごとに注目点が違うからです。

- 検索系は召喚率ではなく、検索結果の質を見る
- 計算系は正確さと安全性を見る
- データベース系は権限と絞り込みを見る
- ファイル系はパスの境界を見る
- 外部サービス系はタイムアウトとリトライを見る

つまり、

> どれも「ツール」ではあるけれど、実装上のリスクはまったく同じではない

ということです。

---

## よくある5種類のツール

### 検索 / 検索取得系

向いている用途:

- ドキュメントを探す
- ナレッジベースを検索する
- Webページを探す

特徴:

- 入力は通常 query
- 出力は通常、候補の一覧

### 計算系

向いている用途:

- 四則演算
- 統計指標の計算
- 小さなデータ変換

特徴:

- 出力は安定して正確である必要がある
- 安全性に特に注意する必要がある

### データアクセス系

向いている用途:

- データベースを検索する
- 注文情報を確認する
- ユーザー状態を確認する

特徴:

- パラメータと権限が最重要
- 多くの業務ロジックがこの層で決まる

### ファイル / 環境操作系

向いている用途:

- ファイルを読む
- ファイルを書く
- ディレクトリを列挙する
- コードを実行する

特徴:

- リスクが高い
- 境界管理が非常に重要

### 外部サービス呼び出し系

向いている用途:

- メールを送る
- 外部の API を呼ぶ
- チケットを作成する

特徴:

- 失敗、タイムアウト、リトライがよく発生する

---

## 統一されたツール登録表

実際のシステムでは、ツールをあちこちに散らすのではなく、まとめて登録することがよくあります。

### 最小実行例

```python
import ast
import operator

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_calculate(expression):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError("unsupported_expression")

    return visit(ast.parse(expression, mode="eval"))


def search_docs(keyword):
    docs = {
        "返金": "コース購入後 7 日以内に返金申請ができます",
        "証明書": "プロジェクトを完了し、テストに合格すると証明書を取得できます"
    }
    return docs.get(keyword, "関連ドキュメントが見つかりませんでした")

def calculator(expression):
    return safe_calculate(expression)

def get_user_status(user_id):
    mock_db = {
        1: {"name": "Alice", "progress": 0.15},
        2: {"name": "Bob", "progress": 0.35}
    }
    return mock_db.get(user_id, {"error": "user_not_found"})

TOOLS = {
    "search_docs": search_docs,
    "calculator": calculator,
    "get_user_status": get_user_status
}

print(TOOLS.keys())
```

### なぜ統一登録が重要なのか？

後で次のようなことが必要になるからです。

- schema を統一して説明する
- 権限管理をまとめて行う
- ログを統一して取る
- ディスパッチと集計を一元化する

ツールに登録表がないと、システムはどんどん保守しづらくなります。

---

## 統一ディスパッチャ

### 最小ディスパッチャ例

```python
def dispatch(call):
    name = call["name"]
    arguments = call["arguments"]

    if name not in TOOLS:
        return {"error": "unknown_tool"}

    try:
        result = TOOLS[name](**arguments)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

calls = [
    {"name": "search_docs", "arguments": {"keyword": "返金"}},
    {"name": "calculator", "arguments": {"expression": "12 * 7"}},
    {"name": "get_user_status", "arguments": {"user_id": 1}}
]

for call in calls:
    print(call, "->", dispatch(call))
```

### このコードで分かること

このコードが教えてくれるのは、次の点です。

- 異なるツールでも、同じ呼び出し口を共有できる
- プログラム側でエラー処理をまとめられる
- 後からツールを増やしても構造が崩れにくい

---

## 種類ごとに何へ注意すべきか？

### 検索系ツール

重点ポイント:

- query を書き換える必要があるか
- 何件返すか
- 結果を rerank する必要があるか

### 計算系ツール

重点ポイント:

- 安全性
- 精度
- 式が正しいかどうか

安全な計算機の簡単な例:

```python
import ast
import operator

OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}


def safe_calculate(expression):
    def visit(node):
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](visit(node.left), visit(node.right))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -visit(node.operand)
        raise ValueError("unsupported_expression")

    return visit(ast.parse(expression, mode="eval"))


def safe_calculator(expression):
    allowed = set("0123456789+-*/(). ")
    if not set(expression) <= allowed:
        return {"error": "invalid_expression"}
    return {"result": safe_calculate(expression)}

print(safe_calculator("3 * (4 + 5)"))
print(safe_calculator("__import__('os').system('rm -rf /')"))
```

### データベース系ツール

重点ポイント:

- 権限
- パラメータの完全性
- クエリの境界

たとえば、モデルに任意の SQL を自由に生成させて、そのまま実行するのは避けるべきです。

### ファイル系ツール

重点ポイント:

- パスのホワイトリスト
- 書き込み権限
- 人の確認が必要かどうか

### 外部サービス系ツール

重点ポイント:

- タイムアウト
- リトライ
- 冪等性

---

## Agent らしいツール組み合わせの例

### シナリオ：ユーザーが返金できるか判断する

この処理には、次の 2 つのツールが必要かもしれません。

1. ユーザーの学習進捗を確認する
2. 返金ポリシーを確認する

```python
def refund_eligibility_agent(user_id):
    status = get_user_status(user_id)
    if "error" in status:
        return {"error": "ユーザーが存在しません"}

    policy = search_docs("返金")
    progress = status["progress"]

    can_refund = progress < 0.2
    return {
        "user": status["name"],
        "progress": progress,
        "policy": policy,
        "can_refund": can_refund
    }

print(refund_eligibility_agent(1))
print(refund_eligibility_agent(2))
```

### このコードが本当に示していること

このコードが示しているのは、

> ツール統合とは、各ツールを単独で置くことではなく、複数のツールを協力させて 1 つの目的を達成すること

という点です。

だからこそ、今後の Agent はツールのオーケストレーション能力にますます依存していきます。

---

## ツール統合でよくある失敗点

### schema が一致しない

たとえば:

- ツールは `user_id` を必要とする
- でもモデルは `id` を送ってしまう

### 戻り値の形式が統一されていない

あるツールは文字列、別のツールは dict、さらに別のツールは list を返すと、システムはだんだん接続しづらくなります。

### エラー処理が統一されていない

あるツールは `None` を返し、別のツールは例外を投げ、さらに別のツールは `"failed"` を返す。  
これでは後続の処理がすぐに崩れます。

### ログと再現手段がない

本番で問題が起きたとき、どの種類のツールに問題があったのか分かりません。

---

## 実用的な提案：ツールの戻り値形式を統一する

もっとも安定しやすい方法の1つは、ツールの出力構造を統一することです。たとえば、すべて次の形式にそろえます。

```python
{
  "ok": True,
  "data": ...
}
```

または:

```python
{
  "ok": False,
  "error": ...
}
```

簡単な例:

```python
def wrapped_search(keyword):
    try:
        result = search_docs(keyword)
        return {"ok": True, "data": result}
    except Exception as e:
        return {"ok": False, "error": str(e)}

print(wrapped_search("返金"))
```

こうしておくと、後の Agent 層で統一的に判定しやすくなります。

---

## 初心者がよくつまずくポイント

### とりあえず全部のツールをつなぐ

ツールが増えるほど、システムは複雑になります。  
より安定したやり方は、

- まず本当に必要な 2〜3 個だけつなぐ

ことです。

### 高リスクなツールと低リスクなツールを区別しない

ファイル削除、支払い操作、データベース書き込みは、ドキュメント検索とは危険度がまったく違います。

### ツール API の約束事が統一されていない

これは、Agent システムがだんだん混乱していく大きな原因の1つです。

---

## まとめ

この節で一番大切なのは、「どんなツールがあるか」を覚えることではなく、次の点を理解することです。

> **よく使うツール統合のポイントは、ツールを接続することだけではなく、それらを統一されたインターフェース、統一されたエラー処理、統一された境界制御でまとめることにある。**

こうして初めて、ツール層は Agent の能力を広げる存在になり、障害を増やす存在にはなりません。

---

## 練習

1. この節のツール登録表に `get_weather(city)` ツールを追加してください。
2. すべてのツールの戻り値を `{"ok": ..., "data": ..., "error": ...}` の形式に統一してください。
3. 考えてみましょう。なぜデータベース書き込みツールと検索ツールを同じ権限レベルにしてはいけないのでしょうか？
4. 自分の言葉で説明してください。なぜツール登録表と統一ディスパッチャが Agent 実装でとても重要な 2 つの構造だと言えるのでしょうか？
