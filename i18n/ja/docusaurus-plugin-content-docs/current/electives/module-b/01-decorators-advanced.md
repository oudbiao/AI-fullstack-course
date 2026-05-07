---
title: "E.B.1 装飾器の応用"
sidebar_position: 8
description: "ログ、計測、リトライ、権限制御といった実務でよく出る要件から、装飾器がなぜ Python のサービスコードで頻出パターンなのかを理解します。"
keywords: [decorators, Python, wraps, retry, logging, timing, authorization]
---

# E.B.1 装飾器の応用

![Python 装飾器実行フローチャート](/img/course/elective-python-decorator-flow-ja.png)

![装飾器の横断ロジック分層図](/img/course/elective-decorator-crosscutting-layers-ja.png)

:::tip 図の見方
装飾器は、ログ、計測、リトライ、権限のような横断ロジックをまとめるのに最適です。図を見るときは、wrapper が元の関数をどう包んでいるか、そしてなぜ `functools.wraps` が関数の身元情報を保って、デバッグやフレームワークの認識エラーを防げるのかに注目してください。
:::

:::tip この節の位置づけ
装飾器を初めて学ぶとき、多くの人は次のような印象を持ちます。

- 文法が少しややこしい
- なんだか上級っぽい

でも、実務では装飾器の本当の価値はとてもシンプルです。

> **「横断ロジック」をまとめて包み、各関数に同じ処理を何度も書かないようにすること。**

よくある横断ロジックは次のようなものです。

- ログ
- 計測
- リトライ
- 権限チェック

そのため、この講義では装飾器を「派手な文法」としてではなく、実務上の問題に結びつけて見ていきます。
:::

## 学習目標

- 装飾器が実務でよく使われる用途を理解する
- 引数付き装飾器と、メタ情報を保持する装飾器の書き方を学ぶ
- `functools.wraps` が重要な理由を理解する
- 実行できる例を通して、ログ・計測・リトライという3つの代表パターンを身につける

---

## 一、なぜ装飾器は実務でこんなに使われるのか？

### 多くの処理が、いろいろな関数の外側に繰り返し付くから

たとえば、多くの関数で次のような処理が必要になります。

- ログを出す
- 処理時間を計測する
- 例外を捕まえる
- 権限を確認する

これを各関数で毎回手書きすると、コードはすぐに重複だらけになります。

### 装飾器の本質的な価値

装飾器は「関数を魔法のように書き換える」ものではなく、次のようなものです。

- 関数を受け取る
- その外側に共通処理を1枚かぶせる
- 新しい関数を返す

つまり、

> **装飾器は「関数の外側の振る舞い」を再利用する仕組みです。**

### たとえで考える

関数は、実際に仕事をする人のようなものです。  
装飾器は、その人たち全員に共通で付ける

- 社員証
- 打刻
- 入退室チェック
- タイマー

のようなものです。

人そのものは変わりませんが、仕事の流れは統一されます。

---

## 二、まずは最もよくあるログ装飾器から

次のコードでは、次のことを示します。

- 呼び出し前後にログを出す
- 元の関数のメタ情報を保持する

```python
from functools import wraps


def log_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[LOG] calling {fn.__name__} args={args} kwargs={kwargs}")
        result = fn(*args, **kwargs)
        print(f"[LOG] {fn.__name__} returned {result}")
        return result

    return wrapper


@log_call
def add(a, b):
    return a + b


print(add(3, 5))
print(add.__name__)
```

### `wraps` はなぜ重要なのか？

`@wraps(fn)` がないと、  
装飾後の関数のメタ情報が失われることがあります。たとえば：

- `__name__`
- `__doc__`

これは次のような場面に影響します。

- デバッグ
- ログ
- ドキュメント生成
- 一部のフレームワークの挙動

### なぜログ装飾器はよく使われるのか？

これは非常に典型的な横断ロジックだからです。

- ビジネスロジックそのものとは関係がない
- でも多くの関数で必要になる

---

## 三、計測装飾器：性能問題をまず見える化する

実務の問題は「機能が間違っている」ことではなく、

- 遅い

ことだったりします。

計測装飾器があると、ボトルネックを素早く見つけやすくなります。

```python
import time
from functools import wraps


def measure_time(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - started
        print(f"[TIME] {fn.__name__} took {elapsed:.4f}s")
        return result

    return wrapper


@measure_time
def fake_inference():
    time.sleep(0.2)
    return "done"


print(fake_inference())
```

### AI エンジニアリングでかなり便利

たとえば、次の処理にどれくらい時間がかかっているかをすぐに見たいとします。

- tokenizer
- 検索
- モデル推論

そのたびに、各関数へ手で計測コードを書く必要はありません。

---

## 四、引数付き装飾器：なぜ実務により近いのか？

実務では、単に「この処理を有効にするかどうか」だけでなく、次のような指定もしたくなります。

- 何回リトライするか
- 必要な権限レベルは何か
- タイムアウトのしきい値はどれくらいか

そのために引数付き装飾器を使います。

```python
from functools import wraps


def require_role(role):
    def decorator(fn):
        @wraps(fn)
        def wrapper(user, *args, **kwargs):
            if user.get("role") != role:
                raise PermissionError(f"required role: {role}")
            return fn(user, *args, **kwargs)

        return wrapper

    return decorator


@require_role("admin")
def delete_model(user, model_name):
    return f"deleted:{model_name}"


admin = {"name": "alice", "role": "admin"}
guest = {"name": "bob", "role": "guest"}

print(delete_model(admin, "v1"))
try:
    print(delete_model(guest, "v1"))
except PermissionError as e:
    print("error:", e)
```

### なぜここでは3層構造になるのか？

理由は次の通りです。

- 第1層: 装飾器の引数を受け取る
- 第2層: 装飾される関数を受け取る
- 第3層: 実際の呼び出し時に動く包装ロジック

最初は少しややこしく感じますが、  
この3層の役割を覚えると混乱しにくくなります。

---

## 五、リトライ装飾器：本番コードで非常によくあるパターン

次の例では、不安定な関数をシミュレートし、  
装飾器でリトライ処理をまとめます。

```python
from functools import wraps


def retry(max_retries=2):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries + 1):
                try:
                    return fn(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    print(f"[RETRY] attempt={attempt + 1} error={e}")
            raise last_error

        return wrapper

    return decorator


state = {"count": 0}


@retry(max_retries=2)
def flaky_call():
    state["count"] += 1
    if state["count"] < 3:
        raise RuntimeError("temporary error")
    return "ok"


print(flaky_call())
```

### このコードの実務上の意味は？

装飾器が次のような周辺制御ロジックをまとめるのに向いていることがわかります。

- リトライ
- レート制限
- サーキットブレーカー

### なぜリトライ装飾器は慎重に使うべきか？

すべてのエラーがリトライに向いているわけではないからです。  
たとえば：

- 引数エラー
- 権限エラー

これらをリトライしても、無駄にリソースを消費するだけです。

---

## 六、装飾器でよくある落とし穴

### 誤解その1：装飾器を見るとすぐ「上級者向け」と思ってしまう

装飾器は見せびらかすための技術ではなく、  
重複を減らすための仕組みです。

### 誤解その2：装飾器を何層も重ねすぎる

1つの関数に装飾器がたくさん付くと、  
デバッグも理解も難しくなります。

### 誤解その3：`wraps` を使わない

これをするとメタ情報が失われ、後で原因調査がとても大変になります。

---

## まとめ

この節で最も大事なのは、装飾器を文法の暗記問題として覚えることではなく、  
次のような実務的な判断を持つことです。

> **装飾器は、ログ・計測・リトライ・権限のような横断ロジックをまとめて、業務ロジック本体をすっきり保つのに最適です。**

この考え方が身につくと、  
今後フレームワークのコード、サービスのミドルウェア、ライブラリのソースを読むときにずっと理解しやすくなります。

---

## 練習

1. `measure_time` に引数 `label` を追加して、引数付き装飾器を練習してみましょう。
2. どのようなエラーが `retry` 装飾器に向いていて、どのようなエラーが向いていないか考えてみましょう。
3. ログ、計測、リトライの3つの装飾器を1つの関数に重ねて、実行順を観察してみましょう。
4. 自分の言葉で説明してみましょう。なぜ装飾器は「横断ロジック」に特に向いているのでしょうか。
