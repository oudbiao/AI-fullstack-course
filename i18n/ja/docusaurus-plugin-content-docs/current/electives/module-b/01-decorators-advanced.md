---
title: "E.B.1 デコレータの応用"
sidebar_position: 8
description: "ログ、リトライ、計測、権限チェックをデコレータでまとめ、各関数への重複を減らす。"
keywords: [decorators, Python, wraps, retry, logging, timing, authorization]
---

# E.B.1 デコレータの応用

![Python デコレータ実行フロー図](/img/course/elective-python-decorator-flow-ja.webp)

![デコレータによる横断的ロジックの階層図](/img/course/elective-decorator-crosscutting-layers-ja.webp)

デコレータは、関数の外側に再利用できる動作を巻き付けます。ログ、計測、リトライ、権限チェック、トレースのように、多くの関数で同じ処理が必要なときに向いています。

## 準備するもの

- Python 3.10+
- 外部パッケージ不要
- 関数の基本理解

## 重要用語

- **Wrapper（ラッパー）**：元の関数の前後で実際に動く内側の関数。
- **Cross-cutting logic（横断的ロジック）**：多くの場所で必要だが、業務処理そのものではないロジック。
- **`functools.wraps`**：デコレート後も元の関数名やメタ情報を残す。
- **デコレータの順序**：関数呼び出し時は、一番上のデコレータから実行される。

## ログとリトライのデコレータを動かす

`decorator_demo.py` を作成します。

```python
from functools import wraps


def log_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print(f"[LOG] start {fn.__name__}")
        result = fn(*args, **kwargs)
        print(f"[LOG] end {fn.__name__}")
        return result

    return wrapper


def retry(max_retries=2):
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(1, max_retries + 2):
                try:
                    return fn(*args, **kwargs)
                except RuntimeError as error:
                    last_error = error
                    print(f"[RETRY] attempt={attempt} error={error}")
            raise last_error

        return wrapper

    return decorator


state = {"attempt": 0}


@log_call
@retry(max_retries=2)
def fetch_model_info(model_id):
    state["attempt"] += 1
    if state["attempt"] < 2:
        raise RuntimeError("temporary network error")
    return {"model_id": model_id, "status": "ready"}


print(fetch_model_info("demo-v1"))
print(fetch_model_info.__name__)
```

実行します。

```bash
python decorator_demo.py
```

期待される出力：

```text
[LOG] start fetch_model_info
[RETRY] attempt=1 error=temporary network error
[LOG] end fetch_model_info
{'model_id': 'demo-v1', 'status': 'ready'}
fetch_model_info
```

この例では、業務関数を短く保ち、リトライを一箇所にまとめ、`wraps` で関数名を維持できることが分かります。

## 順序を変えてみる

デコレータを次の順に入れ替えます。

```text
@retry(max_retries=2)
@log_call
def fetch_model_info(model_id):
```

この場合、ログは各リトライの内側で実行されます。サービスコードでは、デコレータの順序が重要です。

## デコレータを使う場面

向いているもの：

1. ログとトレース
2. 実行時間の計測
3. 不安定な I/O のリトライ
4. 権限チェック
5. フレームワークへの登録

ラッパーが重要な業務ロジックを隠してしまう場合や、すでに層が多すぎる関数には向きません。

## よくある間違い

- `@wraps` を忘れ、ログやフレームワークからすべて `wrapper` に見えてしまう。
- 検証エラーや権限エラーのように即失敗すべき例外までリトライする。
- デコレータを積みすぎて、実行順序を追いにくくする。

## 練習

`fetch_model_info` の前に `require_role("admin")` デコレータを追加してください。admin 以外なら `PermissionError` を出し、権限エラーはリトライしないようにします。
