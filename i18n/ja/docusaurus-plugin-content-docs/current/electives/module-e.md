---
title: "E.E Web フロントエンド速習"
sidebar_position: 5
description: "最小の AI フロントエンドループを作る。入力、読み込み、成功、空入力、エラー、再試行を扱う。"
keywords: [frontend, HTML, CSS, JavaScript, fetch, UI, AI product]
---

# E.E Web フロントエンド速習

AI 機能には、ユーザーが操作できる画面が必要です。最小限の有用なフロントエンドは、入力、読み込み、成功、空入力、エラー、再試行をはっきり示します。

## まずインタラクションスタックを見る

![AI フロントエンド インタラクションスタック図](/img/course/elective-ai-frontend-stack-ja.webp)

![AI フロントエンド状態機械と体験ループ図](/img/course/elective-ai-frontend-state-machine-map-ja.webp)

モデル呼び出しは常に不確実なリクエストだと考えます。遅い、失敗する、一部だけ返る、再試行が必要になることがあります。

## 最小の静的 AI ページを動かす

`ai-demo.html` として保存し、ブラウザで開きます。

```html
<!doctype html>
<html lang="ja">
<meta charset="utf-8" />
<title>AI デモ</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; }
  textarea, button, pre { width: 100%; box-sizing: border-box; margin-top: 12px; }
  textarea { min-height: 100px; padding: 12px; }
  button { padding: 10px; }
  pre { min-height: 80px; padding: 12px; background: #f3f4f6; white-space: pre-wrap; }
</style>
<h1>AI 機能デモ</h1>
<textarea id="prompt" placeholder="質問を入力"></textarea>
<button id="run">実行</button>
<p id="status">idle</p>
<pre id="result">結果がここに表示されます</pre>
<script>
  const promptEl = document.querySelector("#prompt");
  const statusEl = document.querySelector("#status");
  const resultEl = document.querySelector("#result");

  document.querySelector("#run").addEventListener("click", async () => {
    const text = promptEl.value.trim();
    if (!text) {
      statusEl.textContent = "empty";
      resultEl.textContent = "先にプロンプトを入力してください。";
      return;
    }

    try {
      statusEl.textContent = "loading";
      resultEl.textContent = "お待ちください...";
      await new Promise((resolve) => setTimeout(resolve, 500));
      if (text.toLowerCase().includes("fail")) {
        throw new Error("simulated backend error");
      }
      statusEl.textContent = "success";
      resultEl.textContent = "シミュレーション回答：" + text;
    } catch (error) {
      statusEl.textContent = "error";
      resultEl.textContent = error.message + "。もう一度試してください。";
    }
  });
</script>
</html>
```

期待されるブラウザの動き：

- 空入力：状態が `empty` になる。
- 通常入力：状態が `loading` から `success` になる。
- `fail` を含む入力：状態が `error` になる。

## 後で本物の呼び出しに置き換える

バックエンドができたら、擬似的な待ち時間を `fetch` に置き換えます。

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({prompt: text}),
});
const data = await response.json();
resultEl.textContent = data.answer;
```

## 残す証拠

このページを終えたら、この証拠カードを残します。

```text
フロントエンド画面：チャット、ダッシュボード、エディタ、レビュー画面、またはワークフロー UI
状態モデル：loading、streaming、success、empty、error、retry、review の状態
成果物: UI スケッチ、コンポーネントの挙動、イベントトレース、またはスクリーンショット
失敗確認：遅延の隠蔽、エラー状態の不足、不明確な引用、または弱いレビュー管理
期待される成果：状態と証拠表示を含む AI フロントエンド対話メモ
```

## 合格チェック

1つの AI ページが、入力、読み込み、成功、空入力、エラー、再試行を迷わず扱えるなら合格です。

<details>
<summary>確認の考え方と解説</summary>

合格する答えは、少なくとも 3 つの経路をカバーします。空入力、通常入力、そして失敗のシミュレーションです。証拠としては、小さな表やスクリーンショットが良く、状態、メッセージ、再試行の様子が見える必要があります。

大事なのは、何が起きているのかをユーザーが明確に理解できることです。結果ボックスを 1 つ置くだけでは足りません。

</details>
