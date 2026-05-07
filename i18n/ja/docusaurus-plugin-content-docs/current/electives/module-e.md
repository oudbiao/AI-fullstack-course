---
title: "E.E Web フロントエンド基礎ファストトラック"
sidebar_position: 5
description: "AI プロダクトに必要な最小フロントエンドループの短い実践ガイド。入力、読み込み、結果、エラー、再試行状態を扱います。"
keywords: [frontend, HTML, CSS, JavaScript, fetch, UI, AI product]
---

# E.E Web フロントエンド基礎ファストトラック

AI 機能には、ユーザーが操作できる画面が必要です。最小限でも役立つフロントエンドは、入力、読み込み中、成功、エラー、再試行の状態をはっきり見せます。

## まずインタラクションの層を見る

![AI フロントエンドのインタラクション層図](/img/course/elective-ai-frontend-stack-ja.png)

![AI フロントエンドの状態機械と体験ループ図](/img/course/elective-ai-frontend-state-machine-map-ja.png)

モデル呼び出しは不確実なリクエストだと考えます。遅いことも、失敗することも、途中結果を返すことも、再試行が必要なこともあります。

## 最小の静的 AI ページを動かす

`ai-demo.html` として保存し、ブラウザで開きます。

```html
<!doctype html>
<html lang="ja">
<meta charset="utf-8" />
<title>AI Demo</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; }
  textarea, button, pre { width: 100%; box-sizing: border-box; margin-top: 12px; }
  textarea { min-height: 100px; padding: 12px; }
  button { padding: 10px; }
  pre { min-height: 80px; padding: 12px; background: #f3f4f6; white-space: pre-wrap; }
</style>
<h1>AI Feature Demo</h1>
<textarea id="prompt" placeholder="質問を入力"></textarea>
<button id="run">Run</button>
<p id="status">idle</p>
<pre id="result">result appears here</pre>
<script>
  const promptEl = document.querySelector("#prompt");
  const statusEl = document.querySelector("#status");
  const resultEl = document.querySelector("#result");

  document.querySelector("#run").addEventListener("click", async () => {
    statusEl.textContent = "loading";
    resultEl.textContent = "Please wait...";
    await new Promise((resolve) => setTimeout(resolve, 500));
    statusEl.textContent = "success";
    resultEl.textContent = "Simulated answer: " + promptEl.value.trim();
  });
</script>
</html>
```

期待されるブラウザ動作: 質問を入力して `Run` を押すと、状態が `idle` から `loading`、`success` に変わり、結果欄が更新されます。

## あとで仮の呼び出しを置き換える

バックエンドができたら、シミュレーション用の待ち時間を `fetch` に置き換えます。

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({prompt: promptEl.value}),
});
const data = await response.json();
resultEl.textContent = data.answer;
```

## 合格チェック

1 つの AI 機能について、入力、読み込み中、成功、空入力、エラー状態を画面上で明確に扱えれば合格です。
