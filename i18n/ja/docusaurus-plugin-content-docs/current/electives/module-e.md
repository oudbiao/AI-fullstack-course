---
title: "1.1 Web フロントエンド基礎速習"
sidebar_position: 5
description: "AI 製品の場面に向けて、フロントエンドの核心となる直感を素早く補う：ページ構造、状態更新、フォーム操作、非同期リクエスト、そして最小限で使える画面。"
keywords: [frontend, HTML, CSS, JavaScript, fetch, UI, AI product]
---

# Web フロントエンド基礎速習

![AI フロントエンドのインタラクションスタック図](/img/course/elective-ai-frontend-stack-ja.png)

![AI フロントエンドの状態マシンと体験の閉ループ図](/img/course/elective-ai-frontend-state-machine-map-ja.png)

:::tip 図の見方
AI フロントエンドは結果を表示するだけのページではなく、idle、loading、streaming、success、error、retry、cancel などの状態を扱います。図を見るときは、モデル呼び出しを「所要時間が不確定な」インタラクションの流れとして捉えましょう。
:::

:::tip この節の位置づけ
多くの AI 機能は、実は「作れない」のではなく、「受け止める良い画面がない」だけです。  
ユーザーが実際に触れるのは、たいていモデルそのものではなく、次のようなものです。

- 入力欄
- 履歴メッセージ
- 読み込み状態
- エラーメッセージ

そのため、このレッスンではフロントエンドを完全な学問として教えるのではなく、まず一番実用的な主線を補います。

> **AI 機能を、操作できて、理解できて、改善しやすい最小限のフロントエンドにするにはどう作るか。**
:::

## 学習目標

- AI 製品の実装におけるフロントエンドの基本的な役割を理解する
- HTML / CSS / JS の最小限の連携方法を身につける
- 非同期リクエストと状態更新の最小パターンを理解する
- 実行可能な静的ページの例を通して、画面構成の直感を作る

---

## 一、なぜ AI 製品はフロントエンドなしでは成り立たないのか？

### 1.1 「モデルの能力」はインタラクションを通して見える

どれだけ優れたモデルでも、分かりやすい画面がなければ、  
ユーザーは次のように感じます。

- 反応が遅い
- システムが何をしているのか分からない
- 結果が信頼できるのか分からない

### 1.2 フロントエンドの最も重要な価値

派手な見た目にすることではなく、  
次の状態をはっきり伝えることです。

- 今の入力は何か
- リクエストは送信されたか
- システムは読み込み中か、エラーか
- 出力結果をどう見ればよいか

### 1.3 たとえで考える

モデルはエンジン、フロントエンドはメーターとコックピットです。  
コックピットがなければ、どれだけ強いエンジンでもユーザーはスムーズに使えません。

---

## 二、まず最小限のフロントエンドの心のモデルを作る

### 2.1 HTML は構造を担当する

たとえば：

- 入力欄
- ボタン
- 結果エリア

### 2.2 CSS は見た目を担当する

たとえば：

- 余白
- レイアウト
- 状態ごとの見た目

### 2.3 JavaScript は動きを担当する

たとえば：

- クリックを監視する
- リクエストを送る
- ページ内容を更新する

この 3 層の関係が整理できれば、  
入門レベルの AI ページの多くは読み解けるようになります。

---

## 三、まずは最小限の AI ページを動かしてみる

以下の例では、非常に典型的なことを 1 つ行います。

- 質問を入力する
- ボタンをクリックする
- 「読み込み中」を表示する
- そのあと結果をページに描画する

これは純粋なフロントエンドの模擬で、バックエンドに依存せず、ブラウザでそのまま開けます。

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AI Demo</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 0;
      background: linear-gradient(180deg, #f8fafc, #eef2ff);
      color: #0f172a;
    }
    .container {
      max-width: 720px;
      margin: 48px auto;
      padding: 24px;
      background: white;
      border-radius: 16px;
      box-shadow: 0 16px 40px rgba(15, 23, 42, 0.08);
    }
    h1 {
      margin-top: 0;
      font-size: 28px;
    }
    textarea {
      width: 100%;
      min-height: 120px;
      padding: 12px;
      border: 1px solid #cbd5e1;
      border-radius: 12px;
      font-size: 16px;
      box-sizing: border-box;
      resize: vertical;
    }
    button {
      margin-top: 12px;
      padding: 10px 16px;
      border: none;
      border-radius: 999px;
      background: #0f172a;
      color: white;
      cursor: pointer;
      font-size: 15px;
    }
    .status {
      margin-top: 16px;
      color: #475569;
      font-size: 14px;
    }
    .result {
      margin-top: 20px;
      padding: 16px;
      border-radius: 12px;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      line-height: 1.6;
      min-height: 72px;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>AI 機能の最小ページ</h1>
    <textarea id="prompt" placeholder="質問を入力してください。例：返金ルールは何ですか？"></textarea>
    <button id="runBtn">送信</button>
    <div id="status" class="status">入力待ち</div>
    <div id="result" class="result">結果はここに表示されます</div>
  </div>

  <script>
    const promptInput = document.getElementById("prompt");
    const runBtn = document.getElementById("runBtn");
    const statusEl = document.getElementById("status");
    const resultEl = document.getElementById("result");

    function fakeAiResponse(prompt) {
      return new Promise((resolve) => {
        setTimeout(() => {
          resolve("模擬回答：" + prompt + "。ここは実際の API の返却値に置き換えられます。");
        }, 700);
      });
    }

    runBtn.addEventListener("click", async () => {
      const prompt = promptInput.value.trim();
      if (!prompt) {
        statusEl.textContent = "先に内容を入力してください";
        return;
      }

      statusEl.textContent = "リクエスト処理中...";
      resultEl.textContent = "読み込み中...";
      runBtn.disabled = true;

      try {
        const answer = await fakeAiResponse(prompt);
        statusEl.textContent = "リクエスト完了";
        resultEl.textContent = answer;
      } catch (error) {
        statusEl.textContent = "リクエスト失敗";
        resultEl.textContent = "システムにエラーが発生しました。しばらくしてからもう一度お試しください。";
      } finally {
        runBtn.disabled = false;
      }
    });
  </script>
</body>
</html>
```

### 3.1 この例でいちばん学ぶべきことは？

フロントエンドの最小限の閉ループが、とても分かりやすく表現されています。

1. 入力を取得する
2. 動作を引き起こす
3. 読み込み状態を表示する
4. 結果エリアを更新する

### 3.2 なぜ「読み込み中」がそんなに重要なのか？

多くの AI リクエストは一瞬では返ってきません。  
もしフロントエンドに何の反応もなければ、ユーザーは次のように疑います。

- 固まったのではないか
- 押せていなかったのではないか

そのため、状態のフィードバック自体が体験の一部です。

### 3.3 ここで `fakeAiResponse` を使う理由は？

まずはフロントエンドの流れを理解するためです。  
あとで本物のバックエンド API ができたら、この部分を `fetch` に置き換えるだけで済みます。

---

## 四、ローカル模擬から本物の API に進むときはどうする？

### 4.1 `fetch` でリクエストを送る

もっとも一般的な次の一歩は、次のようになります。

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ prompt }),
});
const data = await response.json();
```

### 4.2 このときフロントエンドがより気にするべきことは？

- リクエストが成功したか
- 返り値が空ではないか
- エラーをどう表示するか
- ユーザーが続けて操作できるか

### 4.3 フロントエンドは「結果を表示するだけ」ではない

実際には、次の役割も担っています。

- ユーザー入力の検証
- インタラクションのテンポ設計
- 例外メッセージの表示

---

## 五、AI フロントエンドでよくある落とし穴

### 5.1 誤解その一：先に複雑な UI を作ってからバックエンドにつなぐ

より安全な順番は、通常次の通りです。

- まず最小限で動くものを作る
- そのあと少しずつ強化する

### 5.2 誤解その二：読み込み状態とエラー状態がない

これでは体験がとても悪くなり、  
しかも問題の切り分けも難しくなります。

### 5.3 誤解その三：フロントエンドを「飾り」と考える

AI 製品では、フロントエンドが実は非常に重要な次の役割を担います。

- 使い方の誘導
- フィードバックの表現
- 信頼の構築

---

## まとめ

この節で最も大事なのは、実用的なフロントエンド観を持つことです。

> **AI フロントエンドの最初の目標は派手さではなく、入力・処理中・結果・エラーという 4 つの状態をはっきり表現し、モデルの能力をユーザーがスムーズに使えるようにすることです。**

この理解がしっかりしていれば、あとで React を学んだり、より複雑な製品を作ったりするときも、ずっと自然に進められます。

---

## 練習

1. サンプルに「入力欄をクリアする」ボタンを追加してみましょう。
2. `fakeAiResponse` を成功と失敗をランダムに返すように変更し、エラー状態の表示を練習しましょう。
3. 考えてみましょう：なぜ AI 製品では「読み込みのフィードバック」が多くの普通の静的ページより重要なのでしょうか？
4. この最小ページを、チャット画面に拡張するならどう設計しますか？
