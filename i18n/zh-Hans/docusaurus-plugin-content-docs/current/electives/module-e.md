---
title: "E.E Web 前端快速入门"
sidebar_position: 5
description: "搭建最小 AI 前端闭环：输入、加载、成功、空输入、错误和重试。"
keywords: [frontend, HTML, CSS, JavaScript, fetch, UI, AI product]
---

# E.E Web 前端快速入门

AI 功能需要用户能操作的界面。最小可用前端要清楚展示输入、加载、成功、空输入、错误和重试状态。

## 先看交互栈

![AI 前端交互栈图](/img/course/elective-ai-frontend-stack.webp)

![AI 前端状态机与体验闭环图](/img/course/elective-ai-frontend-state-machine-map.webp)

把每次模型调用都当成不确定请求：可能慢、失败、返回部分内容，或者需要重试。

## 运行最小静态 AI 页面

保存为 `ai-demo.html`，然后用浏览器打开：

```html
<!doctype html>
<html lang="en">
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
<textarea id="prompt" placeholder="Ask a question"></textarea>
<button id="run">Run</button>
<p id="status">idle</p>
<pre id="result">result appears here</pre>
<script>
  const promptEl = document.querySelector("#prompt");
  const statusEl = document.querySelector("#status");
  const resultEl = document.querySelector("#result");

  document.querySelector("#run").addEventListener("click", async () => {
    const text = promptEl.value.trim();
    if (!text) {
      statusEl.textContent = "empty";
      resultEl.textContent = "Please enter a prompt first.";
      return;
    }

    try {
      statusEl.textContent = "loading";
      resultEl.textContent = "Please wait...";
      await new Promise((resolve) => setTimeout(resolve, 500));
      if (text.toLowerCase().includes("fail")) {
        throw new Error("simulated backend error");
      }
      statusEl.textContent = "success";
      resultEl.textContent = "Simulated answer: " + text;
    } catch (error) {
      statusEl.textContent = "error";
      resultEl.textContent = error.message + ". Try again.";
    }
  });
</script>
</html>
```

预期浏览器行为：

- 空输入：状态变成 `empty`。
- 普通输入：状态从 `loading` 变成 `success`。
- 输入包含 `fail`：状态变成 `error`。

## 之后替换成真实调用

后端准备好后，把模拟等待替换成 `fetch`：

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({prompt: text}),
});
const data = await response.json();
resultEl.textContent = data.answer;
```

## 通过标准

一个 AI 页面能清楚处理输入、加载、成功、空输入、错误和重试状态，且不会让用户迷惑，就算通过本选修。

## 留下的证据

学完这一页，至少保留这张证据卡：

```text
frontend_surface: chat, dashboard, editor, review panel, or workflow UI
state_model: loading, streaming, success, empty, error, retry, and review states
artifact: UI sketch, component behavior, event trace, or screenshot
failure_check: hiding latency, missing error states, unclear citations, or weak review controls
Expected_output: AI frontend interaction note with states and evidence display
```
