---
title: "E.E Web 前端基础速通"
sidebar_position: 5
description: "AI 产品所需最小前端循环的简明实操指南：输入、加载、结果、错误和重试状态。"
keywords: [前端, HTML, CSS, JavaScript, fetch, UI, AI 产品]
---

# E.E Web 前端基础速通

AI 功能需要一个用户能操作的界面。最小可用前端要清楚展示输入、加载、成功、错误和重试状态。

## 先看交互栈

![AI 前端交互栈图](/img/course/elective-ai-frontend-stack.png)

![AI 前端状态机与体验闭环图](/img/course/elective-ai-frontend-state-machine-map.png)

把每次模型调用都当成不确定请求：它可能很慢、失败、流式返回，也可能需要重试。

## 跑最小静态 AI 页面

保存为 `ai-demo.html`，然后用浏览器打开：

```html
<!doctype html>
<html lang="zh-CN">
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
<textarea id="prompt" placeholder="输入问题"></textarea>
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

预期浏览器行为：输入问题并点击 `Run` 后，状态从 `idle` 变成 `loading` 再变成 `success`，结果区域会更新。

## 后续替换成真实调用

后端准备好后，把模拟等待替换成 `fetch`：

```javascript
const response = await fetch("/api/chat", {
  method: "POST",
  headers: {"Content-Type": "application/json"},
  body: JSON.stringify({prompt: promptEl.value}),
});
const data = await response.json();
resultEl.textContent = data.answer;
```

## 通过标准

你的页面能为一个 AI 功能清楚处理输入、加载、成功、空输入和错误状态，就算通过本选修。
